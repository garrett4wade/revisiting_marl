from collections import defaultdict
from tensorboardX import SummaryWriter
import logging
import multiprocessing as mp
import numpy as np
import os
import queue
import time
import torch
import wandb

from algorithm.trainers.mappo import MAPPO
from algorithm.policy import RolloutRequest, RolloutResult
from algorithm.trainer import SampleBatch
from algorithm.modules import gae_trace, masked_normalization
from utils.namedarray import recursive_apply, array_like, recursive_aggregate
from utils.timing import Timing

logger = logging.getLogger('shared_runner')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('log.txt', mode='a')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


class SharedRunner:

    def __init__(self,
                 all_args,
                 policy,
                 storages,
                 env_ctrls,
                 info_queue,
                 eval_storages,
                 eval_env_ctrls,
                 eval_info_queue,
                 device,
                 run_dir=None):

        self.all_args = all_args
        self.device = device
        self.num_agents = all_args.num_agents

        # parameters
        self.env_name = self.all_args.env_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.num_train_envs = self.all_args.num_train_envs
        self.num_eval_envs = self.all_args.num_eval_envs
        self.num_env_splits = self.all_args.num_env_splits
        # interval
        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # TODO: wandb mode
        if not (all_args.eval or all_args.render):
            if self.all_args.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = run_dir
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writer = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        self.storages = storages
        self.policy = policy
        self.env_ctrls = env_ctrls
        self.info_queue = info_queue

        self.eval_storages = eval_storages
        self.eval_env_ctrls = eval_env_ctrls
        self.eval_info_queue = eval_info_queue

        if self.model_dir is not None:
            self.restore()

        self.trainer = MAPPO(self.all_args, self.policy)

    def run(self):
        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.num_train_envs

        for episode in range(episodes):
            timing = Timing()

            train_ep_ret = train_ep_length = train_ep_cnt = 0

            for step in range(self.episode_length):
                for s_i in range(self.num_env_splits):
                    storage = self.storages[s_i]
                    assert step == storage.step

                    # Sample actions
                    with timing.add_time("envstep"):
                        for ctrl in self.env_ctrls[s_i]:
                            ctrl.obs_ready.acquire()
                            assert not ctrl.obs_ready.acquire(block=False)

                    with timing.add_time("inference"):
                        rollout_result = self.collect(s_i, step)

                    with timing.add_time("storage"):
                        storage.value_preds[step] = rollout_result.value
                        storage.actions[step] = rollout_result.action.float()
                        storage.action_log_probs[
                            step] = rollout_result.log_prob
                        if storage.policy_state is not None:
                            storage.policy_state[
                                step + 1] = rollout_result.policy_state

                        storage.step += 1

                    for ctrl in self.env_ctrls[s_i]:
                        assert not ctrl.act_ready.acquire(block=False)
                        ctrl.act_ready.release()

            with timing.add_time("gae"):
                for s_i in range(self.num_env_splits):
                    for ctrl in self.env_ctrls[s_i]:
                        ctrl.obs_ready.acquire()

                    storage = self.storages[s_i]

                    assert storage.step == self.episode_length
                    rollout_result = self.collect(s_i, self.episode_length)
                    storage.value_preds[
                        self.episode_length] = rollout_result.value

                sample = recursive_aggregate(self.storages,
                                             lambda x: torch.cat(x, dim=1))
                if self.policy.popart_head is not None:
                    trace_target_value = self.policy.denormalize_value(
                        sample.value_preds.to(self.device)).cpu()
                else:
                    trace_target_value = sample.value_preds
                adv = gae_trace(sample.rewards, trace_target_value,
                                sample.masks, self.all_args.gamma,
                                self.all_args.gae_lambda, sample.bad_masks)
                sample.returns = adv + trace_target_value
                sample.advantages = adv
                sample.advantages[:-1] = masked_normalization(
                    adv[:-1], mask=sample.active_masks[:-1])

            with timing.add_time("train"):
                train_infos = self.train(sample)

            for s_i, storage in enumerate(self.storages):
                storage[0] = storage[-1]
                # storage.step must be changed via inplace operations
                assert storage.step == self.episode_length
                storage.step %= self.episode_length

                for ctrl in self.env_ctrls[s_i]:
                    ctrl.obs_ready.release()

            logger.debug(timing)

            while True:
                try:
                    info = self.info_queue.get_nowait()
                    train_ep_ret += info['episode']['r']
                    train_ep_length += info['episode']['l']
                    train_ep_cnt += 1
                except queue.Empty:
                    break

            # post process
            total_num_steps = (episode +
                               1) * self.episode_length * self.num_train_envs
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                logger.info(
                    "Updates {}/{} episodes, total num timesteps {}/{}, FPS {}."
                    .format(episode, episodes, total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

                if train_ep_cnt > 0:
                    train_env_info = dict(
                        train_episode_length=train_ep_length / train_ep_cnt,
                        train_episode_return=train_ep_ret / train_ep_cnt)
                    train_infos = {**train_env_info, **train_infos}

                self.log_info(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 or episode == episodes - 1:
                self.eval(total_num_steps)

    @torch.no_grad()
    def collect(self, split, step) -> RolloutResult:
        self.trainer.prep_rollout()
        storage = self.storages[split]
        request = RolloutRequest(
            storage.obs[step], storage.policy_state[step]
            if storage.policy_state is not None else None, storage.masks[step])
        request = recursive_apply(
            request, lambda x: x.flatten(end_dim=1).to(self.device))
        rollout_result = self.policy.rollout(request, deterministic=False)
        return recursive_apply(
            rollout_result,
            lambda x: x.view(self.num_train_envs // self.num_env_splits, self.
                             num_agents, *x.shape[1:]).cpu())

    def train(self, sample):
        train_infos = defaultdict(lambda: 0)
        self.trainer.prep_training()
        for _ in range(self.all_args.sample_reuse):
            train_info = self.trainer.train(
                recursive_apply(sample[:-1], lambda x: x.to(self.device)))
            for k, v in train_info.items():
                train_infos[k] += v
        self.policy.inc_version()

        return {
            k: float(v / self.all_args.sample_reuse)
            for k, v in train_infos.items()
        }

    def eval(self, step):
        for s_i in range(self.num_env_splits):
            for ctrl in self.eval_env_ctrls[s_i]:
                ctrl.eval_finish.clear()
                ctrl.eval_start.set()
                while ctrl.act_ready.acquire(block=False):
                    continue
                while ctrl.obs_ready.acquire(block=False):
                    continue

        self.trainer.prep_rollout()
        if self.eval_storages[0].policy_state is not None:
            policy_states = [
                array_like(eval_storage.policy_state, default_value=0)
                for eval_storage in self.eval_storages
            ]
        else:
            policy_states = [None for _ in range(self.num_env_splits)]
        eval_ep_cnt = eval_ep_len = eval_ep_ret = 0
        s_i = 0

        while eval_ep_cnt < self.all_args.eval_episodes:
            for ctrl in self.eval_env_ctrls[s_i]:
                ctrl.obs_ready.acquire()

            eval_storage = self.eval_storages[s_i]

            while True:
                try:
                    info = self.eval_info_queue.get_nowait()
                    eval_ep_ret += info['episode']['r']
                    eval_ep_len += info['episode']['l']
                    eval_ep_cnt += 1
                    if eval_ep_cnt >= self.all_args.eval_episodes:
                        break
                except queue.Empty:
                    break

            request = RolloutRequest(eval_storage.obs, policy_states[s_i],
                                     eval_storage.masks)
            request = recursive_apply(
                request, lambda x: x.flatten(end_dim=1).to(self.device))
            rollout_result = recursive_apply(
                self.policy.rollout(request, deterministic=True),
                lambda x: x.view(self.num_eval_envs // self.num_env_splits, self
                                 .num_agents, *x.shape[1:]).cpu())

            eval_storage.actions[:] = rollout_result.action.float()
            policy_states[s_i] = rollout_result.policy_state

            for ctrl in self.eval_env_ctrls[s_i]:
                ctrl.act_ready.release()

            s_i = (s_i + 1) % self.num_env_splits

        for s_i in range(self.num_env_splits):
            for ctrl in self.eval_env_ctrls[s_i]:
                ctrl.eval_start.clear()
                ctrl.eval_finish.set()

        eval_info = dict(eval_episode_return=eval_ep_ret / eval_ep_cnt,
                         eval_episode_length=eval_ep_len / eval_ep_cnt)
        self.log_info(eval_info, step)

    def save(self):
        torch.save(self.policy.get_checkpoint(),
                   os.path.join(str(self.save_dir), "model.pt"))

    def restore(self):
        checkpoint = torch.load(os.path.join(str(self.model_dir), "model.pt"))
        self.policy.load_checkpoint(checkpoint)
        logger.info(f"Loaded checkpoint from {self.model_dir}.")

    def log_info(self, infos, step):
        logger.info('-' * 40)
        for k, v in infos.items():
            key = ' '.join(k.split('_')).title()
            logger.info("{}: \t{:.2f}".format(key, float(v)))
        logger.info('-' * 40)

        if not self.all_args.eval:
            if self.all_args.use_wandb:
                wandb.log(infos, step=step)
            else:
                for k, v in infos.items():
                    self.writer.add_scalars(k, {k: v}, step)
