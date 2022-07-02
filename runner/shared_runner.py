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
from utils.namedarray import recursive_apply, array_like
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
                 storage,
                 env_ctrls,
                 info_queue,
                 eval_storage,
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
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        # interval
        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # TODO: wandb mode
        if not all_args.eval:
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

        self.storage = storage
        self.policy = policy
        self.env_ctrls = env_ctrls
        self.info_queue = info_queue

        self.eval_storage = eval_storage
        self.eval_env_ctrls = eval_env_ctrls
        self.eval_info_queue = eval_info_queue

        if self.model_dir is not None:
            self.restore()

        self.trainer = MAPPO(self.all_args, self.policy)

    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            timing = Timing()

            train_ep_ret = train_ep_length = train_ep_cnt = 0

            for step in range(self.episode_length):
                # Sample actions
                with timing.add_time("envstep"):
                    for ctrl in self.env_ctrls:
                        ctrl.obs_ready.acquire()

                with timing.add_time("inference"):
                    rollout_result = self.collect(step)

                with timing.add_time("storage"):
                    step = self.storage.step

                    self.storage.value_preds[step] = rollout_result.value
                    self.storage.actions[step] = rollout_result.action.float()
                    self.storage.action_log_probs[
                        step] = rollout_result.log_prob
                    if self.storage.policy_state is not None:
                        self.storage.policy_state[
                            step + 1] = rollout_result.policy_state

                    self.storage.step += 1

                for ctrl in self.env_ctrls:
                    ctrl.act_ready.release()

            for ctrl in self.env_ctrls:
                ctrl.obs_ready.acquire()

            while True:
                try:
                    info = self.info_queue.get_nowait()
                    train_ep_ret += info['episode']['r']
                    train_ep_length += info['episode']['l']
                    train_ep_cnt += 1
                except queue.Empty:
                    break

            # compute return and update network
            with timing.add_time("gae"):
                assert self.storage.step == self.episode_length
                rollout_result = self.collect(self.episode_length)
                self.storage.value_preds[
                    self.episode_length] = rollout_result.value
                if self.policy.popart_head is not None:
                    trace_target_value = self.policy.denormalize_value(
                        self.storage.value_preds.to(self.device)).cpu()
                else:
                    trace_target_value = self.storage.value_preds
                adv = gae_trace(self.storage.rewards, trace_target_value,
                                self.storage.masks, self.all_args.gamma,
                                self.all_args.gae_lambda,
                                self.storage.bad_masks)
                self.storage.returns = adv + trace_target_value
                self.storage.advantages = adv
                self.storage.advantages[:-1] = masked_normalization(
                    adv[:-1], mask=self.storage.active_masks[:-1])

            with timing.add_time("train"):
                train_infos = self.train()

            self.storage[0] = self.storage[-1]
            # storage.step must be changed via inplace operations
            assert self.storage.step == self.episode_length
            self.storage.step %= self.episode_length

            for ctrl in self.env_ctrls:
                ctrl.obs_ready.release()

            logger.debug(timing)

            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads
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
    def collect(self, step) -> RolloutResult:
        trainer = self.trainer
        trainer.prep_rollout()
        request = RolloutRequest(
            self.storage.obs[step], self.storage.policy_state[step]
            if self.storage.policy_state is not None else None,
            self.storage.masks[step])
        request = recursive_apply(
            request, lambda x: x.flatten(end_dim=1).to(self.device))
        rollout_result = trainer.policy.rollout(request, deterministic=False)
        return recursive_apply(
            rollout_result, lambda x: x.view(self.n_rollout_threads, self.
                                             num_agents, *x.shape[1:]).cpu())

    def train(self):
        train_infos = defaultdict(lambda: 0)
        self.trainer.prep_training()
        for _ in range(self.all_args.sample_reuse):
            train_info = self.trainer.train(
                recursive_apply(self.storage[:-1],
                                lambda x: x.to(self.device)))
            for k, v in train_info.items():
                train_infos[k] += v
        self.policy.inc_version()

        return {
            k: float(v / self.all_args.sample_reuse)
            for k, v in train_infos.items()
        }

    def eval(self, step):
        for ctrl in self.eval_env_ctrls:
            ctrl.eval_finish.clear()
            ctrl.eval_start.set()
            assert not ctrl.act_ready.acquire(block=False)
            while ctrl.obs_ready.acquire(block=False):
                continue

        self.trainer.prep_rollout()
        if self.storage.policy_state is not None:
            policy_state = array_like(self.eval_storage.policy_state,
                                      default_value=0)
        else:
            policy_state = None
        eval_ep_cnt = eval_ep_len = eval_ep_ret = 0

        while eval_ep_cnt < self.all_args.eval_episodes:
            for ctrl in self.eval_env_ctrls:
                ctrl.obs_ready.acquire()

            while True:
                try:
                    info = self.eval_info_queue.get_nowait()
                    eval_ep_ret += info['episode']['r']
                    eval_ep_len += info['episode']['l']
                    eval_ep_cnt += 1
                except queue.Empty:
                    break

            request = RolloutRequest(self.eval_storage.obs, policy_state,
                                     self.eval_storage.masks)
            request = recursive_apply(
                request, lambda x: x.flatten(end_dim=1).to(self.device))
            rollout_result = recursive_apply(
                self.policy.rollout(request, deterministic=True),
                lambda x: x.view(self.n_eval_rollout_threads, self.num_agents,
                                 *x.shape[1:]).cpu())

            self.eval_storage.actions[:] = rollout_result.action.float()
            policy_state = rollout_result.policy_state

            for ctrl in self.eval_env_ctrls:
                ctrl.act_ready.release()

        for ctrl in self.eval_env_ctrls:
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
