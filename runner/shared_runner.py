from collections import defaultdict
import time
import os
import numpy as np
from itertools import chain
import torch
import wandb
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer

from algorithm.trainers.mappo import MAPPO
from algorithm.policy import RolloutRequest, RolloutResult
from algorithm.trainer import SampleBatch
from utils.namedarray import recursive_apply
from utils.timing import Timing


class SharedRunner:

    def __init__(self, config, policy):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_render = self.all_args.use_render
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.all_args.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writer = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        self.policy = policy

        if self.model_dir is not None:
            self.restore()

        self.buffer = SharedReplayBuffer(
            self.num_agents,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            self.episode_length,
            self.n_rollout_threads,
            self.all_args.gamma,
            self.all_args.gae_lambda,
            self.policy.policy_state_space
            if self.policy.num_rnn_layers > 0 else None,
            device=self.device)

        self.trainer = MAPPO(self.all_args, self.policy)

    def run(self):

        def to_tensor(x):
            return torch.from_numpy(x).to(self.device)

        obs = self.envs.reset()
        self.buffer.storage.obs[0] = recursive_apply(
            obs, lambda x: torch.from_numpy(x).to(self.device))

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            timing = Timing()

            train_ep_ret = train_ep_length = train_ep_cnt = 0

            for step in range(self.episode_length):
                # Sample actions
                with timing.add_time("inference"):
                    rollout_result = self.collect(step)

                with timing.add_time("envstep"):
                    # Obser reward and next obs
                    actions = recursive_apply(rollout_result.action,
                                              lambda x: x.cpu().numpy())
                    (obs, rewards, dones, infos) = self.envs.step(actions)
                    (obs, rewards,
                     dones) = map(lambda x: recursive_apply(x, to_tensor),
                                  (obs, rewards, dones))
                    assert rewards.shape == (self.n_rollout_threads, self.num_agents, 1), rewards.shape
                    assert dones.shape == (self.n_rollout_threads, self.num_agents, 1), dones.shape

                    for (done, info) in zip(dones, infos):
                        if done.all():
                            train_ep_ret += info[0]['episode']['r']
                            train_ep_cnt += 1
                            train_ep_length += info[0]['episode']['l']

                with timing.add_time("buffer"):
                    dones_env = dones.all(1, keepdim=True).float()
                    masks = 1 - dones_env

                    active_masks = 1 - dones
                    active_masks = active_masks * (1 - dones_env) + dones_env

                    bad_masks = torch.tensor(
                        [[[0.0]
                          if info[agent_id].get('bad_transition') else [1.0]
                          for agent_id in range(self.num_agents)]
                         for info in infos],
                        dtype=torch.float32,
                        device=self.device)

                    data = SampleBatch(
                        obs=obs,
                        value_preds=rollout_result.value,
                        returns=None,
                        actions=rollout_result.action,
                        action_log_probs=rollout_result.log_prob,
                        rewards=rewards,
                        masks=masks,
                        active_masks=active_masks,
                        bad_masks=bad_masks)

                    # insert data into buffer
                    self.buffer.insert(data)

            # compute return and update network
            with timing.add_time("gae"):
                rollout_result = self.collect(self.episode_length)
                self.buffer.storage.value_preds[self.episode_length] = rollout_result.value
                self.buffer.compute_returns(
                    value_normalizer=self.policy.popart_head)
            with timing.add_time("train"):
                train_infos = self.train()

            print(timing)
            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.experiment_name, episode, episodes,
                            total_num_steps, self.num_env_steps,
                            int(total_num_steps / (end - start))))

                if train_ep_cnt > 0:
                    train_env_info = dict(
                        train_episode_length=train_ep_length / train_ep_cnt,
                        train_episode_return=train_ep_ret / train_ep_cnt)
                    print(
                        "Average training episode return is {}, episode length is {}."
                        .format(train_ep_ret / train_ep_cnt,
                                train_ep_length / train_ep_cnt))
                    train_infos = {**train_env_info, **train_infos}

                self.log_info(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                # TODO: add eval
                pass
                # self.eval(total_num_steps)

    @torch.no_grad()
    def collect(self, step) -> RolloutResult:
        trainer = self.trainer
        trainer.prep_rollout()
        request = RolloutRequest(
            self.buffer.storage.obs[step],
            self.buffer.storage.policy_state[step]
            if self.buffer.storage.policy_state is not None else None,
            self.buffer.storage.masks[step])
        request = recursive_apply(request, lambda x: x.flatten(end_dim=1))
        rollout_result = trainer.policy.rollout(request, deterministic=False)
        return recursive_apply(
            rollout_result, lambda x: x.view(self.n_rollout_threads, self.
                                             num_agents, *x.shape[1:]))

    def train(self):
        train_infos = defaultdict(lambda: 0)
        for _ in range(self.all_args.sample_reuse):
            self.trainer.prep_training()
            train_info = self.trainer.train(self.buffer)
            for k, v in train_info.items():
                train_infos[k] += v
        self.buffer.after_update()
        return {
            k: v / self.all_args.sample_reuse
            for k, v in train_infos.items()
        }

    def save(self):
        torch.save(self.policy.get_checkpoint(),
                   os.path.join(str(self.save_dir), "model.pt"))

    def restore(self):
        checkpoint = torch.load(os.path.join(str(self.model_dir), "model.pt"))
        self.policy.load_checkpoint(checkpoint)

    def log_info(self, infos, step):
        if self.all_args.use_wandb:
            wandb.log(infos, step=step)
        else:
            for k, v in infos.items():
                self.writer.add_scalars(k, {k: v}, step)
