from typing import Union
import gym
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from utils.namedarray import namedarray, recursive_apply
from environment.env_base import Observation, Action
from algorithm.policy import PolicyState
from algorithm.trainer import SampleBatch


@torch.no_grad()
def masked_normalization(x,
                         mask=None,
                         dim=None,
                         inplace=False,
                         unbiased=False,
                         eps=torch.tensor(1e-5)):
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        mask = torch.ones_like(x)
    x = x * mask
    factor = mask.sum(dim=dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return (x - mean) / (var.sqrt() + eps)


class SharedReplayBuffer(object):

    def __init__(self,
                 num_agents,
                 obs_space,
                 act_space,
                 episode_length,
                 n_rollout_threads,
                 gamma,
                 gae_lambda,
                 policy_state_space=None,
                 use_gae=True,
                 use_popart=True,
                 use_valuenorm=True,
                 use_proper_time_limits=False,
                 device=torch.device("cuda:0")):
        self.episode_length = episode_length
        self.n_rollout_threads = n_rollout_threads
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._use_gae = use_gae
        self._use_popart = use_popart
        self._use_valuenorm = use_valuenorm
        self._use_proper_time_limits = use_proper_time_limits

        self.num_agents = num_agents
        self.device = device

        if isinstance(act_space, gym.spaces.Discrete):
            act_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            act_dim = act_space.shape[0]
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_dim = sum(act_space.nvec)
        else:
            raise NotImplementedError()

        self.storage = SampleBatch(
            obs=obs_space.sample(),
            value_preds=torch.zeros(1),
            returns=torch.zeros(1),
            actions=torch.zeros(act_dim),
            action_log_probs=torch.zeros(1),
            rewards=torch.zeros(1),
            masks=torch.ones(1),
            active_masks=torch.ones(1),
            bad_masks=torch.ones(1),
        )

        if policy_state_space is not None:
            self.storage.policy_state = policy_state_space.sample()

        self.storage = recursive_apply(
            self.storage,
            lambda x: x.to(self.device).repeat(self.episode_length + 1, self.
                                               n_rollout_threads, num_agents,
                                               *((1, ) * len(x.shape))),
        )

        self.step = 0

    def insert(self, sample):
        self.storage.obs[self.step + 1] = sample.obs
        if self.storage.policy_state is not None:
            self.storage.policy_state[self.step + 1] = sample.policy_state

        self.storage.actions[self.step] = sample.actions
        self.storage.action_log_probs[self.step] = sample.action_log_probs
        self.storage.value_preds[self.step] = sample.value_preds
        self.storage.rewards[self.step] = sample.rewards

        self.storage.masks[self.step + 1] = sample.masks
        self.storage.bad_masks[self.step + 1] = sample.bad_masks
        self.storage.active_masks[self.step + 1] = sample.active_masks

        self.step += 1

    def after_update(self):
        self.storage[0] = self.storage[-1]
        assert self.step == self.episode_length, self.step
        self.step = 0

    def compute_returns(self, value_normalizer=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        if self._use_popart or self._use_valuenorm:
            value_preds = value_normalizer.denormalize(
                self.storage.value_preds)
        else:
            value_preds = self.storage.value_preds
        rewards = self.storage.rewards
        masks = self.storage.masks
        bad_masks = self.storage.bad_masks
        if self._use_proper_time_limits:
            if self._use_gae:
                gae = 0
                for step in reversed(range(self.episode_length)):
                    delta = rewards[step] + self.gamma * value_preds[
                        step + 1] * masks[step + 1] - value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * masks[step +
                                                                       1] * gae
                    gae = gae * bad_masks[step + 1]
                    self.storage.returns[step] = gae + value_preds[step]
            else:
                for step in reversed(range(self.episode_length)):
                    self.storage.returns[step] = (
                        self.storage.returns[step + 1] * self.gamma *
                        masks[step + 1] +
                        rewards[step]) * bad_masks[step + 1] + (
                            1 - bad_masks[step + 1]) * value_preds[step]
        else:
            if self._use_gae:
                gae = 0
                for step in reversed(range(self.episode_length)):
                    delta = rewards[step] + self.gamma * value_preds[
                        step + 1] * masks[step + 1] - value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * masks[step +
                                                                       1] * gae
                    self.storage.returns[step] = gae + value_preds[step]
            else:
                for step in reversed(range(self.episode_length)):
                    self.storage.returns[step] = self.storage.returns[
                        step + 1] * self.gamma * masks[step +
                                                       1] + rewards[step]
        self.storage.advantages = self.storage.returns - value_preds
        self.storage.advantages[:-1] = masked_normalization(
            self.storage.advantages[:-1], mask=self.storage.active_masks[:-1])

    def feed_forward_generator(
        self,
        num_mini_batch,
    ):
        batch_size = self.n_rollout_threads * self.episode_length * self.num_agents

        assert batch_size >= num_mini_batch and batch_size % num_mini_batch == 0, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(n_rollout_threads, episode_length,
                      n_rollout_threads * episode_length, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size)
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
        sample = self.storage[:-1]
        sample = recursive_apply(sample, lambda x: x.flatten(end_dim=2))
        for indices in sampler:
            yield sample[indices]

    def recurrent_generator(
        self,
        num_mini_batch,
        data_chunk_length,
    ):
        episode_length, n_rollout_threads = self.episode_length, self.n_rollout_threads
        num_chunks = episode_length // data_chunk_length
        assert data_chunk_length <= episode_length and episode_length % data_chunk_length == 0
        data_chunks = n_rollout_threads * num_chunks * self.num_agents  # [C=r*T/L]
        assert data_chunks >= num_mini_batch and data_chunks % num_mini_batch == 0
        mini_batch_size = data_chunks // num_mini_batch

        def _cast(x):
            x = x.reshape(num_chunks, x.shape[0] // num_chunks, *x.shape[1:])
            x = x.transpose(1, 0)
            return x.flatten(start_dim=1, end_dim=3)

        rand = torch.randperm(data_chunks)
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
        sample = self.storage[:-1]
        sample = recursive_apply(sample, _cast)  # [T, B, *]

        for indices in sampler:
            yield sample[:, indices]
