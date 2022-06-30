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
            # NOTE: sampled available actions should be 1
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

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.storage[0] = self.storage[-1]
        assert self.step == 0, self.step
        assert (self.storage.actions[-1] == 0).all()
        assert (self.storage.returns[-1] == 0).all()
        assert (self.storage.rewards[-1] == 0).all()
        assert (self.storage.action_log_probs[-1] == 0).all()
