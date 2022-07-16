from torch.distributions import Categorical
from typing import List, Dict, Union, Tuple
import copy
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.policy import RolloutRequest, RolloutResult, register
from algorithm.policies.actor_critic_policy import PolicyState
from algorithm.policies.ar_policy_base import ARPolicy
from algorithm.trainer import SampleBatch
from utils.namedarray import recursive_apply, namedarray
import algorithm.modules


class BridgeAgentwiseObsEncoder(nn.Module):

    def __init__(self, obs_shapes: Dict, hidden_dim):
        super().__init__()

        for k, shape in obs_shapes.items():
            if 'mask' not in k:
                setattr(self, k + '_embedding',
                        algorithm.modules.get_layer(shape[-1], hidden_dim))
        self.attn = algorithm.modules.MultiHeadSelfAttention(hidden_dim,
                                                             hidden_dim,
                                                             4,
                                                             entry=0)
        l = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(l.bias.data)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        self.dense = nn.Sequential(nn.LayerNorm(hidden_dim), l,
                                   nn.ReLU(inplace=True),
                                   nn.LayerNorm(hidden_dim))

    def forward(self, obs):
        obs_ = {}
        for k, x in obs.items():
            if 'mask' not in k:
                assert hasattr(self, k + '_embedding')
                obs_[k] = getattr(self, k + '_embedding')(x)
            else:
                assert k == 'obs_mask'
        obs_['obs_self'] = obs_['obs_self'].unsqueeze(-2)
        obs_embedding = torch.cat(list(obs_.values()), -2)
        return self.dense(self.attn(obs_embedding, mask=obs.obs_mask))


class BridgeARActor(nn.Module):

    def __init__(
        self,
        obs_shape: Union[Tuple, Dict],
        hidden_dim: int,
        act_dim: int,
        n_agents: int,
        agent_specific_obs: bool = False,
    ):
        super().__init__()

        if agent_specific_obs:
            assert isinstance(obs_shape, Dict), obs_shape
            self.actor_base = algorithm.modules.ARAgentwiseObsEncoder(
                obs_shape, act_dim, hidden_dim)
        else:
            self.actor_base = algorithm.modules.mlp([
                obs_shape[0] + act_dim * (n_agents - 1), hidden_dim,
                hidden_dim, act_dim
            ])
        self.agent_specific_obs = agent_specific_obs

    def forward(self,
                obs,
                action,
                execution_mask,
                actor_hx,
                mask,
                available_actions=None):
        if hasattr(obs, 'obs'):
            obs = obs.obs
        if self.agent_specific_obs:
            logits = self.actor_base(obs, action, execution_mask)
        else:
            action = action * execution_mask.unsqueeze(-1)
            action = action.view(*action.shape[:-2], -1)
            logits = self.actor_base(torch.cat([obs, action], -1))
        return logits, None


class BridgeARCritic(nn.Module):

    def __init__(
        self,
        obs_shape: Union[Tuple, Dict],
        hidden_dim: int,
        agent_specific_obs: bool = False,
    ):
        super().__init__()

        if agent_specific_obs:
            assert isinstance(obs_shape, Dict), obs_shape
            self.critic_base = BridgeAgentwiseObsEncoder(obs_shape, hidden_dim)
        else:
            self.critic_base = algorithm.modules.mlp(
                [obs_shape[0], hidden_dim, hidden_dim])
        self.agent_specific_obs = agent_specific_obs

        self.v_out = algorithm.modules.PopArtValueHead(hidden_dim, 1)

    def forward(self, obs, critic_hx, mask):
        if hasattr(obs, 'obs'):
            obs = obs.obs
        critic_features = self.critic_base(obs)
        return self.v_out(critic_features), None


class BridgeARPolicy(ARPolicy):

    def __init__(
            self,
            observation_space,
            action_space,
            n_agents: int,
            random_order: bool,
            agent_specific_obs: bool,
            hidden_dim: int = 64,
            value_dim: int = 1,
            num_dense_layers: int = 2,
            rnn_type: str = "gru",
            num_rnn_layers: int = 0,
            popart: bool = True,
            activation: str = "relu",
            layernorm: bool = True,
            use_feature_normalization: bool = True,
            device=torch.device("cuda:0"),
            **kwargs,
    ):
        actor = BridgeARActor(observation_space.shape,
                              hidden_dim,
                              act_dim=5,
                              n_agents=n_agents,
                              agent_specific_obs=agent_specific_obs)
        critic = BridgeARCritic(observation_space.shape,
                                hidden_dim,
                                agent_specific_obs=agent_specific_obs)
        super().__init__(actor,
                         critic,
                         act_dim=5,
                         n_agents=n_agents,
                         random_order=kwargs.get('random_order', True),
                         num_rnn_layers=num_rnn_layers,
                         hidden_dim=hidden_dim,
                         popart=popart,
                         device=device,
                         **kwargs)


register("bridge_ar", BridgeARPolicy)
