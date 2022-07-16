from typing import Dict
import copy
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import algorithm.modules
from algorithm.policies.actor_critic_policy import PolicyState, ActorCriticPolicy, ActorCriticPolicyStateSpace
from algorithm.policy import RolloutRequest, RolloutResult
from algorithm.trainer import SampleBatch
from utils.namedarray import recursive_apply, namedarray


def generate_mask_from_order(agent_order, ego_exclusive):
    """Generate execution mask from agent order.

    Used during autoregressive training.

    Args:
        agent_order (torch.Tensor): Agent order of shape [*, n_agents].

    Returns:
        torch.Tensor: Execution mask of shape [*, n_agents, n_agents - 1].
    """
    shape = agent_order.shape
    n_agents = shape[-1]
    agent_order = agent_order.view(-1, n_agents)
    bs = agent_order.shape[0]

    cur_execution_mask = torch.zeros(bs, n_agents).to(agent_order)
    all_execution_mask = torch.zeros(bs, n_agents, n_agents).to(agent_order)

    batch_indices = torch.arange(bs)
    for i in range(n_agents):
        agent_indices = agent_order[:, i].long()

        cur_execution_mask[batch_indices, agent_indices] = 1
        all_execution_mask[batch_indices, :,
                           agent_indices] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_indices, agent_indices] = 1
    if not ego_exclusive:
        # [*, n_agent, n_agents]
        return all_execution_mask.view(*shape[:-1], n_agents, n_agents)
    else:
        # [*, n_agents, n_agents - 1]
        execution_mask = torch.zeros(bs, n_agents,
                                     n_agents - 1).to(agent_order)
        for i in range(n_agents):
            execution_mask[:, i] = torch.cat([
                all_execution_mask[..., i, :i], all_execution_mask[..., i,
                                                                   i + 1:]
            ], -1)
        return execution_mask.view(*shape[:-1], n_agents, n_agents - 1)


class ARPolicy(ActorCriticPolicy):

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        act_dim: int,
        n_agents: int,
        random_order: bool,
        num_rnn_layers: int,
        hidden_dim: int,
        popart: bool,
        device: torch.device,
        **kwargs,
    ):
        self._popart = popart
        self._device = device
        self._num_rnn_layers = num_rnn_layers

        self.n_agents = n_agents
        self.act_dim = act_dim
        self.random_order = random_order

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        if self.num_rnn_layers > 0:
            self.policy_state_space = ActorCriticPolicyStateSpace(
                num_rnn_layers, hidden_dim)

        self._version = 0

    def analyze(self, sample: SampleBatch, **kwargs):
        if sample.policy_state is not None:
            actor_hx = sample.policy_state.actor_hx[0].transpose(0, 1)
            critic_hx = sample.policy_state.critic_hx[0].transpose(0, 1)
        else:
            actor_hx = critic_hx = None
        obs = sample.obs.obs
        if hasattr(sample.obs, "state"):
            state = sample.obs.state
        else:
            state = obs

        action_ = sample.actions.squeeze(
            -1).long()  # [chunk_len, bs, n_agents]
        ego_exclusive_action = [
            torch.cat([action_[..., :i], action_[..., i + 1:]], -1)
            for i in range(self.n_agents)
        ]
        ego_exclusive_action = torch.stack(
            ego_exclusive_action,
            -2)  # [chunk_len, bs, n_agents, n_agents - 1]

        bs = sample.masks.shape[1]
        T = sample.masks.shape[0]

        if self.num_rnn_layers > 0:
            if self.random_order:
                agent_order = torch.stack([
                    torch.randperm(self.n_agents) for _ in range(bs)
                ])[None, :]
            else:
                agent_order = torch.stack(
                    [torch.arange(self.n_agents) for _ in range(bs)])[None, :]
            agent_order = torch.repeat(agent_order, (on_reset.shape[0], 1, 1))
        else:
            if self.random_order:
                agent_order = torch.stack(
                    [torch.randperm(self.n_agents) for _ in range(bs * T)])
            else:
                agent_order = torch.stack(
                    [torch.arange(self.n_agents) for _ in range(bs * T)])
        agent_order = agent_order.view(T, bs, self.n_agents).to(sample.masks)

        all_execution_mask = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()  # [T, bs, n_agents, n_agents]
        all_execution_mask *= sample.active_masks.float().squeeze(
            -1).unsqueeze(-2)

        # construct ego-exclusive execution mask
        execution_mask = [
            torch.cat([
                all_execution_mask[..., i, :i], all_execution_mask[..., i,
                                                                   i + 1:]
            ], -1) for i in range(self.n_agents)
        ]
        execution_mask = torch.stack(execution_mask,
                                     -2)  # [T, bs, n_agents, n_agents - 1]

        onehot_action = F.one_hot(ego_exclusive_action.long(),
                                  self.act_dim).float()

        logits, _ = self.actor(obs, onehot_action, execution_mask, actor_hx,
                               sample.masks,
                               getattr(sample.obs, 'available_actions', None))
        cf_logits, _ = self.actor(
            state, torch.zeros_like(onehot_action),
            torch.zeros_like(execution_mask), actor_hx, sample.masks,
            getattr(sample.obs, 'available_actions', None))
        values, _ = self.critic(state, critic_hx, sample.masks)

        action_distribution = torch.distributions.Categorical(logits=logits)
        cf_action_distribution = torch.distributions.Categorical(
            logits=cf_logits)

        new_log_probs = action_distribution.log_prob(action_).unsqueeze(-1)

        entropy = cf_action_distribution.entropy()
        joint_ent = action_distribution.entropy().detach().sum(-1,
                                                               keepdim=True)

        return new_log_probs, values, entropy, joint_ent, all_execution_mask

    @torch.no_grad()
    def rollout(self, requests: RolloutRequest,
                deterministic) -> RolloutResult:
        bs = requests.mask.shape[0]

        requests = recursive_apply(requests, lambda x: x.unsqueeze(0))
        obs = requests.obs.obs
        if hasattr(requests.obs, "state"):
            state = requests.obs.state
        else:
            state = obs
        if requests.policy_state is not None:
            actor_hx = requests.policy_state.actor_hx[0].transpose(0, 1)
            critic_hx = requests.policy_state.critic_hx[0].transpose(0, 1)
        else:
            actor_hx = critic_hx = None

        # prepare placeholders
        value = torch.zeros(bs, self.n_agents, 1).to(dtype=torch.float32,
                                                     device=self.device)
        log_probs = torch.zeros(bs, self.n_agents).to(dtype=torch.float32,
                                                      device=self.device)
        actions = torch.zeros(bs, self.n_agents).to(dtype=torch.long,
                                                    device=self.device)
        if self.num_rnn_layers > 0:
            actor_hx_ = torch.zeros_like(actor_hx)
            critic_hx_ = torch.zeros_like(critic_hx)

        for agent_idx in range(self.n_agents):
            # construct ego-exclusive one-hot actions based on current available actions
            ego_exclusive_action = torch.cat(
                [actions[..., :agent_idx], actions[..., agent_idx + 1:]],
                -1).unsqueeze(0)  # [1, bs, n_agents - 1]

            # select inputs
            a_obs = requests.obs[:, :, agent_idx]
            a_state = state[:, :, agent_idx]
            if self.num_rnn_layers > 0:
                a_actor_hx = actor_hx[:, :, agent_idx]
                a_critic_hx = critic_hx[:, :, agent_idx]
            else:
                a_actor_hx = a_critic_hx = None

            execution_mask = torch.stack([torch.ones(bs)] * agent_idx +
                                         [torch.zeros(bs)] *
                                         (self.n_agents - 1 - agent_idx), -1)
            alive_mask = torch.cat([
                requests.active_mask[..., :agent_idx, :],
                requests.active_mask[..., agent_idx + 1:, :]
            ], -2)  # [1, bs, n_agents-1, 1]
            execution_mask = execution_mask.to(
                self.device) * alive_mask.squeeze(-1)  # [1, bs, n_agents - 1]
            onehot_action = F.one_hot(ego_exclusive_action.long(),
                                      self.act_dim).float()

            # output tensors should have shape [1, bs, *D] (without agent dim)
            a_logits, a_actor_hx = self.actor(
                a_obs, onehot_action, execution_mask,
                a_actor_hx, requests.mask,
                getattr(requests.obs, "available_actions", None))
            a_value, a_critic_hx = self.critic(a_state, a_critic_hx,
                                               requests.mask)

            a_action_distribution = torch.distributions.Categorical(
                logits=a_logits)

            if deterministic:
                a_action = a_action_distribution.probs.argmax(-1).squeeze(
                    0)  # [bs]
            else:
                a_action = a_action_distribution.sample().squeeze(0)  # [bs]
            a_log_probs = a_action_distribution.log_prob(a_action).squeeze(
                0)  # [bs]

            # set placeholders
            value[:, agent_idx] = a_value
            log_probs[:, agent_idx] = a_log_probs
            actions[:, agent_idx] = a_action.long()
            if self.num_rnn_layers > 0:
                actor_hx_[:, :, agent_idx] = a_actor_hx
                critic_hx_[:, :, agent_idx] = a_critic_hx

        # .unsqueeze(-1) adds a trailing dimension 1
        return RolloutResult(
            action=actions.unsqueeze(-1),
            log_prob=log_probs.unsqueeze(-1),
            value=value,
            policy_state=PolicyState(actor_hx_.transpose(0, 1),
                                     critic_hx_.transpose(0, 1))
            if self.num_rnn_layers > 0 else None,
        )
