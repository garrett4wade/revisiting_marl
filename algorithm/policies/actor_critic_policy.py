import dataclasses
import itertools
import gym
import math
import torch
import torch.nn as nn

from algorithm import modules
from algorithm.policy import register, RolloutRequest, RolloutResult
from utils.namedarray import namedarray, recursive_apply
from utils.shared_buffer import SampleBatch


@namedarray
class PolicyState:
    actor_hx: torch.Tensor
    critic_hx: torch.Tensor


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_initialized_linear(input_dim, output_dim, gain):

    return init(nn.Linear(input_dim, output_dim), nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), gain)


class Actor(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_space,
        hidden_dim,
        num_dense_layers,
        num_rnn_layers,
        use_feature_normalization=True,
        dense_layer_gain=math.sqrt(2),
        activation='relu',
        layernorm=True,
        act_layer_gain=0.01,
        rnn_type='gru',
        std_type='fixed',
        init_log_std=-0.5,
        **kwargs,
    ):
        super(Actor, self).__init__()

        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        self.base = modules.RecurrentBackbone(obs_dim, num_dense_layers,
                                              hidden_dim, rnn_type,
                                              num_rnn_layers, dense_layer_gain,
                                              activation, layernorm)

        self.action_type = None
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_type = 'discrete'
            action_dim = action_space.n
            self.action_out = get_initialized_linear(hidden_dim, action_dim,
                                                     act_layer_gain)
        elif isinstance(action_space, gym.spaces.Box):
            self.action_type = 'continuous'
            action_dim = action_space.shape[0]
            self.fc_mean = get_initialized_linear(hidden_dim, action_dim,
                                                  act_layer_gain)
            self.std_type = std_type
            if self.std_type == 'fixed':
                self.log_std = nn.Parameter(float(init_log_std) *
                                            torch.ones(num_outputs),
                                            requires_grad=False)
            elif self.std_type == 'separate_learnable':
                self.log_std = nn.Parameter(float(init_log_std) *
                                            torch.ones(num_outputs),
                                            requires_grad=True)
            elif self.std_type == 'shared_learnable':
                self.log_std = get_initialized_linear(hidden_dim, action_dim,
                                                      act_layer_gain)
            else:
                raise NotImplementedError(
                    f"Standard deviation type {self.std_type} not implemented."
                )
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_type = 'multidiscrete'
            self.action_outs = []
            for action_dim in action_space.nvec:
                self.action_outs.append(
                    get_initialized_linear(hidden_dim, action_dim,
                                           act_layer_gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:
            raise NotImplementedError(
                f"Action space {action_space} not implemented.")

    def forward(self, obs, rnn_states, masks, available_actions=None):
        actor_features, rnn_states = self.base(obs, rnn_states, masks)

        if self.action_type == 'multidiscrete':
            actor_output = [action_out(x) for action_out in self.action_outs]
        elif self.action_type == 'discrete':
            actor_output = self.action_out(x, available_actions)
        elif self.action_type == 'continuous':
            action_mean = self.fc_mean(x)
            if self.std_type == 'fixed' or self.std_type == 'separate_learnable':
                action_std = self.log_std.exp() * torch.ones_like(action_mean)
            elif self.std_type == 'shared_learnable':
                action_std = self.log_std(x).exp()
            acotr_output = (action_mean, action_std)
        else:
            raise NotImplementedError()

        return actor_output, rnn_states


class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_dense_layers,
        num_rnn_layers,
        use_feature_normalization=True,
        v_out_gain=0.01,
        rnn_type='gru',
        dense_layer_gain=math.sqrt(2),
        activation='relu',
        layernorm=True,
        critic_dim=1,
        popart=True,
        popart_burn_in_updates=torch.inf,
        **kwargs,
    ):
        super(Critic, self).__init__()
        self.base = modules.RecurrentBackbone(state_dim, num_dense_layers,
                                              hidden_dim, rnn_type,
                                              num_rnn_layers, dense_layer_gain,
                                              activation, layernorm)

        if popart:
            self.v_out = modules.PopArtValueHead(
                self.base.feature_dim,
                critic_dim,
                burn_in_updates=popart_burn_in_updates,
                init_gain=v_out_gain)
        else:
            self.v_out = get_initialized_linear(self.base.feature_dim,
                                                critic_dim, v_out_gain)

    def forward(self, cent_obs, rnn_states, masks):
        critic_features, rnn_states = self.base(cent_obs)
        values = self.v_out(critic_features)
        return values, rnn_states


class ActorCriticPolicyStateSpace:

    def __init__(self, num_rnn_layers, hidden_dim):
        self.num_rnn_layers = num_rnn_layers
        self.hidden_dim = hidden_dim

    def sample(self):
        return PolicyState(torch.zeros(self.num_rnn_layers, self.hidden_dim),
                           torch.zeros(self.num_rnn_layers, self.hidden_dim))


class ActorCriticPolicy:

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_dim: int = 64,
            value_dim: int = 1,
            num_dense_layers: int = 2,
            rnn_type: str = "gru",
            num_rnn_layers: int = 1,
            popart: bool = True,
            activation: str = "relu",
            layernorm: bool = True,
            use_feature_normalization: bool = True,
            device=torch.device("cuda:0"),
            **kwargs,
    ):
        self.__rnn_hidden_dim = hidden_dim
        self.__popart = popart
        self.__value_dim = value_dim
        self.device = device
        self._num_rnn_layers = num_rnn_layers

        x = observation_space.sample()
        obs_dim = x.obs.shape[-1]
        if hasattr(x, "state"):
            state_dim = x.state.shape[-1]
        else:
            state_dim = obs_dim
        self.actor = Actor(obs_dim, action_space, hidden_dim, num_dense_layers,
                           num_rnn_layers, use_feature_normalization,
                           **kwargs).to(device)

        self.critic = Critic(state_dim, hidden_dim, num_dense_layers,
                             num_rnn_layers, use_feature_normalization,
                             **kwargs).to(device)

        self.policy_state_space = ActorCriticPolicyStateSpace(
            num_rnn_layers, hidden_dim)

        self._version = 0

    @property
    def version(self):
        return self._version

    def inc_version(self):
        self._version += 1

    def parameters(self):
        return itertools.chain(self.actor.parameters(),
                               self.critic.parameters())

    def load_checkpoint(self, checkpoint):
        """Load a checkpoint.
        If "steps" is missing in the checkpoint. We assume that the checkpoint is from a pretrained model. And
        set version to 0. So that the trainer side won't ignore the sample generated by this version.
        """
        self._version = checkpoint.get("steps", 0)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint['critic'])

    def get_checkpoint(self):
        return {
            "steps": self._version,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    @property
    def num_rnn_layers(self):
        return self._num_rnn_layers

    @property
    def popart_head(self):
        if not self.__popart:
            raise ValueError(
                "Set popart=True in policy config to activate popart value head."
            )
        return self.net.critic_head

    def normalize_value(self, x):
        return self.popart_head.normalize(x)

    def denormalize_value(self, x):
        return self.popart_head.denormalize(x)

    def update_popart(self, x):
        return self.popart_head.update(x)

    def analyze(self, sample: SampleBatch, target="ppo", **kwargs):
        actor_hx = sample.policy_state.actor_hx[0].transpose(0, 1)
        critic_hx = sample.policy_state.critic_hx[0].transpose(0, 1)
        obs = sample.obs.obs
        if hasattr(sample.obs, "state"):
            state = sample.obs.state
        else:
            state = obs
        actor_output, _ = self.actor(
            obs, actor_hx, sample.masks,
            getattr(sample.obs, 'available_actions', None))
        values, _ = self.critic(state, critic_hx, sample.masks)
        new_log_probs, entropy = self.get_log_prob_and_entropy(
            self.get_action_distribution(actor_output), sample.actions)

        return new_log_probs, values, entropy

    def get_action_distribution(self, actor_output):
        if self.actor.action_type == 'continuous':
            mean, std = actor_output
            return [torch.distributions.Normal(mean, std)]
        elif self.actor.action_type == 'discrete':
            return [torch.distributions.Categorical(logits=actor_output)]
        elif self.actor.action_type == 'multidiscrete':
            return [
                torch.distributions.Categorical(logits=x) for x in actor_output
            ]

    def get_log_prob_and_entropy(self, action_dists, actions):
        if self.actor.action_type == 'discrete' or self.actor.action_type == 'multidiscrete':
            new_log_probs = torch.sum(torch.stack([
                dist.log_prob(actions[..., i])
                for i, dist in enumerate(action_dists)
            ],
                                                  dim=-1),
                                      dim=-1,
                                      keepdim=True)
            entropy = torch.sum(torch.stack(
                [dist.entropy() for dist in action_dists], dim=-1),
                                dim=-1,
                                keepdim=True)
        elif self.actor.action_type == 'continuous':
            [action_dist] = action_dists
            new_log_probs = action_dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = action_dist.entropy().sum(-1, keepdim=True)
        return new_log_probs, entropy

    @torch.no_grad()
    def rollout(self, requests: RolloutRequest, deterministic):
        requests = recursive_apply(requests, lambda x: x.unsqueeze(0))
        obs = requests.obs.obs
        if hasattr(requests.obs, "state"):
            state = requests.state
        else:
            state = obs
        actor_hx = requests.policy_state.actor_hx[0].transpose(0, 1)
        critic_hx = requests.policy_state.critic_hx[0].transpose(0, 1)

        actor_output, actor_hx = self.actor(
            obs, actor_hx, requests.mask,
            getattr(requests.obs, "available_actions", None))
        value, critic_hx = self.critic(state, critic_hx, requests.mask)

        value = value.squeeze(0)
        if self.actor.action_type == 'discrete':
            actor_output = actor_output.squeeze(0)
        elif self.actor.action_type == 'multidiscrete' or self.actor.action_type == 'continuous':
            actor_output = [x.squeeze(0) for x in actor_output]
        action_dists = self.get_action_distribution(actor_output)

        if self.actor.action_type == 'discrete' or self.actor.action_type == 'multidiscrete':
            if deterministic:
                # .squeeze(0) removes the time dimension
                actions = torch.stack(
                    [dist.probs.argmax(dim=-1) for dist in action_dists],
                    dim=-1)
            else:

                # dist.sample adds an additional dimension
                actions = torch.stack(
                    [dist.sample().squeeze(0) for dist in action_dists],
                    dim=-1)
            log_probs = torch.sum(torch.stack([
                dist.log_prob(actions[..., i])
                for i, dist in enumerate(action_dists)
            ],
                                              dim=-1),
                                  dim=-1,
                                  keepdim=True)
        elif self.actor.action_type == 'continuous':
            [action_dist] = action_dists
            if deterministic:
                mean, _ = actor_output
                actions = mean
            else:
                actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions).sum(-1, keepdim=True)

        # .unsqueeze(-1) adds a trailing dimension 1
        policy_state = PolicyState(actor_hx.transpose(0, 1),
                                   critic_hx.transpose(0, 1))
        return RolloutResult(action=actions,
                             log_probs=log_probs,
                             value=value,
                             policy_state=policy_state)


register("actor-critic", ActorCriticPolicy)