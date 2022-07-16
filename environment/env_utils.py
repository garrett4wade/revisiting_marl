from typing import Dict
import gym
import torch

from utils.namedarray import namedarray
from environment import env_base


@namedarray
class DiscreteAction(env_base.Action):
    x: torch.Tensor

    def __eq__(self, other):
        assert isinstance(other, DiscreteAction), \
            "Cannot compare DiscreteAction to object of class{}".format(other.__class__.__name__)
        return self.key == other.key

    def __hash__(self):
        return hash(self.x.item())

    @property
    def key(self):
        return self.x.item()


class DiscreteActionSpace(env_base.ActionSpace):

    def __init__(self, space: gym.Space, shared=False, n_agents=-1):
        self.__shared = shared
        self.__n_agents = n_agents
        if shared and n_agents == -1:
            raise ValueError(
                "n_agents must be given to a shared action space.")
        self.__space = space
        assert isinstance(
            space,
            (gym.spaces.Discrete, gym.spaces.MultiDiscrete)), type(space)

    @property
    def n(self):
        return self.__space.n

    def sample(self, available_action: torch.Tensor = None) -> DiscreteAction:
        if available_action is None:
            if self.__shared:
                x = torch.tensor([[self.__space.sample()]
                                  for _ in range(self.__n_agents)],
                                 dtype=torch.int32)
            else:
                x = torch.tensor([self.__space.sample()], dtype=torch.int32)
            return DiscreteAction(x)
        else:
            if self.__shared:
                assert available_action.shape == (self.__n_agents,
                                                  self.__space.n)
                x = []
                for agent_idx in range(self.__n_agents):
                    a_x = self.__space.sample()
                    while not available_action[agent_idx, a_x]:
                        a_x = self.__space.sample()
                    x.append([a_x])
                x = torch.tensor(x, dtype=torch.int32)
            else:
                assert available_action.shape == (self.__space.n, )
                x = self.__space.sample()
                while not available_action[x]:
                    x = self.__space.sample()
                x = torch.tensor([x], dtype=torch.int32)
            return DiscreteAction(x)


@namedarray
class ContinuousAction(env_base.Action):
    x: torch.Tensor

    def __eq__(self, other):
        assert isinstance(other, ContinuousAction), \
            "Cannot compare ContinuousAction to object of class{}".format(other.__class__.__name__)
        return self.key == other.key

    @property
    def key(self):
        return self.x


class ContinuousActionSpace(env_base.ActionSpace):

    def __init__(self, space: gym.Space, shared=False, n_agents=-1):
        self.__shared = shared
        self.__n_agents = n_agents
        if shared and n_agents == -1:
            raise ValueError(
                "n_agents must be given to a shared action space.")
        self.__space = space
        assert isinstance(space, gym.spaces.Box) and len(
            space.shape) == 1, type(space)

    @property
    def n(self):
        return self.__space.shape[0]

    def sample(self) -> ContinuousAction:
        if self.__shared:
            x = torch.stack(
                [self.__space.sample() for _ in range(self.__n_agents)])
        else:
            x = self.__space.sample()
        return ContinuousAction(x)


@namedarray
class BasicObservation(env_base.Observation):
    obs: torch.Tensor


class BasicObservationSpace:

    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def sample(self):
        return BasicObservation(torch.randn(*self._shape))


class DictObservationSpace:

    def __init__(self, shapes: Dict, obs_cls):
        self._shapes = shapes
        self._obs_cls = obs_cls

    @property
    def shape(self):
        return self._shapes

    def sample(self):
        return self._obs_cls(
            **{k: torch.randn(*v)
               for k, v in self.shape.items()})
