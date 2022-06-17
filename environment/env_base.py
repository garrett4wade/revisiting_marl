"""Abstraction of the RL environment and related concepts.

This is basically a clone of the gym interface. The reasons of replicating are:
- Allow easy changing of APIs when necessary.
- Avoid hard dependency on gym.
"""
from typing import List, Optional, Any, Union, Dict
import dataclasses
import torch


class Action:
    pass


class ActionSpace:

    def sample(self) -> Action:
        raise NotImplementedError()


class Observation:
    pass


class EpisodeInfo:
    pass


@dataclasses.dataclass
class StepResult:
    """Step result for a single agent. In multi-agent scenario, env.step() essentially returns
    List[StepResult].
    """
    obs: Observation
    reward: torch.tensor
    done: torch.tensor
    info: Optional[EpisodeInfo]


class Environment:

    @property
    def observation_spaces(self) -> List[dict]:
        raise NotImplementedError()

    @property
    def action_spaces(self) -> List[ActionSpace]:
        raise NotImplementedError()

    def reset(self) -> List[StepResult]:
        raise NotImplementedError()

    def step(self, actions: List[Action]) -> List[StepResult]:
        raise NotImplementedError()

    def render(self) -> None:
        pass

    def seed(self, seed=None):
        raise NotImplementedError()


ALL_ENVIRONMENT_CLASSES = {}


def register(name, env_class):
    ALL_ENVIRONMENT_CLASSES[name] = env_class


def make(cfg: Dict, split='train') -> Environment:
    env_type_ = cfg['type']
    if env_type_ == 'football':
        import environment.football.football_env
    cls = ALL_ENVIRONMENT_CLASSES[env_type_]
    if cfg['args']:
        args = {**cfg['args'].get("base", {}), **cfg['args'].get(split, {})}
    else:
        args = {}
    return cls(**args)
