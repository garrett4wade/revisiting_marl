from abc import ABC
from typing import Any, List, Union, Dict
import copy
import dataclasses
import numpy as np
import torch.distributed as dist
import torch.nn

from utils.namedarray import namedarray
from environment.env_base import Observation, Action


class PolicyState:
    pass


@namedarray
class RolloutResult:
    action: Action
    log_prob: torch.Tensor
    value: torch.Tensor
    policy_state: PolicyState = None


@namedarray
class RolloutRequest:
    obs: Observation
    policy_state: PolicyState = None
    mask: torch.Tensor = torch.ones(1, dtype=torch.bool)


ALL_POLICY_CLASSES = {}


def register(name, policy_class):
    ALL_POLICY_CLASSES[name] = policy_class


def make(cfg: Dict, observation_space, action_space):
    cls = ALL_POLICY_CLASSES[cfg['type']]
    return cls(observation_space=observation_space,
               action_space=action_space,
               **cfg.get("args", {}))
