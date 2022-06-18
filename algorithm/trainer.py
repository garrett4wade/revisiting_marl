from typing import Union
import torch

from algorithm.policy import PolicyState
from environment.env_base import Observation, Action
from utils.namedarray import namedarray


@namedarray
class SampleBatch:
    obs: Observation
    value_preds: torch.Tensor
    returns: torch.Tensor
    actions: Union[Action, torch.Tensor]
    action_log_probs: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    active_masks: torch.Tensor
    bad_masks: torch.Tensor
    policy_state: PolicyState = None
    advantages: torch.Tensor = None