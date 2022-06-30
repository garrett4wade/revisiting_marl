from typing import Union
import dataclasses
import numpy as np
import torch

from algorithm.policy import PolicyState
from environment.env_base import Observation, Action
from utils.namedarray import namedarray, recursive_apply


@namedarray
class SampleBatch:

    @dataclasses.dataclass
    class MetaData:
        step: torch.tensor = torch.tensor(0, dtype=torch.long).share_memory_()

    obs: Observation
    value_preds: torch.Tensor
    actions: Union[Action, torch.Tensor]
    action_log_probs: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    active_masks: torch.Tensor
    bad_masks: torch.Tensor
    policy_state: PolicyState = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    metadata: MetaData = MetaData()


def feed_forward_generator(sample, num_mini_batch, shared=True):
    # TODO: support separate policy
    assert shared
    batch_size = np.prod(sample.masks.shape[:3])

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
    sample = recursive_apply(sample, lambda x: x.flatten(end_dim=2))
    for indices in sampler:
        yield sample[indices]


def recurrent_generator(sample,
                        num_mini_batch,
                        data_chunk_length,
                        shared=True):
    # TODO: support separate policy
    assert shared
    episode_length, n_rollout_threads, num_agents = sample.masks.shape[:3]
    num_chunks = episode_length // data_chunk_length
    assert data_chunk_length <= episode_length and episode_length % data_chunk_length == 0
    data_chunks = n_rollout_threads * num_chunks * num_agents  # [C=r*T/L]
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
    sample = recursive_apply(sample, _cast)  # [T, B, *]

    for indices in sampler:
        yield sample[:, indices]