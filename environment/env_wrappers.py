import dataclasses
import gym
import multiprocessing as mp
import numpy as np
import torch
import queue

from algorithm.trainer import SampleBatch
from utils.namedarray import recursive_aggregate, recursive_apply
import environment.env_base as env_base


class TorchTensorWrapper(gym.Wrapper):

    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self._device = device

    def _to_tensor(self, x):
        return recursive_apply(x,
                               lambda y: torch.from_numpy(y).to(self._device))

    def step(self, action):
        obs, r, d, info = self.env.step(action.cpu().numpy())
        return (*list(
            map(lambda x: recursive_apply(x, self._to_tensor), [obs, r, d])),
                info)

    def reset(self):
        return recursive_apply(self.env.reset(), self._to_tensor)


@dataclasses.dataclass
class EnvironmentControl:
    act_ready: mp.Semaphore
    obs_ready: mp.Semaphore
    exit_: mp.Event


def shared_env_worker(rank, environment_configs, env_ctrl: EnvironmentControl,
                      storage: SampleBatch, info_queue: mp.Queue):

    def _check_shm(x):
        assert isinstance(x, torch.Tensor) and x.is_shared

    recursive_apply(storage, _check_shm)

    # TODO: seed
    envs = [
        TorchTensorWrapper(env_base.make(environment_config))
        for environment_config in environment_configs
    ]
    offset = rank * len(envs)

    for i, env in enumerate(envs):
        obs = env.reset()
        storage.obs[0, offset + i] = obs
    env_ctrl.obs_ready.release()

    while not env_ctrl.exit_.is_set():

        if env_ctrl.act_ready.acquire(timeout=0.1):
            step = storage.step
            for i, env in enumerate(envs):
                act = storage.actions[step - 1, offset + i]
                obs, reward, done, info = env.step(act)
                if done.all():
                    obs = env.reset()
                    try:
                        info_queue.put_nowait(info[0])
                    except queue.Full:
                        pass

                done_env = done.all(1, keepdim=True).float()
                mask = 1 - done_env

                active_mask = 1 - done
                active_mask = active_mask * (1 - done_env) + done_env

                bad_mask = torch.tensor(
                    [[0.0] if info_.get('bad_transition') else [1.0]
                     for info_ in info],
                    dtype=torch.float32)

                storage.obs[step, offset + i] = obs
                storage.rewards[step - 1, offset + i] = reward
                storage.masks[step, offset + i] = mask
                storage.active_masks[step, offset + i] = active_mask
                storage.bad_masks[step, offset + i] = bad_mask

            env_ctrl.obs_ready.release()
