from gfootball.env import create_environment
import copy
import gym
import numpy as np
import os

import environment.env_base
from environment.env_utils import BasicObservation
from utils.namedarray import namedarray

map_agent_registry = {
    # evn_name: (left, right, game_length, total env steps)
    "11_vs_11_competition": (11, 11, 3000, None),
    "11_vs_11_easy_stochastic": (11, 11, 3000, None),
    "11_vs_11_hard_stochastic": (11, 11, 3000, None),
    "11_vs_11_kaggle": (11, 11, 3000, None),
    "11_vs_11_stochastic": (11, 11, 3000, None),
    "1_vs_1_easy": (1, 1, 500, None),
    "5_vs_5": (5, 5, 3000, None),
    "academy_3_vs_1_with_keeper": (3, 2, 400, int(25e6)),
    "academy_corner": (11, 11, 400, int(50e6)),
    "academy_counterattack_easy": (11, 11, 400, int(25e6)),
    "academy_counterattack_hard": (11, 11, 400, int(50e6)),
    "academy_run_pass_and_shoot_with_keeper": (3, 2, 400, int(25e6)),
    "academy_pass_and_shoot_with_keeper": (3, 2, 400, int(25e6)),
}


class FootballEnvironment:
    """A wrapper of google football environment
    """

    def seed(self, seed):
        self.__env.seed(seed)

    def __init__(self, seed=None, share_reward=False, **kwargs):
        self.__env_name = kwargs["env_name"]
        self.__step_limit = map_agent_registry[self.__env_name][-1]
        self.__representation = "simple115v2"
        kwargs['representation'] = self.__representation
        self.control_left = map_agent_registry[self.__env_name][0]
        self.control_right = 0
        self.__render = kwargs.get("render", False)
        self.__env = create_environment(
            number_of_left_players_agent_controls=self.control_left,
            number_of_right_players_agent_controls=self.control_right,
            **kwargs)
        self.seed(seed)
        self.__share_reward = share_reward

        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros((self.num_agents, 1),
                                         dtype=np.float32)

    @property
    def n_agents(self):
        return self.num_agents

    @property
    def num_agents(self) -> int:
        return self.control_left + self.control_right

    @property
    def observation_spaces(self):
        return [
            environment.env_utils.BasicObservationSpace((115, ))
            for _ in range(self.num_agents)
        ]

    @property
    def action_spaces(self):
        return [
            self.__env.action_space[0]
            if self.num_agents > 1 else self.__env.action_space
            for _ in range(self.num_agents)
        ]

    def reset(self):
        obs = self.__env.reset()
        self.__step_count[:] = self.__episode_return[:] = 0
        obs, _ = self.__post_process_obs_and_rew(obs,
                                                 np.zeros(self.num_agents))
        available_actions = np.ones(
            (obs.shape[0], self.__env.action_space[0].n), dtype=np.uint8)
        return BasicObservation(obs)

    def __post_process_obs_and_rew(self, obs, reward):
        if self.num_agents == 1:
            obs = obs[np.newaxis, :]
            reward = [reward]
        if self.__representation == "extracted":
            obs = np.swapaxes(obs, 1, 3)
        if self.__representation in ("simple115", "simple115v2"):
            obs[obs == -1] = 0
        if self.__share_reward:
            left_reward = np.mean(reward[:self.control_left])
            if self.control_right > 0:
                right_reward = np.mean(reward[self.control_left:])
            else:
                right_reward = 0
            reward = np.array([left_reward] * self.control_left +
                              [right_reward] * self.control_right)
        return obs, reward

    def step(self, actions):
        assert len(actions) == self.num_agents, len(actions)
        obs, reward, done, info = self.__env.step([int(a) for a in actions])
        obs, reward = self.__post_process_obs_and_rew(obs, reward)
        self.__step_count += 1
        self.__episode_return += reward[:, np.newaxis]
        available_actions = np.ones(
            (obs.shape[0], self.__env.action_space[0].n), dtype=np.uint8)
        info['episode'] = dict(r=self.__episode_return.mean().item(),
                               l=self.__step_count.item())
        info['bad_transition'] = (
            done and self.__step_count.item() >= self.__step_limit)
        return (
            BasicObservation(obs),
            np.array(reward[:, None], dtype=np.float32),
            np.array([done for _ in range(self.num_agents)], dtype=np.uint8),
            [copy.deepcopy(info) for _ in range(self.num_agents)],
        )

    def render(self, mode='human'):
        return self.__env.render(mode=mode)

    def close(self):
        self.__env.close()


environment.env_base.register("football", FootballEnvironment)
