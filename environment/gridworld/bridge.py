from typing import List, Optional, Union, Dict
import copy
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image

from utils.namedarray import namedarray, recursive_aggregate
import environment.env_base as env_base
import environment.env_utils as env_utils

# yapf: disable
MAPS = {

    0: [
    'AABAA',
    'AAAAA',
    'AABAA'],

    1: [
    'AAABBBBBBBBBBBBBAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAAAABBBBBBBBBAAAAA',
    '22AAAABBBBBBBAAAA11',
    '222AAAAAAAAAAAAA111',
    '222AAAAAAAAAAAAA111',
    '22AAAABBBBBBBAAAA11',
    'AAAAABBBBBBBBBAAAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAABBBBBBBBBBBBBAAA'],

    2: [
    'AAABBBBBBBBBBBBBAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAAAABBBBBBBBBAAAAA',
    '22AAAABBBBBBBAAAA11',
    '222AAAAABAAAAAAA111',
    '222AAAAAAABAAAAA111',
    '22AAAABBBBBBBAAAA11',
    'AAAAABBBBBBBBBAAAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAABBBBBBBBBBBBBAAA'],

    3: [
    'AAABBBBBBBBBBBBBAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAAAABBBBBBBBBAAAAA',
    '22AAAABBBBBBBAAAA11',
    '222AAARRRRRRRAAA111',
    '222AAARRRRRRRAAA111',
    '22AAAABBBBBBBAAAA11',
    'AAAAABBBBBBBBBAAAAA',
    'AAAABBBBBBBBBBBAAAA',
    'AAABBBBBBBBBBBBBAAA'],

    4: [
    '1ABBAA',
    'AAAAAA',
    'AABBA2',],

    5: [
    '1ABBA2',
    'AAAAAA',
    '1ABBA2',],
} # ACCESSIBLE, BLOCK


# yapf: enable
def split_to_shapes(x: np.ndarray, shapes: Dict, axis: int = -1):
    """Split an array and reshape to desired shapes.

    Args:
        x (np.ndarray): The array to be splitted
        shapes (Dict): Dict of shapes (tuples) specifying how to split.
        axis (int): Split dimension.

    Returns:
        List: Splitted observations.
    """
    axis = len(x.shape) + axis if axis < 0 else axis
    split_lengths = [np.prod(shape) for shape in shapes.values()]
    assert x.shape[axis] == sum(split_lengths)
    accum_split_lengths = [
        sum(split_lengths[:i]) for i in range(1, len(split_lengths))
    ]
    splitted_x = np.split(x, accum_split_lengths, axis)
    return {
        k: x.reshape(*x.shape[:axis], *shape, *x.shape[axis + 1:])
        for x, (k, shape) in zip(splitted_x, shapes.items())
    }


@namedarray
class BridgeAgentSpecificObs(env_base.Observation):
    obs_self: np.ndarray
    obs_allies: np.ndarray
    obs_mask: np.ndarray


class BridgeAgentSpecificObservationSpace(env_utils.DictObservationSpace):

    def sample(self):
        return BridgeObservation(super().sample())


@namedarray
class BridgeObservation(env_base.Observation):
    obs: Union[np.ndarray, BridgeAgentSpecificObs]


# yapf: enable
CHANNELS = {
    0: [1, 1],
    1: [2, 7],
}  # W, H
ACTION = {
    0: np.array([0, 0]),  # stay
    1: np.array([-1, 0]),  # left
    2: np.array([1, 0]),  # right
    3: np.array([0, 1]),  # up
    4: np.array([0, -1])  # down
}
ACTION180 = {
    0: 0,  # stay
    1: 2,  # left
    2: 1,  # right
    3: 4,  # up
    4: 3  # down
}
ACT_DIM = 5
MAP_COLOR = {'A': np.array([255, 255, 255]), 'B': np.array([0, 0, 0])}
AGENT_COLOR = {
    0: np.array([255, 0, 0], dtype=np.uint8),
    1: np.array([0, 0, 255], dtype=np.uint8)
}
GOAL_COLOR = {
    0: np.array([128, 0, 0], dtype=np.uint8),
    1: np.array([0, 0, 128], dtype=np.uint8)
}
VIEW_LEN = 2  # 5  in total


class Agent:

    def __init__(self, idx, group_idx):
        self.state = np.zeros(2)
        self.nex_state = None
        self.action = -1
        self.group_idx = group_idx
        self.goal = np.zeros(2)
        self.done = False
        self.idx = idx
        self.valid_move = False
        self.resolve_move_visited = False
        self.max_history_c = 0


class BridgeEnvironment:

    def __init__(
        self,
        num_agents=4,
        map_type=3,
        num_random_blocks=2,
        num_groups=2,
        agent_specific_obs=False,
        episode_length=50,
        use_image_obs=False,
        use_image_act=False,
        use_clean_shareobs=False,
        use_agent_id=False,
        use_ally_id=False,
        share_reward=False,
        seed=None,
    ):
        self.string_map = MAPS[map_type]
        self.num_random_blocks = num_random_blocks
        self.map_maxr, self.map_maxc = len(self.string_map), len(
            self.string_map[0])
        self.map_center = np.array([self.map_maxr - 1, self.map_maxc - 1],
                                   dtype=np.float32) / 2
        # self.channel_w, self.channel_h = CHANNELS[map_type]
        self.starts, self.blocks = {i + 1: [] for i in range(num_groups)}, []
        self.color_map = np.ones(
            (3, self.map_maxr, self.map_maxc), dtype=np.uint8) * 255
        self.waiting_blocks = []
        for r in range(self.map_maxr):
            for c in range(self.map_maxc):
                char = self.string_map[r][c]
                if char == 'B': self.blocks.append(np.array([r, c]))
                elif '1' <= char <= '9':
                    if int(char) in self.starts.keys():
                        self.starts[int(char)].append(np.array([r, c]))
                elif char == 'R':
                    self.waiting_blocks.append(np.array([r, c]))
        self.__share_reward = share_reward
        self.num_agents = num_agents
        self.num_groups = num_groups
        self.use_image_obs = use_image_obs

        self.agents = [
            Agent(idx=i,
                  group_idx=1 + i // (self.num_agents // self.num_groups))
            for i in range(self.num_agents)
        ]
        assert self.num_agents % self.num_groups == 0
        self.n_agents_per_group = n_agents_per_group = len(
            self.agents) // self.num_groups
        self.grouped_agents = [
            self.agents[i * n_agents_per_group:(i + 1) * n_agents_per_group]
            for i in range(self.num_groups)
        ]
        self.episode_length = episode_length
        self.image_act = use_image_act
        self.clean_shareobs = use_clean_shareobs

        self.current_step = 0
        self.__episode_length = self.__episode_return = 0
        self.seed()

        self.__obs_dim = 2 * (1 + self.num_agents)
        self.__use_agent_id = use_agent_id
        self.__use_ally_id = use_ally_id
        if use_agent_id:
            self.__obs_dim += self.num_agents
        if use_ally_id:
            self.__obs_dim += (self.num_agents - 1) * self.num_agents
        self.__agent_specific_obs = agent_specific_obs
        if self.__agent_specific_obs:
            self.__obs_space = BridgeAgentSpecificObservationSpace(
                {
                    'obs_self':
                    (4 if not self.__use_agent_id else 4 + self.num_agents, ),
                    'obs_allies':
                    (self.num_agents - 1,
                     2 if not self.__use_ally_id else 2 + self.num_agents),
                    'obs_mask': (self.num_agents, ),
                }, BridgeAgentSpecificObs)
        else:
            self.__obs_space = env_utils.BasicObservationSpace(
                (self.__obs_dim, ))

        self.__initiated_render = False

    @property
    def action_spaces(self):
        return [gym.spaces.Discrete(5) for _ in range(self.n_agents)]

    @property
    def observation_spaces(self):
        return [self.__obs_space for _ in range(self.n_agents)]

    @property
    def n_agents(self):
        return self.num_agents

    def _check_block(self, waits, col):
        for c in waits:
            if np.abs(col - c) <= 1:
                return False
        return True

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        plt.clf()
        self.seed()
        self.__initiated_render = False

        # ranodm blocks
        self.random_blocks = []
        waiting_block_cols = []
        while len(self.random_blocks) < self.num_random_blocks:
            idx = np.random.randint(len(self.waiting_blocks))
            col = self.waiting_blocks[idx][1]
            if self._check_block(waiting_block_cols, col):
                waiting_block_cols.append(col)
                self.random_blocks.append(self.waiting_blocks[idx])
        # self.random_blocks = [np.array([4,6]), np.array([5,10])]
        self.total_blocks = [list(i) for i in self.blocks
                             ] + [list(i) for i in self.random_blocks]
        for (r, c) in self.total_blocks:
            self.color_map[:, r, c] = MAP_COLOR['B']

        for agent in self.agents:
            agent.max_history_c = 0
        # print (group_goal_id, group_state_id)
        for group_idx in range(self.num_groups):
            indices = np.random.choice(len(self.starts[group_idx + 1]),
                                       self.n_agents_per_group,
                                       replace=False)
            start_states = [self.starts[group_idx + 1][i] for i in indices]

            indices = np.random.choice(len(
                self.starts[(group_idx + 1) % self.num_groups + 1]),
                                       self.n_agents_per_group,
                                       replace=False)
            goal_states = [
                self.starts[(group_idx + 1) % self.num_groups + 1][i]
                for i in indices
            ]

            for j, (start, goal) in enumerate(zip(start_states, goal_states)):
                agent = self.grouped_agents[group_idx][j]
                assert agent.group_idx == group_idx + 1, (agent.group_idx, j)
                agent.state = start.copy()
                agent.goal = goal.copy()
                agent.done = False

                self.color_map[:, start[0],
                               start[1]] = AGENT_COLOR[agent.group_idx - 1]
                self.color_map[:, goal[0],
                               goal[1]] = GOAL_COLOR[agent.group_idx - 1]

        obs_n = []
        for agent in self.agents:
            if self.use_image_obs:
                obs_n.append(self._set_image_obs(agent))
            else:
                obs_n.append(self._set_state_obs(agent))
        obs = np.stack(obs_n)
        # assert (obs[0] == obs[1]).all(), (obs, [agent.state for agent in self.agents], [agent.goal for agent in self.agents], self.blocks + self.random_blocks)

        self.current_step = 0
        self.__episode_length = 0
        self.__episode_return = 0
        if self.__agent_specific_obs:
            obs = np.concatenate([
                obs,
                np.ones((self.num_agents, self.num_agents), dtype=np.float32)
            ], -1)
            obs = BridgeAgentSpecificObs(
                **split_to_shapes(obs, self.__obs_space.shape))
        return BridgeObservation(obs)

    def _collision(self, agent):
        # boundary
        if agent.nex_state[0] >= self.map_maxr or agent.nex_state[
                1] >= self.map_maxc or agent.nex_state[
                    0] < 0 or agent.nex_state[1] < 0:
            return True
        # wall
        if list(agent.nex_state) in self.total_blocks:
            return True
        # agent
        return False

    def _move_resolved(self, agent, movable):
        if not movable:
            agent.nex_state = agent.state.copy()
            agent.action = 0
        agent.valid_move = True
        agent.resolve_move_visited = True

    def _resolve_move(self, agent):
        if agent.valid_move:
            assert agent.resolve_move_visited
            return
        if agent.done or (agent.nex_state == agent.goal).all():
            self._move_resolved(agent, True)
            return
        if agent.action == 0:
            assert (agent.nex_state == agent.state).all()
            self._move_resolved(agent, True)
            return
        unknown_collisions = []
        for other_agent in self.agents:
            if other_agent is agent or other_agent.done:
                continue
            if other_agent.valid_move:
                if (agent.nex_state == other_agent.nex_state).all():
                    # must collide
                    self._move_resolved(agent, False)
                    return
                else:
                    # otherwise can not collide with this agent
                    pass
            else:
                if (other_agent.state == agent.nex_state).all() and (
                        other_agent.nex_state == agent.nex_state).all():
                    assert other_agent.action == 0
                    self._move_resolved(agent, False)
                    return
                elif (other_agent.nex_state == agent.nex_state).all():
                    assert other_agent.action != 0
                    self._move_resolved(other_agent, False)
                elif (other_agent.state == agent.nex_state).all():
                    if (other_agent.nex_state == agent.state).all():
                        self._move_resolved(agent, False)
                        self._move_resolved(other_agent, False)
                        return
                    else:
                        if not other_agent.resolve_move_visited:
                            unknown_collisions.append(other_agent)
                        else:
                            self._move_resolved(agent, False)
                            return
        if len(unknown_collisions) == 0:
            self._move_resolved(agent, True)
            return
        agent.resolve_move_visited = True
        for other_agent in unknown_collisions:
            self._resolve_move(other_agent)
        self._resolve_move(agent)

    def _set_state_obs(self, agent):
        obs_self = [agent.state, agent.goal]
        obs_others = [
            other.state for other in self.agents if other is not agent
        ]

        # symmetric observation
        if self.num_groups == 2:
            if agent.group_idx == 2:
                tmp = np.array([self.map_maxr - 1, self.map_maxc - 1],
                               dtype=np.float32)
                obs_self = list(map(lambda x: tmp - x, obs_self))
                obs_others = list(map(lambda x: tmp - x, obs_others))
        else:
            raise NotImplementedError

        if self.__use_agent_id:
            obs_self.append(
                np.eye(self.num_agents)[agent.idx].astype(np.float32))
        if self.__use_ally_id:
            ally_ids = [
                np.eye(self.num_agents)[other.idx].astype(np.float32)
                for other in self.agents if other is not agent
            ]
            obs_others = [
                val for pair in zip(obs_others, ally_ids) for val in pair
            ]

        return np.concatenate(obs_self + obs_others)

    def _set_image_obs(self, agent):
        pos_r, pos_c = agent.state[0], agent.state[1]
        obs = np.zeros((3, 2 * VIEW_LEN + 1, 2 * VIEW_LEN + 1), dtype=np.uint8)
        min_r, max_r = max(0, pos_r - VIEW_LEN), min(pos_r + VIEW_LEN + 1,
                                                     self.map_maxr)
        min_c, max_c = max(0, pos_c - VIEW_LEN), min(pos_c + VIEW_LEN + 1,
                                                     self.map_maxc)
        loc_r, loc_c = VIEW_LEN - (pos_r - min_r), VIEW_LEN - (pos_c - min_c)
        # c_len, r_len = max_c-min_c, max_r-min_r
        # print (pos_c, pos_r, min_r, max_r, min_c, max_c, loc_r, loc_c)
        # print (obs[:, loc_r:loc_r+max_r-min_r+1, loc_c:loc_c+max_c-min_c+1].shape)
        # print (self.color_map[:, min_r:max_r, min_c:max_c].shape)
        obs[:, loc_r:loc_r + max_r - min_r,
            loc_c:loc_c + max_c - min_c] = self.color_map[:, min_r:max_r,
                                                          min_c:max_c]
        return obs

    def step(self, actions):
        for agent in self.agents:
            agent.valid_move = False
            agent.resolve_move_visited = False

        for j, agent in enumerate(self.agents):
            act = int(actions[j]) if not agent.done else 0
            # indistinguishable symmetry actions
            if self.num_groups == 2:
                if agent.group_idx == 2:
                    act = ACTION180[act]
            else:
                raise NotImplementedError
            agent.action = act
            agent.nex_state = agent.state + ACTION[act]

            # check block collision
            if self._collision(agent):
                # rwds[agent.idx] -= 1
                self._move_resolved(agent, False)

        for agent in self.agents:
            self._resolve_move(agent)

        # for i, agent in enumerate(self.agents):
        #     if agent.done or (agent.nex_state == agent.goal).all():
        #         continue
        #     for other_agent in self.agents[i + 1:]:
        #         if other_agent.done or (other_agent.nex_state == other_agent.goal).all():
        #             continue
        #         assert not (agent.nex_state == other_agent.nex_state).all(), ([agent.state for agent in self.agents], [agent.action for agent in self.agents], [agent.nex_state for agent in self.agents])

        # apply action
        for agent in self.agents:
            self.color_map[:, agent.state[0],
                           agent.state[1]] = np.ones(3).astype(np.uint8) * 255
            agent.state = agent.nex_state.copy()
            agent.nex_state = None
            self.color_map[:, agent.state[0],
                           agent.state[1]] = AGENT_COLOR[agent.group_idx - 1]

        rwds = np.zeros((self.num_agents, 1), dtype=np.float32)
        obs_n, done_n = [], []
        for i, agent in enumerate(self.agents):
            if not agent.done:
                # r = np.all(agent.state == agent.goal)
                # if not any(agent.done for agent in self.agents) and agent.group_idx == 1:
                #     # if the first group reaches the goal first
                #     eps = 0.0
                #     r *= 0.9 + eps * (2 * np.random.rand() - 1)
                # if self.num_groups == 2:
                #     state = np.array([self.map_maxr - 1, self.map_maxc - 1]) - agent.state if agent.group_idx == 2 else agent.state
                #     if state[1] > agent.max_history_c:
                #         assert state[1] - agent.max_history_c == 1, (state, agent.max_history_c)
                #         agent.max_history_c = state[1]
                #         r += 0.1
                # else:
                #     raise NotImplementedError
                rwds[i] += -np.linalg.norm(agent.goal - agent.state) * 0.01
                if self.use_image_obs:
                    obs = self._set_image_obs(agent)
                else:
                    obs = self._set_state_obs(agent)
            else:
                obs = np.zeros(self.__obs_dim, dtype=np.float32)
            agent.done = (self.current_step + 1
                          == self.episode_length) or (agent.goal
                                                      == agent.state).all()
            obs_n.append(obs)
            done_n.append(np.array([agent.done], dtype=np.uint8))
        self.current_step += 1
        self.__episode_length += 1
        if self.__share_reward:
            rwds[:] = rwds.mean()
        self.__episode_return += rwds.sum()
        obs = np.stack(obs_n)
        if self.__agent_specific_obs:
            obs_mask = np.zeros((self.num_agents, self.num_agents - 1),
                                dtype=np.float32)
            for i, agent in enumerate(self.agents):
                cnt = 0
                for other in self.agents:
                    if other is agent:
                        continue
                    obs_mask[i, cnt] = 1 - other.done
                    cnt += 1
            obs = np.concatenate([
                obs,
                np.ones((self.num_agents, 1), dtype=np.float32), obs_mask
            ], -1)
            obs = BridgeAgentSpecificObs(
                **split_to_shapes(obs, self.__obs_space.shape))
        return (BridgeObservation(obs), rwds * (1 - np.stack(done_n)),
                np.stack(done_n), [
                    dict(episode=dict(l=self.__episode_length,
                                      r=self.__episode_return))
                    for _ in range(self.n_agents)
                ])

    def render(self, mode='human'):
        if not self.__initiated_render:
            plt.ion()
            plt.show()
            plt.margins(0, 0)
            plt.axis('off')
            for i in range(-1, self.map_maxr):
                plt.axhline(y=i + 0.5, color=(0, 0, 0))
            for i in range(self.map_maxc + 1):
                plt.axvline(x=i - 0.5, color=(0, 0, 0))
            self.__initiated_render = True

        # fig = plt.figure('test')
        img = np.ones(
            (self.map_maxr, self.map_maxc, 3), dtype=np.uint8) * int(255)
        for i, b in enumerate(self.total_blocks):
            img[b[0], b[1], :] = 0
        for a in self.agents:
            img[a.state[0], a.state[1], :] = AGENT_COLOR[a.group_idx - 1]
            img[a.goal[0], a.goal[1], :] = GOAL_COLOR[a.group_idx - 1]

        if mode == 'human':
            plt.imshow(img, cmap='gray')
            # plt.show()
            plt.pause(0.001)

        # canvas = FigureCanvasAgg(plt.gcf())
        # canvas.draw()
        # w, h = canvas.get_width_height()
        # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        # buf.shape = (w, h, 4)
        # buf = np.roll(buf, 3, axis=2)
        # img = Image.frombytes("RGBA", (w, h), buf.tostring())
        # img = np.asarray(img)
        # img = img[:, :, :3]
        return img

    def close(self):
        del self.agents


env_base.register('bridge', BridgeEnvironment)
