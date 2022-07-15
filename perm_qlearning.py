import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


def moving_average(interval, windowsize):

    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


class DMAQ_SI_Weight(nn.Module):

    def __init__(self, n_agents):
        super(DMAQ_SI_Weight, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_agents
        self.action_dim = n_agents * self.n_actions

        self.num_kernel = 4

        self.key_extractors = nn.ParameterList()
        self.agents_extractors = nn.ParameterList()
        self.action_extractors = nn.ModuleList()

        for i in range(self.num_kernel):  # multi-head attention
            # if getattr(args, "adv_hypernet_layers", 1) == 1:
            self.key_extractors.append(nn.Parameter(torch.randn(1), requires_grad=True))  # key
            self.agents_extractors.append(nn.Parameter(torch.randn(self.n_agents),
                                                       requires_grad=True))  # agent
            self.action_extractors.append(nn.Linear(self.action_dim, self.n_agents))  # action

    def forward(self, actions):
        data = actions = actions.reshape(-1, self.action_dim)
        bs = data.shape[0]

        all_head_key = [k_ext * torch.ones(bs, 1).to(data) for k_ext in self.key_extractors]
        all_head_agents = [k_ext * torch.ones(bs, self.n_agents).to(data) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents,
                                                                     all_head_action):
            x_key = torch.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1)

        return head_attend


class Qatten_Weight(nn.Module):

    def __init__(self, n_agents):
        super(Qatten_Weight, self).__init__()

        self.name = 'qatten_weight'
        self.n_agents = n_agents
        self.n_actions = n_agents
        self.sa_dim = self.n_agents * self.n_actions
        self.n_head = 4

        self.embed_dim = 64
        self.attend_reg_coef = 0.001

        self.key_extractors = nn.ParameterList()
        self.selector_extractors = nn.ParameterList()

        for i in range(self.n_head):  # multi-head attention
            self.selector_extractors.append(nn.Parameter(torch.randn(self.embed_dim),
                                                         requires_grad=True))  # query
            self.key_extractors.append(nn.Parameter(torch.randn(self.embed_dim), requires_grad=True))  # key

        # V(s) instead of a bias for the last layers
        self.V = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, agent_qs, actions):
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)
        bs = agent_qs.shape[0]

        all_head_selectors = [
            sel_ext * torch.ones(bs, self.embed_dim).to(agent_qs) for sel_ext in self.selector_extractors
        ]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        all_head_keys = [
            k_ext * torch.ones(bs, self.n_agents, self.embed_dim).to(agent_qs)
            for k_ext in self.key_extractors
        ]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = torch.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                         curr_head_keys.permute(0, 2, 1))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = torch.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V * torch.ones(bs, 1).to(agent_qs)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]

        head_attend = torch.sum(head_attend, dim=1)

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit**2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean())
                          for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies


class DMAQ_QattenMixer(nn.Module):

    def __init__(
        self,
        n_agents,
    ):
        super(DMAQ_QattenMixer, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_agents
        self.action_dim = n_agents * self.n_actions

        self.attention_weight = Qatten_Weight(n_agents)
        self.si_weight = DMAQ_SI_Weight(n_agents)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = torch.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, actions, max_q_i):
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        return adv_tot

    def calc(self, agent_qs, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies


class ReplayBuffer:

    def __init__(self, maxsize, n_agents):
        self.__maxsize = maxsize
        self.__pointer = 0
        self.__actions = torch.zeros((maxsize, n_agents), dtype=torch.float32)
        self.__rewards = torch.zeros((maxsize, 1), dtype=torch.float32)

        self.size = 0

    def put(self, action, reward):
        bs = action.shape[0]
        self.__actions[self.__pointer:self.__pointer + bs] = action
        self.__rewards[self.__pointer:self.__pointer + bs] = reward
        self.__pointer = (self.__pointer + bs) % self.__maxsize
        self.size = min(self.size + bs, self.__maxsize)

    def get(self, bs):
        idx = np.random.choice(self.size, bs, replace=False)
        return self.__actions[idx], self.__rewards[idx]


def main(n_agents, from_scrath=True, algos=[], eps_greedy=False):
    assert len(algos) > 0

    seeds = [1, 2, 3, 4, 5, 6]
    n_epoches = int(2e3) if not eps_greedy else int(4e3)
    losses = torch.zeros(len(seeds), n_epoches)
    smooth_window_size = 10
    bs = 128
    lr = 0.1

    if from_scrath:
        payoff = torch.zeros(4, dtype=torch.float32)
        payoff[1] = payoff[2] = 1

        if 'vdn' in algos:
            for j, seed in enumerate(seeds):
                torch.manual_seed(seed)
                buffer = ReplayBuffer(int(1e6), n_agents)

                local_q = torch.randn((n_agents, n_agents), requires_grad=True)
                mixer = torch.randn((n_agents, 1), requires_grad=True)

                optimizer = torch.optim.SGD([local_q, mixer], lr=lr)

                for epoch in range(1, n_epoches + 1):
                    if eps_greedy:
                        eps = max(0.05, 1.0 - epoch / 1000)
                    else:
                        # random data, used to validate the claim that VD cannot represent the payoff matrix
                        eps = 1

                    with torch.no_grad():
                        greedy_actions = local_q.argmax(-1).unsqueeze(0)
                        random_actions = torch.randint(0, n_agents, (bs, n_agents))
                        mask = (torch.rand(bs, n_agents) < eps).float()
                        actions = mask * random_actions + (1 - mask) * greedy_actions

                        rewards = torch.zeros((bs, 1), dtype=torch.float32)
                        for i in range(bs):
                            rewards[i] = (n_agents == len(torch.unique(actions[i].flatten())))
                        buffer.put(actions, rewards)

                    actions, rewards = buffer.get(bs)
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1, actions.unsqueeze(-1).long())
                    q_tot = torch.matmul(q.squeeze(-1), mixer.abs()).flatten()

                    loss = (rewards.flatten() - q_tot).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    actions = torch.from_numpy(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).to(torch.int64)
                    bs = 4
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1,
                                                                     actions.unsqueeze(-1).long()).squeeze(-1)
                    q_tot = torch.matmul(q.squeeze(-1), mixer.abs()).flatten()
                    assert q_tot.shape == (4,)
                    losses[j, epoch - 1] = (q_tot - payoff).square().sum().detach()
                losses[j] = torch.from_numpy(moving_average(losses[j].numpy(), smooth_window_size))
                print("Algorithm: VDN, Seed: {}, Fitted Qtot: {}".format(seed, q_tot.detach().flatten()))

            with open('vdn_perm_loss.npy', 'wb') as f:
                np.save(f, losses.numpy())

        if 'qmix' in algos:
            for j, seed in enumerate(seeds):
                torch.manual_seed(seed)
                buffer = ReplayBuffer(int(1e6), n_agents)

                local_q = torch.randn((n_agents, n_agents), requires_grad=True)
                mixer_w1 = torch.randn((n_agents, 64), requires_grad=True)
                mixer_b1 = torch.zeros(64, requires_grad=True)
                mixer_w2 = torch.randn((64, 1), requires_grad=True)

                optimizer = torch.optim.SGD([local_q, mixer_w1, mixer_w2, mixer_b1], lr=lr)

                for epoch in range(1, n_epoches + 1):
                    if eps_greedy:
                        eps = max(0.05, 1.0 - epoch / 1000)
                    else:
                        # random data, used to validate the claim that VD cannot represent the payoff matrix
                        eps = 1

                    with torch.no_grad():
                        greedy_actions = local_q.argmax(-1).unsqueeze(0)
                        random_actions = torch.randint(0, n_agents, (bs, n_agents))
                        mask = (torch.rand(bs, n_agents) < eps).float()
                        actions = mask * random_actions + (1 - mask) * greedy_actions

                        rewards = torch.zeros((bs, 1), dtype=torch.float32)
                        for i in range(bs):
                            rewards[i] = (n_agents == len(torch.unique(actions[i].flatten())))
                        buffer.put(actions, rewards)

                    actions, rewards = buffer.get(bs)
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1, actions.unsqueeze(-1).long())
                    q_tot = torch.matmul(q.squeeze(-1), mixer_w1.abs() * 0.01) + mixer_b1
                    q_tot = torch.matmul(F.elu(q_tot), mixer_w2.abs() * 0.01)

                    loss = (rewards.flatten() - q_tot).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    actions = torch.from_numpy(np.array([[0, 0], [0, 1], [1, 0],
                                                         [1, 1]])).to(local_q.data.device).to(torch.int64)
                    bs = 4
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1,
                                                                     actions.unsqueeze(-1).long()).squeeze(-1)
                    q_tot = torch.matmul(q.squeeze(-1), mixer_w1.abs() * 0.01) + mixer_b1
                    q_tot = torch.matmul(F.elu(q_tot), mixer_w2.abs() * 0.01).flatten()

                    assert q_tot.shape == (4,)
                    losses[j, epoch - 1] = (q_tot - payoff).square().sum().detach()
                losses[j] = torch.from_numpy(moving_average(losses[j].numpy(), smooth_window_size))

                print("Algorithm: QMIX, Seed: {}, Fitted Qtot: {}".format(seed, q_tot.detach().flatten()))

            with open('qmix_perm_loss.npy', 'wb') as f:
                np.save(f, losses.numpy())

        if 'qplex' in algos:
            for j, seed in enumerate(seeds):
                torch.manual_seed(seed)
                buffer = ReplayBuffer(int(1e6), n_agents)

                local_q = torch.randn((n_agents, n_agents), requires_grad=True)
                mixer = DMAQ_QattenMixer(n_agents)

                optimizer = torch.optim.SGD(list(mixer.parameters()) + [local_q], lr=lr)

                for epoch in range(1, n_epoches + 1):
                    if eps_greedy:
                        eps = max(0.05, 1.0 - epoch / 1000)
                    else:
                        # random data, used to validate the claim that VD cannot represent the payoff matrix
                        eps = 1

                    with torch.no_grad():
                        greedy_actions = local_q.argmax(-1).unsqueeze(0)
                        random_actions = torch.randint(0, n_agents, (bs, n_agents))
                        mask = (torch.rand(bs, n_agents) < eps).float()
                        actions = mask * random_actions + (1 - mask) * greedy_actions

                        rewards = torch.zeros((bs, 1), dtype=torch.float32)
                        for i in range(bs):
                            rewards[i] = (n_agents == len(torch.unique(actions[i].flatten())))
                        buffer.put(actions, rewards)

                    actions, rewards = buffer.get(bs)
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1,
                                                                     actions.unsqueeze(-1).long()).squeeze(-1)
                    v_tot, _, _ = mixer(q, is_v=True)
                    adv_tot, _, _ = mixer(q,
                                          F.one_hot(actions.squeeze(-1).long(), n_agents).float(),
                                          local_q.max(-1, keepdim=True)[0].repeat(bs, 1),
                                          is_v=False)
                    q_tot = v_tot + adv_tot
                    # print(q_tot, actions)

                    loss = (rewards.flatten() - q_tot.flatten()).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    actions = torch.from_numpy(np.array([[0, 0], [0, 1], [1, 0],
                                                         [1, 1]])).to(local_q.data.device).to(torch.int64)
                    bs = 4
                    q = local_q.unsqueeze(0).repeat(bs, 1, 1).gather(-1,
                                                                     actions.unsqueeze(-1).long()).squeeze(-1)
                    v_tot, _, _ = mixer(q, is_v=True)
                    adv_tot, _, _ = mixer(q,
                                          F.one_hot(actions.squeeze(-1).long(), n_agents).float(),
                                          local_q.max(-1, keepdim=True)[0].repeat(bs, 1),
                                          is_v=False)
                    q_tot = (v_tot + adv_tot).flatten()

                    assert q_tot.shape == (4,)
                    losses[j, epoch - 1] = (q_tot - payoff).square().sum().detach()

                losses[j] = torch.from_numpy(moving_average(losses[j].numpy(), smooth_window_size))
                # input()

                print("Algorithm: QPLEX, Seed: {}, Fitted Qtot: {}".format(seed, q_tot.detach().flatten()))

            with open('qplex_perm_loss.npy', 'wb') as f:
                np.save(f, losses.numpy())

    sns.set(font_scale=1.6)
    example_losses = None
    if 'vdn' in algos:
        vdn_losses = np.load('vdn_perm_loss.npy')[:, smooth_window_size:-smooth_window_size].flatten()
    if 'qmix' in algos:
        qmix_losses = np.load('qmix_perm_loss.npy')[:, smooth_window_size:-smooth_window_size].flatten()
    if 'qplex' in algos:
        qplex_losses = np.load('qplex_perm_loss.npy')[:, smooth_window_size:-smooth_window_size].flatten()

    steps = np.tile(np.arange(1, len(qplex_losses) // len(seeds) + 1), len(seeds))
    seeds = np.array(list(seeds)).reshape(-1, 1)
    seeds = np.tile(seeds, (1, len(qplex_losses) // len(seeds))).reshape(-1)

    methods = []
    losses = []
    if 'vdn' in algos:
        methods += ['VDN' for _ in range(len(seeds))]
        losses += [vdn_losses]
    if 'qmix' in algos:
        methods += ['QMIX' for _ in range(len(seeds))]
        losses += [qmix_losses]
    if 'qplex' in algos:
        methods += ['QPLEX' for _ in range(len(seeds))]
        losses += [qplex_losses]

    df = pd.DataFrame({
        'step': np.concatenate([steps for _ in range(len(algos))]),
        'loss': np.concatenate(losses),
        'seed': np.concatenate([seeds for _ in range(len(algos))]),
        'method': methods
    })
    sns.set_style('whitegrid')
    plt.ylim(-0.2, 2.0)
    # plt.xlabel('training step', fontsize=16)
    # plt.ylabel('loss', fontsize=16)
    ax = sns.lineplot(data=df, x='step', y='loss', hue='method', lw=3)
    plt.gcf().subplots_adjust(bottom=0.15)

    if eps_greedy:
        ax.get_figure().savefig('xor_vd_loss_eps_greedy.png')
    else:
        ax.get_figure().savefig('xor_vd_loss.png')


if __name__ == '__main__':
    main(2, True, ['qplex', 'vdn', 'qmix'], True)
