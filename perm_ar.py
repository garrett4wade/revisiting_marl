import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm


def moving_average(interval, windowsize):

    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


class AutoregressiveActor(nn.Module):

    def __init__(self, n_agents):
        super().__init__()
        self.net = nn.Sequential(nn.Linear((n_agents - 1) * n_agents, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, n_agents))

    def forward(self, onehot_action, execution_mask):
        x = onehot_action * execution_mask.unsqueeze(-1)
        return self.net(x.view(*x.shape[:-2], -1))


def generate_mask_from_order(agent_order, ego_exclusive):
    """Generate execution mask from agent order.

    Used during autoregressive training.

    Args:
        agent_order (torch.Tensor): Agent order of shape [*, n_agents].

    Returns:
        torch.Tensor: Execution mask of shape [*, n_agents, n_agents - 1].
    """
    shape = agent_order.shape
    n_agents = shape[-1]
    agent_order = agent_order.view(-1, n_agents)
    bs = agent_order.shape[0]

    cur_execution_mask = torch.zeros(bs, n_agents).to(agent_order)
    all_execution_mask = torch.zeros(bs, n_agents, n_agents).to(agent_order)

    batch_indices = torch.arange(bs)
    for i in range(n_agents):
        agent_indices = agent_order[:, i].long()

        cur_execution_mask[batch_indices, agent_indices] = 1
        all_execution_mask[batch_indices, :, agent_indices] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_indices, agent_indices] = 1
    if not ego_exclusive:
        # [*, n_agent, n_agents]
        return all_execution_mask.view(*shape[:-1], n_agents, n_agents)
    else:
        # [*, n_agents, n_agents - 1]
        execution_mask = torch.zeros(bs, n_agents, n_agents - 1).to(agent_order)
        for i in range(n_agents):
            execution_mask[:, i] = torch.cat(
                [all_execution_mask[..., i, :i], all_execution_mask[..., i, i + 1:]], -1)
        return execution_mask.view(*shape[:-1], n_agents, n_agents - 1)


def main(n_agents, from_scrath=True, algos=[]):
    seeds = [1]
    n_epoches = int(4e3)
    smooth_window_size = 10
    bs = 128

    if from_scrath:

        for j, seed in enumerate(seeds):
            torch.manual_seed(seed)

            actor = AutoregressiveActor(n_agents)

            optimizer = torch.optim.SGD(actor.parameters(), lr=1e-1)

            for epoch in tqdm(range(1, n_epoches + 1)):

                # rollout
                with torch.no_grad():
                    actions = torch.zeros(bs, n_agents)
                    action_type_cnt = {}
                    for agent_idx in range(n_agents):

                        ego_exclusive_action = torch.cat(
                            [actions[..., :agent_idx], actions[..., agent_idx + 1:]],
                            -1)  # [bs, n_agents - 1]

                        # select inputs
                        execution_mask = torch.stack([torch.ones(bs)] * agent_idx + [torch.zeros(bs)] *
                                                     (n_agents - 1 - agent_idx), -1)  # [bs, n_agents - 1]
                        onehot_action = F.one_hot(ego_exclusive_action.long(),
                                                  n_agents).float()  # [bs, n_agents - 1, n_agents]

                        logits = actor(onehot_action, execution_mask)
                        dist = torch.distributions.Categorical(logits=logits)

                        actions[:, agent_idx] = dist.sample()  # [bs]
                    rewards = torch.zeros((bs, 1), dtype=torch.float32)
                    for i in range(bs):
                        rewards[i] = (n_agents == len(torch.unique(actions[i].flatten())))

                # training
                ego_exclusive_action = [
                    torch.cat([actions[..., :i], actions[..., i + 1:]], -1) for i in range(n_agents)
                ]
                ego_exclusive_action = torch.stack(ego_exclusive_action, -2)  # [bs, n_agents, n_agents - 1]

                agent_order = np.stack([np.random.permutation(n_agents) for _ in range(bs)])
                agent_order = torch.from_numpy(agent_order)
                execution_mask = generate_mask_from_order(
                    agent_order, ego_exclusive=True).float()  # [T, bs, n_agents, n_agents - 1]

                onehot_action = F.one_hot(ego_exclusive_action.long(), n_agents).float()

                logits = actor(onehot_action, execution_mask)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)

                loss = (-log_probs * rewards).mean()

                logits_ = actor(torch.zeros_like(onehot_action), torch.zeros_like(execution_mask))
                dist_ = torch.distributions.Categorical(logits=logits_)

                loss += (-0.1 * dist_.entropy()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for action, reward in zip(actions, rewards):
                    if reward == 1:
                        key = ''.join([str(int(s)) for s in action.numpy().tolist()])
                        if key not in action_type_cnt:
                            action_type_cnt[key] = 1
                        else:
                            action_type_cnt[key] += 1

                # print(rewards.mean().item(), len(action_type_cnt), list(action_type_cnt.values()))
                assert sum(action_type_cnt.values()) == rewards.mean().item() * bs

            print("Training finishes! Generating heatmap...")
            # generate heatmap
            bs = 1000
            ar_heatarray = np.zeros((16, 16), dtype=np.float32)
            axis_tuples = list(itertools.product(list(range(4)), list(range(4))))
            entropies = []
            with torch.no_grad():
                actions = torch.zeros(bs, n_agents)
                for agent_idx in range(n_agents):

                    ego_exclusive_action = torch.cat([actions[..., :agent_idx], actions[..., agent_idx + 1:]],
                                                     -1)  # [bs, n_agents - 1]

                    # select inputs
                    execution_mask = torch.stack([torch.ones(bs)] * agent_idx + [torch.zeros(bs)] *
                                                 (n_agents - 1 - agent_idx), -1)  # [bs, n_agents - 1]
                    onehot_action = F.one_hot(ego_exclusive_action.long(),
                                              n_agents).float()  # [bs, n_agents - 1, n_agents]

                    logits = actor(onehot_action, execution_mask)
                    dist = torch.distributions.Categorical(logits=logits)

                    actions[:, agent_idx] = dist.sample()  # [bs]

                    entropies.append([agent_idx + 1, float(dist.entropy().mean()), 'ar'])

            for action in actions:
                action_ints = [int(a.item()) for a in action]
                idx1 = axis_tuples.index((action_ints[0], action_ints[1]))
                idx2 = axis_tuples.index((action_ints[2], action_ints[3]))
                ar_heatarray[idx1, idx2] += 1

            with open('ar_perm_heatarray.npy', 'wb') as f:
                np.save(f, ar_heatarray)
            with open('ar_perm_entropy.npy', 'wb') as f:
                np.save(f, np.array(entropies))

    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    df = pd.DataFrame(entropies, columns=['index', 'entropy', 'policy'])
    fig = sns.lineplot(data=df,
                       x='index',
                       y='entropy',
                       hue='policy',
                       err_style="bars",
                       ci=95,
                       marker='o',
                       linewidth=5,
                       markersize=10)
    # plt.yticks([0.0, math.log(2), math.log(3), math.log(4)])
    plt.xticks([1, 2, 3, 4])
    plt.xlabel('Agent Execution Index', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.axhline(math.log(2), ls='--', lw=2)
    plt.axhline(math.log(3), ls='--', lw=2)
    plt.axhline(math.log(4), ls='--', lw=2)
    plt.legend(labels=['AR', 'Ind.'], loc='upper right', fontsize=16)
    fig.get_figure().savefig('ar_perm_entropy.png')

    sns.set(font_scale=0.8)
    ar_heatarray = np.load('ar_perm_heatarray.npy')
    plt.figure(figsize=(10, 8))
    fig = sns.heatmap(ar_heatarray / 1000,
                      cmap="YlGnBu",
                      cbar=True,
                      annot=True,
                      fmt='.2f',
                      yticklabels=False,
                      xticklabels=False)
    fig.get_figure().savefig('ar_perm_payoff.png')


if __name__ == '__main__':
    main(4, True)
