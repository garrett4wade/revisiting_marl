import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.modules.attention import MultiHeadSelfAttention


def get_layer(input_dim, output_dim, num_layers=1, layer_norm=False):
    layers = []
    for i in range(num_layers):
        l = nn.Linear(input_dim, output_dim)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        nn.init.zeros_(l.bias.data)
        if i == 0:
            layers += [l, nn.ReLU(inplace=True)]
        else:
            layers += [l, nn.ReLU(inplace=True)]
        if layer_norm:
            layers += [nn.LayerNorm([output_dim])]
    return nn.Sequential(nn.LayerNorm(input_dim), *layers)


class ARAgentwiseObsEncoder(nn.Module):

    def __init__(self, obs_shapes, act_dim, hidden_dim):
        super().__init__()

        for k, shape in obs_shapes.items():
            if 'mask' not in k:
                setattr(self, k + '_embedding',
                        get_layer(shape[-1], hidden_dim))
        self.act_embedding = get_layer(act_dim, hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, 4, entry=0)
        l = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(l.bias.data)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        self.dense = nn.Sequential(nn.LayerNorm(hidden_dim), l,
                                   nn.ReLU(inplace=True),
                                   nn.LayerNorm(hidden_dim))

        self.policy_head = nn.Linear(hidden_dim, act_dim)
        # policy head should have a smaller scale
        nn.init.orthogonal_(self.policy_head.weight.data, gain=0.01)

    def forward(self, obs, onehot_action, execution_mask=None):
        obs_ = {}
        for k, x in obs.items():
            if 'mask' not in k:
                assert hasattr(self, k + '_embedding')
                obs_[k] = getattr(self, k + '_embedding')(x)
            else:
                assert k == 'obs_mask'
        obs_['obs_self'] = obs_['obs_self'].unsqueeze(-2)
        x = obs_embedding = torch.cat(list(obs_.values()), -2)

        if execution_mask is None:
            x = self.dense(self.attn(x, mask=None))
        else:
            act_embedding = self.act_embedding(onehot_action)
            delta = torch.cat([
                torch.zeros(*act_embedding.shape[:-2], 1,
                            act_embedding.shape[-1]).to(act_embedding),
                act_embedding * execution_mask.unsqueeze(-1)
            ], -2)

            if 'obs_mask' in obs.keys():
                x = self.dense(self.attn(x - delta, mask=obs.obs_mask))
            else:
                x = self.dense(self.attn(x - delta, mask=None))
        return self.policy_head(x)


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
        all_execution_mask[batch_indices, :,
                           agent_indices] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_indices, agent_indices] = 1
    if not ego_exclusive:
        # [*, n_agent, n_agents]
        return all_execution_mask.view(*shape[:-1], n_agents, n_agents)
    else:
        # [*, n_agents, n_agents - 1]
        execution_mask = torch.zeros(bs, n_agents,
                                     n_agents - 1).to(agent_order)
        for i in range(n_agents):
            execution_mask[:, i] = torch.cat([
                all_execution_mask[..., i, :i], all_execution_mask[..., i,
                                                                   i + 1:]
            ], -1)
        return execution_mask.view(*shape[:-1], n_agents, n_agents - 1)
