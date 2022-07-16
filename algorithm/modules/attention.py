import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CatSelfEmbedding(nn.Module):

    def __init__(self, self_dim, others_shape_dict, d_embedding, num_layers=1, layer_norm=False):
        super().__init__()
        self.self_dim = self_dim
        self.others_shape_dict = others_shape_dict
        self.d_embedding = d_embedding

        def get_layer(input_dim, output_dim):
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
            return nn.Sequential(*layers)

        self.others_keys = sorted(self.others_shape_dict.keys())
        self.self_embedding = get_layer(self_dim, d_embedding)
        for k in self.others_keys:
            if 'mask' not in k:
                setattr(self, k + '_fc', get_layer(others_shape_dict[k][-1] + self_dim, d_embedding))

    def forward(self, self_vec, **inputs):
        other_embeddings = []
        self_embedding = self.self_embedding(self_vec)
        self_vec_ = self_vec.unsqueeze(-2)
        for k, x in inputs.items():
            assert k in self.others_keys
            expand_shape = [-1 for _ in range(len(x.shape))]
            expand_shape[-2] = x.shape[-2]
            x_ = torch.cat([self_vec_.expand(*expand_shape), x], -1)
            other_embeddings.append(getattr(self, k + '_fc')(x_))

        other_embeddings = torch.cat(other_embeddings, dim=-2)
        return self_embedding, other_embeddings


def ScaledDotProductAttention(q, k, v, mask=None, dropout=None):
    """Compute attention given query, key and value.

    Args:
        q (torch.Tensor): Query with shape (*, NUM_HEADS, NUM_TOKENS_1, HEAD_DIM).
        k (torch.Tensor): Key with shape (*, NUM_HEADS, NUM_TOKENS_2, HEAD_DIM).
        v (torch.Tensor): Value with shape (*, NUM_HEADS, NUM_TOKENS_2, HEAD_DIM).
        mask (torch.Tensor, optional): Attention mask with shape (*, NUM_TOKENS_2).
            Defaults to None.
        dropout (torch.nn.Dropout, optional): Optional dropout layer. Defaults to None.

    Returns:
        torch.Tensor: Attention score with the same shape as query.
    """
    d_k = q.shape[-1]
    assert k.shape[-1] == v.shape[-1] == d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (NUM_HEADS, NUM_TOKENS_1, NUM_TOKENS_2)

    if mask is not None:
        mask = mask.unsqueeze(-2).unsqueeze(-2)
        scores = scores - (1 - mask) * 1e10

    # in case of overflow
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    scores = F.softmax(scores, dim=-1)

    if mask is not None:
        # in case of all-zero
        scores = scores * mask

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, input_dim, d_attn, n_heads, dropout=0.0, entry='all'):
        super().__init__()
        assert entry == 'all' or isinstance(entry, int)

        self.d_attn = d_attn
        self.n_heads = n_heads
        if not d_attn % n_heads == 0:
            raise ValueError('Attention model dimension is not a multiple of the number of attention heads.')
        self.d_head = d_attn // n_heads

        self.pre_norm = nn.LayerNorm(input_dim)
        self.q_linear = nn.Linear(input_dim, self.d_attn)
        self.k_linear = nn.Linear(input_dim, self.d_attn)
        self.v_linear = nn.Linear(input_dim, self.d_attn)
        nn.init.normal_(self.q_linear.weight.data, std=math.sqrt(0.125 / input_dim))
        nn.init.normal_(self.k_linear.weight.data, std=math.sqrt(0.125 / input_dim))
        nn.init.normal_(self.v_linear.weight.data, std=math.sqrt(0.125 / input_dim))

        self.attn_dropout = None if dropout == 0.0 else nn.Dropout(dropout)

        self.entry_dim = entry if entry != 'all' else None

    def forward(self, x, mask):
        x = self.pre_norm(x)
        # perform linear operation, split into several heads, and put NUM_TOKENS as the last second dim
        if self.entry_dim is None:
            q = self.q_linear(x).view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)
        else:
            qx = x[..., self.entry_dim: self.entry_dim + 1, :]
            q = self.q_linear(qx).view(*qx.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)
        k = self.k_linear(x).view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)
        v = self.v_linear(x).view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)

        if self.entry_dim is not None and mask is not None:
            mask = mask[..., self.entry_dim: self.entry_dim + 1]
        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, mask, self.attn_dropout)

        if self.entry_dim is None:
            # concatenate heads and put through final linear layer
            return scores.transpose(-2, -3).contiguous().view(*x.shape[:-1], self.d_attn)
        else:
            return scores.transpose(-2, -3).contiguous().view(*x.shape[:-2], self.d_attn)


class ResidualMultiHeadSelfAttention(nn.Module):

    def __init__(self, input_dim, d_attn, n_heads, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(input_dim, d_attn, n_heads, dropout)

        self.dense = nn.Linear(d_attn, d_attn)
        nn.init.normal_(self.dense.weight.data, std=math.sqrt(0.125 / d_attn))
        self.residual_norm = nn.LayerNorm(d_attn)
        self.dropout_after_attn = None if dropout == 0 else nn.Dropout(dropout)

    def forward(self, x, mask):
        scores = self.dense(self.attn(x, mask))
        if self.dropout_after_attn is not None:
            scores = self.dropout_after_attn(scores)
        return self.residual_norm(x + scores)


def masked_avg_pooling(scores, mask=None):
    if mask is None:
        return scores.mean(-2)
    else:
        assert mask.shape[-1] == scores.shape[-2]
        masked_scores = scores * mask.unsqueeze(-1)
        return masked_scores.sum(-2) / (mask.sum(-1, keepdim=True) + 1e-5)
