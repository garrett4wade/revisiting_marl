import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, layernorm=True):
    # refer to https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L15
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        if layernorm:
            layers += [nn.LayerNorm([sizes[j + 1]])]
    return nn.Sequential(*layers)
