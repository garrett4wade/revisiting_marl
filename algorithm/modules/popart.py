import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd(nn.Module):

    def __init__(self, input_shape, beta=0.999, epsilon=1e-5):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon
        self.__input_shape = input_shape

        self.__mean = nn.Parameter(torch.zeros(input_shape),
                                   requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape),
                                      requires_grad=False)
        self.__debiasing_term = nn.Parameter(torch.zeros(1),
                                             requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError

    def __check(self, x):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}'
        )

    @torch.no_grad()
    def update(self, x):
        self.__check(x)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))

        batch_mean = x.mean(dim=norm_dims)
        batch_sq_mean = x.square().mean(dim=norm_dims)

        self.__mean.mul_(self.__beta).add_(batch_mean * (1.0 - self.__beta))
        self.__mean_sq.mul_(self.__beta).add_(batch_sq_mean *
                                              (1.0 - self.__beta))
        self.__debiasing_term.mul_(self.__beta).add_(1.0 * (1.0 - self.__beta))

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(
            min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(
            min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return (x - mean) / std

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return x * std + mean


class PopArtValueHead(nn.Module):

    def __init__(
        self,
        input_dim,
        critic_dim,
        beta=0.999,
        epsilon=1e-5,
        burn_in_updates=torch.inf,
        init_gain=0.01,
    ):
        super().__init__()
        self.__rms = RunningMeanStd((critic_dim, ), beta, epsilon)

        self.__weight = nn.Parameter(torch.zeros(critic_dim, input_dim))
        nn.init.orthogonal_(self.__weight, gain=init_gain)
        self.__bias = nn.Parameter(torch.zeros(critic_dim))

        self.__burn_in_updates = burn_in_updates
        self.__update_cnt = 0

    def forward(self, feature):
        return F.linear(feature, self.__weight, self.__bias)

    @torch.no_grad()
    def update(self, x):
        old_mean, old_std = self.__rms.mean_std()
        self.__rms.update(x)
        new_mean, new_std = self.__rms.mean_std()
        self.__update_cnt += 1

        if self.__update_cnt > self.__burn_in_updates:
            self.__weight.data[:] = self.__weight * (old_std /
                                                     new_std).unsqueeze(-1)
            self.__bias.data[:] = (old_std * self.__bias + old_mean -
                                   new_mean) / new_std

    @torch.no_grad()
    def normalize(self, x):
        return self.__rms.normalize(x)

    @torch.no_grad()
    def denormalize(self, x):
        return self.__rms.denormalize(x)