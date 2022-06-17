import torch
import torch.nn as nn


class AutoResetRNN(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=1,
                 batch_first=False,
                 rnn_type='gru'):
        super().__init__()
        self.__type = rnn_type
        if self.__type == 'gru':
            self.__net = nn.GRU(input_dim,
                                output_dim,
                                num_layers=num_layers,
                                batch_first=batch_first)
        elif self.__type == 'lstm':
            self.__net = nn.LSTM(input_dim,
                                 output_dim,
                                 num_layers=num_layers,
                                 batch_first=batch_first)
        else:
            raise NotImplementedError(
                f'RNN type {self.__type} has not been implemented.')

    def __forward(self, x, h):
        if self.__type == 'lstm':
            h = torch.split(h, h.shape[-1] // 2, dim=-1)
            h = (h[0].contiguous(), h[1].contiguous())
        x_, h_ = self.__net(x, h)
        if self.__type == 'lstm':
            h_ = torch.cat(h_, -1)
        return x_, h_

    def forward(self, x, h, mask=None):
        if mask is None:
            x_, h_ = self.__forward(x, h)
        else:
            outputs = []
            for t in range(mask.shape[0]):
                x_, h = self.__forward(x[t:t + 1],
                                       (h * mask[t:t + 1]).contiguous())
                outputs.append(x_)
            x_ = torch.cat(outputs, 0)
            h_ = h
        return x_, h_
