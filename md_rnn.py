import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List, Dict, Tuple
import sys
import h5py
import numpy as np
from md_cnn import ResNet1d, ResBlock_config


class LSTM_layer(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 dropout: float):
        super(LSTM_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        y = self.lstm(inputs)
        return y


class BiLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float,
                 layer: int,
                 shortcut: bool):
        super(BiLSTM, self).__init__()
        self.forward_lstm = nn.ModuleList(
            [LSTM_layer(input_size, hidden_size, 0.0)] +
            [LSTM_layer(hidden_size, hidden_size, dropout) for _ in range(layer - 1)])
        self.backward_lstm = nn.ModuleList(
            [LSTM_layer(input_size, hidden_size, 0.0)] +
            [LSTM_layer(hidden_size, hidden_size, dropout)
             for _ in range(layer - 1)]
        )
        self.shortcut = shortcut
        assert layer > 0, "layer is too small"

    def forward(self, inputs):
        forward_out = inputs
        for i, lstm_layer in enumerate(self.forward_lstm):
            if self.shortcut and i > 0:
                shortcut_out = forward_out
                forward_out, _ = lstm_layer(forward_out)
                forward_out = forward_out + shortcut_out
            else:
                forward_out, _ = lstm_layer(forward_out)

        backward_out = self.reverse(inputs)
        for i, lstm_layer in enumerate(self.backward_lstm):
            if self.shortcut and i > 0:
                shortcut_out = backward_out
                backward_out, _ = lstm_layer(backward_out)
                backward_out = backward_out + shortcut_out
            else:
                backward_out, _ = lstm_layer(backward_out)
        backward_out = self.reverse(backward_out)

        return torch.cat((forward_out, backward_out), 2)

    def reverse(self, sequence):
        ranges = torch.arange(sequence.size(1)-1, -1, -
                              1, device=sequence.device)
        reversed_sequence = sequence.index_select(1, ranges)
        # reversed_sequence.to(sequence.device)
        return reversed_sequence


class RCNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 conv1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 using_shortcut=True,
                 using_bert=True):
        super(RCNN, self).__init__()
        self.with_bert = using_bert
        self.resnet1d = ResNet1d(conv1d_config)
        self.bilstm = BiLSTM(conv1d_config[-1][1].channel[3],
                             lstm_hidden,
                             0.3,
                             lstm_layer,
                             using_shortcut)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        if self.with_bert:
            x = inputs[:, 1:-1, :]
            x = x.permute(0, 2, 1)
        else:
            x = inputs
        x = self.resnet1d(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        x = self.bilstm(x)
        return self.to_2d(x)

    def to_2d(self, inputs):
        """ Assume that input a 3d tensor with size (batch, seq_len, feature)
            output a 4d tensor with size (batch, feature * 2, seq_len, seq_len) """
        stack = []
        for t in inputs:
            # concat
            # length = t.size(0)
            # broadcaster1 = torch.arange(length).view(1, 1, length)
            # broadcaster2 = torch.arange(length).view(1, length, 1)
            # input1 = t.permute(1, 0).unsqueeze(2)
            # input2 = t.permute(1, 0).unsqueeze(1)
            # output_2d1, _ = torch.broadcast_tensors(input1, broadcaster1)
            # output_2d2, _ = torch.broadcast_tensors(input2, broadcaster2)

            # sum & product
            input1 = t.permute(1, 0).unsqueeze(2)
            input2 = t.permute(1, 0).unsqueeze(1)
            output_2d1 = input1 + input2
            output_2d2 = torch.bmm(input1, input2)

            output = torch.cat((output_2d1, output_2d2), dim=0)
            stack.append(output)

        return torch.stack(tuple(stack), dim=0)
