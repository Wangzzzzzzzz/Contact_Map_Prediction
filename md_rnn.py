import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List, Dict, Tuple
import sys
import h5py
import numpy as np
from md_cnn import ResNet1d, ResBlock_config
from IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN


class LSTM_layer(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 dropout: float):
        super(LSTM_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs, h_0=None, c_0=None):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        if h_0 is None or c_0 is None:
            y = self.lstm(inputs)
        else:
            y = self.lstm(inputs, (h_0, c_0))
        return y


class BiLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float,
                 layer: int,
                 shortcut: bool,
                 carryon_info: bool):
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
        self.hidden_size = hidden_size
        self.carryon_sig = carryon_info
        assert layer > 0, "layer is too small"

    def forward(self, inputs):
        forward_out = inputs
        forward_h_x = torch.zeros(1, inputs.size(
            0), self.hidden_size, device=inputs.device)
        forward_c_x = torch.zeros(1, inputs.size(
            0), self.hidden_size, device=inputs.device)
        for i, lstm_layer in enumerate(self.forward_lstm):
            if self.shortcut and i > 0:
                shortcut_out = forward_out
                # carry the signal from previous layer to the next layer
                # to increase the global information propagation
                if self.carryon_sig:
                    forward_out, (forward_h_x, forward_c_x) = lstm_layer(
                        forward_out, (forward_h_x, forward_c_x))
                # regular architecture
                else:
                    forward_out, _ = lstm_layer(forward_out)
                forward_out = forward_out + shortcut_out
            else:
                if self.carryon_sig:
                    forward_out, (forward_h_x, forward_c_x) = lstm_layer(
                        forward_out, (forward_h_x, forward_c_x))
                else:
                    forward_out, _ = lstm_layer(forward_out)

        backward_out = self.reverse(inputs)
        backward_h_x = torch.zeros(1, inputs.size(0), self.hidden_size)
        backward_c_x = torch.zeros(1, inputs.size(0), self.hidden_size)
        for i, lstm_layer in enumerate(self.backward_lstm):
            if self.shortcut and i > 0:
                shortcut_out = backward_out
                if self.carryon_sig:
                    backward_out, (backward_h_x, backward_c_x) = lstm_layer(
                        backward_out, (backward_h_x, backward_c_x))
                # regular architecture
                else:
                    backward_out, _ = lstm_layer(backward_out)
                backward_out = backward_out + shortcut_out
            else:
                if self.carryon_sig:
                    backward_out, (backward_h_x, backward_c_x) = lstm_layer(
                        backward_out, (backward_h_x, backward_c_x))
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
                 using_bert=True,
                 using_shortcut=True,
                 using_carryon_info=False):
        super(RCNN, self).__init__()
        self.with_bert = using_bert
        self.resnet1d = ResNet1d(conv1d_config)
        self.bilstm = BiLSTM(conv1d_config[-1][1].channel[-1],
                             lstm_hidden,
                             0.3,
                             lstm_layer,
                             using_shortcut,
                             using_carryon_info)
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

# IndRNN part


class Recurrent_Linear(nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        super(Recurrent_Linear, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=True)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs):
        L = inputs.size(0)
        N = inputs.size(1)
        x = self.fc(inputs.contiguous().view(-1, self.input_size))
        return x.view(L, N, self.output_size)


class Recurrent_BN(nn.Module):
    def __init__(self,
                 num_features):
        super(Recurrent_BN, self).__init__()
        self.BN = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        """ expected (L, N, C) shaped input,
            output (L, N, C) shaped output """
        outputs = inputs.permute(1, 2, 0)
        outputs = self.BN(outputs)
        outputs = outputs.permute(2, 0, 1)

        return outputs


class Res_IndRNN_Block(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 dropout: float,
                 linear_fist: bool):
        super(Res_IndRNN_Block, self).__init__()
        self.linear_first = linear_fist

        self.linear1 = Recurrent_Linear(hidden_size, hidden_size)
        self.bn1 = Recurrent_BN(hidden_size)
        self.indrnn1 = IndRNN(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = Recurrent_Linear(hidden_size, hidden_size)
        self.bn2 = Recurrent_BN(hidden_size)
        self.indrnn2 = IndRNN(hidden_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        if self.linear_first:
            y = self.linear1(y)
            y = self.bn1(y)
            y = self.indrnn1(y)
            y = self.drop1(y)
            y = self.linear2(y)
            y = self.bn2(y)
            y = self.indrnn2(y)
            y = self.drop2(y)
        else:
            y = self.bn1(y)
            y = self.indrnn1(y)
            y = self.drop1(y)
            y = self.linear1(y)
            y = self.bn2(y)
            y = self.indrnn2(y)
            y = self.drop2(y)
            y = self.linear2(y)
        # create short cut
        y = y + x
        return y


class Res_IndRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_blks: int,
                 dropout: float,
                 linear_first: bool,
                 indrnn_end: bool):
        """ linear_first -> run linear before pass into indrnn,
            indrnn_end -> if true, add additional layer of indrnn to end module """
        super(Res_IndRNN, self).__init__()
        self.init_fc = Recurrent_Linear(input_size, hidden_size)
        Block_list = []
        for i in range(n_blks):
            Block_list.append(Res_IndRNN_Block(hidden_size,
                                               dropout,
                                               linear_first))
        self.res_indrnn = nn.ModuleList(Block_list)
        self.indrnn_end = indrnn_end
        if indrnn_end:
            self.bn_fin = Recurrent_BN(hidden_size)
            self.indrnn_fin = IndRNN(hidden_size)
            self.drop_fin = nn.Dropout(dropout)

    def forward(self, x):
        """ inputs shape: (L,N,C),
            output shape: (L,N,C) """
        x = self.init_fc(x)
        for block in self.res_indrnn:
            x = block(x)
        if self.indrnn_end:
            x = self.bn_fin(x)
            x = self.indrnn_fin(x)
            x = self.drop_fin(x)

        return x


class BiResIndRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_blks: int,
                 dropout: float,
                 linear_first: bool = False,
                 indrnn_end: bool = False):
        super(BiResIndRNN, self).__init__()
        self.forward_net = Res_IndRNN(input_size,
                                      hidden_size,
                                      n_blks,
                                      dropout,
                                      linear_first,
                                      indrnn_end)
        self.backward_net = Res_IndRNN(input_size,
                                       hidden_size,
                                       n_blks,
                                       dropout,
                                       linear_first,
                                       indrnn_end)

    def reverse(self, sequence):
        ranges = torch.arange(sequence.size(0)-1, -1, -1,
                              device=sequence.device)
        reversed_sequence = sequence.index_select(0, ranges)
        # reversed_sequence.to(sequence.device)
        return reversed_sequence

    def forward(self, x):
        """ expect input of shape (L,N,C) """
        forward_res = self.forward_net(x)

        x_reverse = self.reverse(x)
        reverse_res = self.backward_net(x_reverse)
        reverse_res = self.reverse(reverse_res)

        return torch.cat((forward_res, reverse_res), dim=2)


def clip_IndRNN_weight(model, val):
    for n, p in model.named_parameters():
        if 'weight_hh' in n:
            p.data.clamp_(-val, val)


def clip_IndRNN_grad(model, val):
    for p in model.parameters():
        p.grad.data.clamp_(-val, val)
