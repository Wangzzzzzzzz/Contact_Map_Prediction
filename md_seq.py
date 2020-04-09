import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List, Dict, Tuple
import sys, h5py
import numpy as np

class LSTM_layer(nn.Module):
    def __init__(self, input_size:int,
                       hidden_size:int,
                       dropout:float):
        super(LSTM_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True)
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        y = self.lstm(inputs)
        return y


class BiLSTM(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 dropout:float,
                 layer:int):
        super(BiLSTM,self).__init__()
        self.forward_lstm = nn.ModuleList(
            [LSTM_layer(input_size, hidden_size, 0.0)] + 
            [LSTM_layer(hidden_size, hidden_size, dropout) for _ in range(layer - 1)])
        self.backward_lstm = nn.ModuleList(
            [LSTM_layer(input_size, hidden_size, 0.0)] +
            [LSTM_layer(hidden_size, hidden_size, dropout) for _ in range(layer - 1)]
        )
        assert layer > 0, "layer is too small"

    def forward(self, inputs):
        forward_out = inputs
        for lstm_layer in self.forward_lstm:
            forward_out, _ = lstm_layer(forward_out)

        backward_out = self.reverse(inputs)
        for lstm_layer in self.backward_lstm:
            backward_out, _ = lstm_layer(backward_out)
        backward_out = self.reverse(backward_out)

        return torch.cat((forward_out,backward_out),2)
    
    def reverse(self, sequence):
        ranges = torch.arange(sequence.size(1)-1, -1, -1, device=sequence.device)
        reversed_sequence = sequence.index_select(1, ranges)
        # reversed_sequence.to(sequence.device)
        return reversed_sequence


class ContactMap_feature_RCNN(nn.Module):
    def __init__(self,
                 Bert_output_dim:int,
                 conv1d_hidden:int,
                 lstm_hidden:int,
                 lstm_layer:int):
        super(ContactMap_feature_RCNN, self).__init__()
        self.conv1d = nn.Conv1d(Bert_output_dim,conv1d_hidden,3)
        self.bilstm = BiLSTM(conv1d_hidden,
                             lstm_hidden,
                             0.3,
                             lstm_layer)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs.permute(0,2,1)
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        x = self.activation(x)
        x = self.bilstm(x)
        return self.to_2d(x)


    def to_2d(self, inputs):
        """ Assume that input a 3d tensor with size (batch, seq_len, feature)
            output a 4d tensor with size (batch, feature * 2, seq_len, seq_len) """
        stack = []
        for t in inputs:
            input1 = t.permute(1,0).unsqueeze(2)
            input2 = t.permute(1,0).unsqueeze(1)
            output_2d1 = input1 + input2
            output_2d2 = torch.bmm(input1, input2)
            output = torch.cat((output_2d1,output_2d2),dim=0)
            stack.append(output)
        return torch.stack(tuple(stack),dim=0)

