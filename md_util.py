import torch
from torch import nn
from typing import List, Dict, Tuple
import sys
import numpy as np


actv_func = {
    "relu":nn.ReLU(),
    "sigmoid":nn.Sigmoid(),
    "tanh":nn.Tanh
}

def determine_activation(activation_name: str):
    global actv_func
    return actv_func[activation_name]


class ResBlock_config:
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str):
        self.kernel_size = kernel_size
        self.channel = channel
        self.activation = activation


class id_Block(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str):
        super(id_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel[0],
                              out_channels=channel[1],
                              kernel_size=kernel_size[0],
                              padding=int((kernel_size[0]-1)/2))
        self.conv2 = nn.Conv2d(in_channels=channel[1],
                               out_channels=channel[2],
                               kernel_size=kernel_size[1],
                               padding=int((kernel_size[1]-1)/2))
        self.conv3 = nn.Conv2d(in_channels=channel[2],
                               out_channels=channel[3],
                               kernel_size=kernel_size[2],
                               padding=int((kernel_size[2]-1)/2))

        self.normalizer1 = nn.BatchNorm2d(channel[1])
        self.normalizer2 = nn.BatchNorm2d(channel[2])
        self.normalizer3 = nn.BatchNorm2d(channel[3])
        self.activation = determine_activation(activation)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.normalizer1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.normalizer2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.normalizer3(x)
        x = x+shortcut
        y = self.activation(x)
        return y

class conv_Block(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str):
        super(conv_Block,self).__init__()
        self.conv = nn.Conv2d(in_channels=channel[0],
                              out_channels=channel[1],
                              kernel_size=kernel_size[0],
                              padding=int((kernel_size[0]-1)/2))
        self.conv2 = nn.Conv2d(in_channels=channel[1],
                               out_channels=channel[2],
                               kernel_size=kernel_size[1],
                               padding=int((kernel_size[1]-1)/2))
        self.conv3 = nn.Conv2d(in_channels=channel[2],
                               out_channels=channel[3],
                               kernel_size=kernel_size[2],
                               padding=int((kernel_size[2]-1)/2))

        self.normalizer1 = nn.BatchNorm2d(channel[1])
        self.normalizer2 = nn.BatchNorm2d(channel[2])
        self.normalizer3 = nn.BatchNorm2d(channel[3])

        self.sc_conv = nn.Conv2d(in_channels=channel[0],
                                 out_channels=channel[3],
                                 kernel_size=1)
        self.sc_norm = nn.BatchNorm2d(channel[3])
        
        self.activation = determine_activation(activation)

    def forward(self, x):
        shortcut = x
        shortcut = self.sc_conv(shortcut)
        shortcut = self.sc_norm(shortcut)
        x = self.conv(x)
        x = self.normalizer1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.normalizer2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.normalizer3(x)
        x = x+shortcut
        y = self.activation(x)
        return y


class Gathering_ResNet(nn.Module):
    def __init__(self,
                 RB_config: List[Tuple[str, ResBlock_config]]):
        super(Gathering_ResNet, self).__init__()
        block_list = []
        for blk_type, blk_config in RB_config:
            if blk_type == "conv":
                block_list.append(conv_Block(blk_config.kernel_size,
                                             blk_config.channel,
                                             blk_config.activation))
            elif blk_type == "id":
                block_list.append(id_Block(blk_config.kernel_size,
                                           blk_config.channel,
                                           blk_config.activation))
            else:
                raise ValueError("Unrecognized block type.\n")

        self.Resblock_list = nn.ModuleList(block_list)
    
    def forward(self, inputs):
        outputs = inputs
        for block in self.Resblock_list:
            outputs = block(outputs)
        
        return outputs
