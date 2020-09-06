import torch
from torch import nn
from typing import List, Dict, Tuple
import sys
import numpy as np
import torch.nn.functional as F

actv_func = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU()
}

# general utils


def determine_activation(activation_name: str):
    global actv_func
    return actv_func[activation_name]


class ResBlock_config:
    def __init__(self,
                 kernel_size: List[int] = [],
                 channel: List[int] = [],
                 activation: str = 'relu',
                 pre_actv: bool = False,
                 stride: int = 1):
        self.kernel_size = kernel_size
        self.channel = channel
        self.activation = activation
        self.pre_actv = pre_actv
        self.stride = stride


# 1d residual blocks
class id_Block1d(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str,
                 pre_actv: bool):
        super(id_Block1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=channel[0],
                              out_channels=channel[1],
                              kernel_size=kernel_size[0],
                              padding=(kernel_size[0]-1)//2)
        self.conv2 = nn.Conv1d(in_channels=channel[1],
                               out_channels=channel[2],
                               kernel_size=kernel_size[1],
                               padding=(kernel_size[1]-1)//2)

        self.normalizer1 = nn.BatchNorm1d(channel[1])
        self.normalizer2 = nn.BatchNorm1d(channel[2])
        self.activation = determine_activation(activation)
        self.pre_actv = pre_actv

    def forward(self, x):
        shortcut = x

        if self.pre_actv:
            x = self.normalizer1(x)
            x = self.activation(x)
            x = self.conv(x)

            x = self.normalizer2(x)
            x = self.activation(x)
            x = self.conv2(x)

            y = x+shortcut

        else:
            x = self.conv(x)
            x = self.normalizer1(x)
            x = self.activation(x)

            x = self.conv2(x)
            x = self.normalizer2(x)
            x = x + shortcut

            y = self.activation(x)

        return y


class conv_Block1d(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str,
                 strides: int = 1):
        super(conv_Block1d, self).__init__()
        self.strides = strides
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels=channel[0],
                              out_channels=channel[1],
                              kernel_size=kernel_size[0],
                              padding=(kernel_size[0]-1)//2)
        self.conv2 = nn.Conv1d(in_channels=channel[1],
                               out_channels=channel[2],
                               kernel_size=kernel_size[1],
                               padding=(kernel_size[1]-1)//2)

        if strides > 1:
            self.conv3 = nn.Conv1d(in_channels=channel[2],
                                   out_channels=channel[3],
                                   kernel_size=kernel_size[2],
                                   stride=strides)

            self.sc_conv = nn.Conv1d(in_channels=channel[0],
                                     out_channels=channel[3],
                                     kernel_size=kernel_size[2],
                                     stride=strides)
        else:
            self.conv3 = nn.Conv1d(in_channels=channel[2],
                                   out_channels=channel[3],
                                   kernel_size=kernel_size[2],
                                   stride=strides,
                                   padding=(kernel_size[2]-1)//2)

            self.sc_conv = nn.Conv1d(in_channels=channel[0],
                                     out_channels=channel[3],
                                     kernel_size=kernel_size[2],
                                     stride=strides,
                                     padding=(kernel_size[2]-1)//2)

        self.normalizer1 = nn.BatchNorm1d(channel[1])
        self.normalizer2 = nn.BatchNorm1d(channel[2])
        self.normalizer3 = nn.BatchNorm1d(channel[3])

        self.sc_norm = nn.BatchNorm1d(channel[3])

        self.activation = determine_activation(activation)

    def forward(self, x):
        # find the length of x
        L = x.size(2)
        shortcut = x

        x = self.conv(x)
        x = self.normalizer1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.normalizer2(x)
        x = self.activation(x)

        if self.strides > 1 and L % 2 == 1:
            shortcut = F.pad(shortcut, [0, 1])
            x = F.pad(x, [0, 1])
        shortcut = self.sc_conv(shortcut)
        shortcut = self.sc_norm(shortcut)

        x = self.conv3(x)
        x = self.normalizer3(x)
        x = x+shortcut
        y = self.activation(x)
        return y

# 1d ResNet


class ResNet1d(nn.Module):
    def __init__(self,
                 RB_config: List[Tuple[str, ResBlock_config]]):
        super(ResNet1d, self).__init__()
        block_list = []
        for blk_type, blk_config in RB_config:
            if blk_type == "conv":
                block_list.append(conv_Block1d(blk_config.kernel_size,
                                               blk_config.channel,
                                               blk_config.activation,
                                               blk_config.stride))
            elif blk_type == "id":
                block_list.append(id_Block1d(blk_config.kernel_size,
                                             blk_config.channel,
                                             blk_config.activation,
                                             blk_config.pre_actv))
            elif blk_type == "reg":
                block_list.append(nn.Conv1d(in_channels=blk_config.channel[0],
                                            out_channels=blk_config.channel[-1],
                                            kernel_size=blk_config.kernel_size[0],
                                            padding=int((blk_config.kernel_size[0]-1)/2)))
            elif blk_type == "none":
                pass
            else:
                raise ValueError("Unrecognized block type.\n")

        self.Resblock_list = nn.ModuleList(block_list)

    def forward(self, inputs):
        outputs = inputs
        for block in self.Resblock_list:
            outputs = block(outputs)

        return outputs

# 2d residual blocks


class id_Block2d(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str,
                 pre_actv: bool):
        super(id_Block2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel[0],
                              out_channels=channel[1],
                              kernel_size=kernel_size[0],
                              padding=int((kernel_size[0]-1)/2))
        self.conv2 = nn.Conv2d(in_channels=channel[1],
                               out_channels=channel[2],
                               kernel_size=kernel_size[1],
                               padding=int((kernel_size[1]-1)/2))

        self.normalizer1 = nn.BatchNorm2d(channel[1])
        self.normalizer2 = nn.BatchNorm2d(channel[2])
        self.activation = determine_activation(activation)
        self.pre_actv = pre_actv

    def forward(self, x):
        shortcut = x
        if self.pre_actv:
            x = self.normalizer1(x)
            x = self.activation(x)
            x = self.conv(x)

            x = self.normalizer2(x)
            x = self.activation(x)
            x = self.conv2(x)

            y = x+shortcut

        else:
            x = self.conv(x)
            x = self.normalizer1(x)
            x = self.activation(x)

            x = self.conv2(x)
            x = self.normalizer2(x)
            x = x + shortcut

            y = self.activation(x)

        return y


class conv_Block2d(nn.Module):
    def __init__(self,
                 kernel_size: List[int],
                 channel: List[int],
                 activation: str):
        super(conv_Block2d, self).__init__()
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


# 2d ResNet


class ResNet2d(nn.Module):
    def __init__(self,
                 RB_config: List[Tuple[str, ResBlock_config]]):
        super(ResNet2d, self).__init__()
        block_list = []
        for blk_type, blk_config in RB_config:
            if blk_type == "conv":
                block_list.append(conv_Block2d(blk_config.kernel_size,
                                               blk_config.channel,
                                               blk_config.activation))
            elif blk_type == "id":
                block_list.append(id_Block2d(blk_config.kernel_size,
                                             blk_config.channel,
                                             blk_config.activation,
                                             blk_config.pre_actv))
            elif blk_type == "reg":
                block_list.append(nn.Conv2d(in_channels=blk_config.channel[0],
                                            out_channels=blk_config.channel[-1],
                                            kernel_size=blk_config.kernel_size[0],
                                            padding=int((blk_config.kernel_size[0]-1)/2)))
            elif blk_type == "none":
                pass
            else:
                raise ValueError("Unrecognized block type.\n")

        self.Resblock_list = nn.ModuleList(block_list)

    def forward(self, inputs):
        outputs = inputs
        for block in self.Resblock_list:
            outputs = block(outputs)

        return outputs


# U-Net
class DoubleConvblk2d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 activation: str = "relu"):
        super(DoubleConvblk2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=kernels[0],
                               padding=int((kernels[0]-1)/2))
        self.conv2 = nn.Conv2d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=kernels[1],
                               padding=int((kernels[1]-1)/2))

        self.norm1 = nn.BatchNorm2d(channels[1])

        self.norm2 = nn.BatchNorm2d(channels[2])
        self.activ = determine_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x


class DownConv2d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 activation: str = "relu",
                 pool_kernel: int = 2):
        super(DownConv2d, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvblk2d(channels,
                                    kernels,
                                    activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x


class UpConv2d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 activation: str = "relu"):
        super(UpConv2d, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=channels[0],
                                         out_channels=channels[0]//2,
                                         kernel_size=2,
                                         stride=2)
        self.conv = DoubleConvblk2d([channels[0], channels[1], channels[2]],
                                    kernels,
                                    activation)

    def forward(self, x, shortcut):
        """ must align with shortcut """
        x = self.upconv(x)

        # pad the intput x
        x_pad = shortcut.size(2) - x.size(2)
        y_pad = shortcut.size(3) - x.size(3)

        x = torch.nn.functional.pad(x, [x_pad//2, x_pad - x_pad//2,
                                        y_pad//2, y_pad - y_pad//2])

        x = torch.cat((x, shortcut), dim=1)
        x = self.conv(x)

        return x


class DoubleConvblk1d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 activation: str = "relu"):
        super(DoubleConvblk1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=kernels[0],
                               padding=int((kernels[0]-1)/2))
        self.conv2 = nn.Conv1d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=kernels[1],
                               padding=int((kernels[1]-1)/2))

        self.norm1 = nn.BatchNorm1d(channels[1])

        self.norm2 = nn.BatchNorm1d(channels[2])
        self.activ = determine_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x


class DownConv1d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 activation: str = "relu",
                 pool_kernel: int = 2):
        super(DownConv1d, self).__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConvblk1d(channels,
                                    kernels,
                                    activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x


class UpConv1d(nn.Module):
    def __init__(self,
                 channels: List[int],
                 kernels: List[int],
                 shortcut_channels: int,
                 activation: str = "relu"):
        super(UpConv1d, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels=channels[0],
                                         out_channels=channels[0]//2,
                                         kernel_size=2,
                                         stride=2)
        #print(channels[0]//2 + shortcut_channels)
        self.conv = DoubleConvblk1d([channels[0]//2 + shortcut_channels,
                                     channels[1],
                                     channels[2]],
                                    kernels,
                                    activation)

    def forward(self, x, shortcut):
        """ must align with shortcut """
        x = self.upconv(x)

        # pad the intput x
        x_pad = shortcut.size(2) - x.size(2)

        x = torch.nn.functional.pad(x, [x_pad//2, x_pad - x_pad//2])

        x = torch.cat((x, shortcut), dim=1)
        x = self.conv(x)

        return x
