import md_cnn
import md_rnn
from md_cnn import ResNet2d, ResNet1d, ResBlock_config
from md_rnn import RCNN, BiLSTM
from md_train import train_an_epoch, eval_an_epoch, load_dataset
from md_train import ContactMap_Set
from md_train import collate_fn_str, collate_fn_ali, collate_fn_mix
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import sys
import threading
from tape import TAPETokenizer, ProteinBertModel
from md_loss import CTmap_CrossEntropyLoss, PosWeighted_CELoss, multiWeighted_CELoss


class Pipe2d_Compression_Base(nn.Module):
    def __init__(self,
                 Block_configs_precompr: List[Tuple[str, ResBlock_config]],
                 Block_configs_1: List[Tuple[str, ResBlock_config]],
                 Block_configs_2: List[Tuple[str, ResBlock_config]]):
        super(Pipe2d_Compression_Base, self).__init__()

        pre_compress_out_channel = Block_configs_precompr[-1][1].channel[-1]
        first_compress_out_channel = Block_configs_1[-1][1].channel[-1]
        second_compress_out_channel = Block_configs_2[-1][1].channel[-1]

        # compression phase
        self.Pipe2d_pre_comp_resnet = ResNet2d(Block_configs_precompr)
        self.Pipe2d_comp_1 = md_cnn.conv_Block2d(kernel_size=[1, 3, 2],
                                                 channel=[pre_compress_out_channel,
                                                          pre_compress_out_channel,
                                                          pre_compress_out_channel,
                                                          pre_compress_out_channel*2],
                                                 activation="relu",
                                                 strides=2)
        self.Pipe2d_pos_comp_resnet_1 = ResNet2d(Block_configs_1)
        self.Pipe2d_comp_2 = md_cnn.conv_Block2d(kernel_size=[1, 3, 2],
                                                 channel=[first_compress_out_channel,
                                                          first_compress_out_channel,
                                                          first_compress_out_channel,
                                                          first_compress_out_channel*2],
                                                 activation="relu",
                                                 strides=2)
        self.Pipe2d_pos_comp_conv = nn.Conv2d(first_compress_out_channel*2,
                                              first_compress_out_channel*2,
                                              2)

        # a resnet in the middle
        self.Pipe2d_pos_comp_resnet_2 = ResNet2d(Block_configs_2)

        # decompression phase
        self.Pipe2d_decomp_1 = md_cnn.UpConv2d(channels=[second_compress_out_channel,
                                                         second_compress_out_channel//2,
                                                         second_compress_out_channel//2],
                                               kernels=[5, 3])
        self.Pipe2d_pos_decomp_resnet_1 = ResNet2d(Block_configs_1)

        self.Pipe2d_decomp_2 = md_cnn.UpConv2d(channels=[first_compress_out_channel,
                                                         first_compress_out_channel//2,
                                                         first_compress_out_channel//2],
                                               kernels=[5, 3])
        self.Pipe2d_pos_decomp_resnet_2 = ResNet2d(Block_configs_precompr)

    def forward(self, x):
        x = self.Pipe2d_pre_comp_resnet(x)
        shortcut1 = x
        x = self.Pipe2d_comp_1(x)
        x = self.Pipe2d_pos_comp_resnet_1(x)
        shortcut2 = x
        x = self.Pipe2d_comp_2(x)
        x = self.Pipe2d_pos_comp_conv(x)
        x = self.Pipe2d_pos_comp_resnet_2(x)
        x = self.Pipe2d_decomp_1(x, shortcut2)
        x = self.Pipe2d_pos_decomp_resnet_1(x)
        x = self.Pipe2d_decomp_2(x, shortcut1)
        x = self.Pipe2d_pos_decomp_resnet_2(x)

        return x


class Alignment_Compression_Base(nn.Module):
    def __init__(self,
                 compresser1_config: ResBlock_config,
                 compresser2_config: ResBlock_config,
                 lstm_input_size: int,
                 lstm_hidden: int,
                 lstm_layers: int,
                 feat_2d_dim: int,
                 Block_configs_precompr: List[Tuple[str, ResBlock_config]],
                 Block_configs_1: List[Tuple[str, ResBlock_config]],
                 Block_configs_2: List[Tuple[str, ResBlock_config]]):
        super(Alignment_Compression_Base, self).__init__()

        self.Compresser1 = md_cnn.conv_Block1d(compresser1_config.kernel_size,
                                               compresser1_config.channel,
                                               compresser1_config.activation,
                                               compresser1_config.stride)

        self.Compresser2 = md_cnn.conv_Block1d(compresser2_config.kernel_size,
                                               compresser2_config.channel,
                                               compresser2_config.activation,
                                               compresser2_config.stride)

        self.compresser2lstm_conv = nn.Conv1d(in_channels=compresser2_config.channel[-1],
                                              out_channels=lstm_input_size,
                                              kernel_size=2)

        self.RNN = BiLSTM(input_size=lstm_input_size,
                          hidden_size=lstm_hidden,
                          dropout=0.0,
                          layer=lstm_layers,
                          shortcut=True,
                          carryon_info=False)

        lstm_out = lstm_hidden*2
        self.Upconv1 = md_cnn.UpConv1d(channels=[lstm_out,
                                                 lstm_out//2,
                                                 lstm_out//2],
                                       kernels=[5, 3],
                                       shortcut_channels=compresser2_config.channel[0])
        self.Upconv2 = md_cnn.UpConv1d(channels=[lstm_out//2,
                                                 lstm_out//4,
                                                 lstm_out//4],
                                       kernels=[5, 3],
                                       shortcut_channels=compresser1_config.channel[0])

        self.Pipe2d_lstm_dim_converter_self = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.Linear(lstm_hidden//2, lstm_hidden//4),
            nn.Linear(lstm_hidden//4, 64),
            nn.ReLU(inplace=True)
        )
        self.Pipe2d_lstm_dim_converter_feat2d = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.Linear(lstm_hidden//2, lstm_hidden//4),
            nn.Linear(lstm_hidden//4, 64),
            nn.ReLU(inplace=True)
        )
        self.Pipe2d_feat2d_dim_converter_self = nn.Sequential(
            nn.Linear(feat_2d_dim, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )
        self.Pipe2d_feat2d_dim_converter_lstm = nn.Sequential(
            nn.Linear(feat_2d_dim, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )

        self.Pipe2d_lstm = Pipe2d_Compression_Base(Block_configs_precompr,
                                                   Block_configs_1,
                                                   Block_configs_2)
        self.Pipe2d_feat2d = Pipe2d_Compression_Base(Block_configs_precompr,
                                                     Block_configs_1,
                                                     Block_configs_2)
        self.dropout = nn.Dropout(0.3)
        self.conv_fin = nn.Conv2d(in_channels=Block_configs_precompr[-1][1].channel[-1],
                                  out_channels=2,
                                  kernel_size=1)
        self.activation = nn.Softmax(1)

    def to_2d(self, inputs):
        """ Assume that input a 3d tensor with size (batch, feature, seq_len)
            output a 4d tensor with size (batch, feature * 2, seq_len, seq_len) """
        stack = []
        for t in inputs:
            # concat
            length = t.size(1)
            broadcaster1 = torch.arange(length).view(1, 1, length)
            broadcaster2 = torch.arange(length).view(1, length, 1)
            input1 = t.unsqueeze(2)
            input2 = t.unsqueeze(1)
            output_2d1, _ = torch.broadcast_tensors(input1, broadcaster1)
            output_2d2, _ = torch.broadcast_tensors(input2, broadcaster2)

            # sum & product
            # input1 = t.unsqueeze(2)
            # input2 = t.unsqueeze(1)
            # output_2d1 = input1 + input2
            # output_2d2 = torch.bmm(input1, input2)

            output = torch.cat((output_2d1, output_2d2), dim=0)
            stack.append(output)

        return torch.stack(tuple(stack), dim=0)

    def forward(self, x_1d, x_2d):
        shortcut1 = x_1d
        x_1d = self.Compresser1(x_1d)
        shortcut2 = x_1d
        x_1d = self.Compresser2(x_1d)
        x_1d = self.compresser2lstm_conv(x_1d)
        x_1d = x_1d.permute(0, 2, 1)
        x_1d = self.RNN(x_1d)
        x_1d = x_1d.permute(0, 2, 1)

        # print(x.size())
        x_1d = self.Upconv1(x_1d, shortcut2)
        x_1d = self.Upconv2(x_1d, shortcut1)

        x_1d = self.to_2d(x_1d)
        x_1d = x_1d.permute(0, 2, 3, 1)
        # print(x_1d.shape)
        x1d_to_1d = self.Pipe2d_lstm_dim_converter_self(x_1d)
        x1d_to_2d = self.Pipe2d_lstm_dim_converter_feat2d(x_1d)
        x2d_to_2d = self.Pipe2d_feat2d_dim_converter_self(x_2d)
        x2d_to_1d = self.Pipe2d_feat2d_dim_converter_lstm(x_2d)
        # x = torch.cat((x, x_2d), 1)

        x_1d = (x1d_to_1d + x2d_to_1d).permute(0, 3, 1, 2)
        x_2d = (x2d_to_2d + x1d_to_2d).permute(0, 3, 1, 2)
        x_2d = self.Pipe2d_feat2d(x_2d)
        x_1d = self.Pipe2d_lstm(x_1d)
        x = x_2d + x_1d
        x = self.dropout(x)
        x = self.conv_fin(x)
        x = self.activation(x)

        return x


# settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_epoch = 15
alignment_features = 71
feat2d_dim = 3
compresser1_config = ResBlock_config(kernel_size=[1, 3, 2],
                                     channel=[alignment_features,
                                              alignment_features,
                                              alignment_features,
                                              alignment_features*2],
                                     stride=2)
compresser2_config = ResBlock_config(kernel_size=[1, 3, 2],
                                     channel=[alignment_features*2,
                                              alignment_features*2,
                                              alignment_features*2,
                                              alignment_features*4],
                                     stride=2)
lstm_input_size = alignment_features*4
lstm_hidden = alignment_features*4
lstm_layer = 4

RB2d_configs0 = [
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu"))
]

RB2d_configs1 = [
    ("id", ResBlock_config([3, 5], [128, 128, 128], "relu")),
    ("id", ResBlock_config([3, 5], [128, 128, 128], "relu")),
    ("id", ResBlock_config([3, 5], [128, 128, 128], "relu")),
    ("id", ResBlock_config([3, 5], [128, 128, 128], "relu"))
]

RB2d_configs2 = [
    ("id", ResBlock_config([3, 5], [256, 256, 256], "relu")),
    ("id", ResBlock_config([3, 5], [256, 256, 256], "relu")),
    ("id", ResBlock_config([3, 5], [256, 256, 256], "relu")),
    ("id", ResBlock_config([3, 5], [256, 256, 256], "relu"))
]

# RB2d_configs = [
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
#     ("conv", ResBlock_config([3, 5, 3], [64, 32, 32, 16], "relu")),
#     ("id", ResBlock_config([3, 5], [16, 8, 16], "relu")),
#     ("id", ResBlock_config([3, 5], [16, 8, 16], "relu")),
#     ("id", ResBlock_config([3, 5], [16, 8, 16], "relu"))
# ]


def train_compression_model():
    # set up datasets
    train_x, train_y = load_dataset(
        '/home/zheng/diskf/data/dataset/alignment_dataset_train.pkl')
    eval_x, eval_y = load_dataset(
        '/home/zheng/diskf/data/dataset/alignment_dataset_val.pkl')
    train_size = len(train_x)
    eval_size = len(eval_x)
    TrainSet = ContactMap_Set(train_x, train_y)
    EvalSet = ContactMap_Set(eval_x, eval_y)

    train_loader = DataLoader(TrainSet, 1, True, collate_fn=collate_fn_ali)
    eval_loader = DataLoader(EvalSet, 1, True, collate_fn=collate_fn_ali)

    # set up models
    model = Alignment_Compression_Base(compresser1_config,
                                       compresser2_config,
                                       lstm_input_size,
                                       lstm_hidden,
                                       lstm_layer,
                                       feat2d_dim,
                                       RB2d_configs0,
                                       RB2d_configs1,
                                       RB2d_configs2)
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,2.5]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.65]))
    # loss_fn = PosWeighted_CELoss(torch.tensor([0.25, 1.5]),
    #                              {'long': 3.0, 'medium': 1.5, 'short': 1.0})

    # send model to gpu
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(N_epoch):
        train_an_epoch(epoch,
                       train_loader,
                       model,
                       optimizer,
                       loss_fn)
        l_test, p_test, r_test, a_test, m_test = eval_an_epoch(epoch,
                                                               eval_loader,
                                                               model,
                                                               loss_fn,
                                                               eval_size)
        print("Epoch: {0:02}".format(epoch+1))
        print("Val. Loss: {0:.3f} | Val. Perc: {1:.3f}".format(l_test, p_test))
        print("Val. Recall: {0:.3f} | Val. Acc: {1:.3f}\n".format(
            r_test, a_test))
        print('Specific Metrics are the following:')
        print("accu_L/10_short: {0:.3f}, accu_L/10_medium: {1:.3f}, accu_L/10_long: {2:.3f}".format(
            m_test[0], m_test[1], m_test[2]))
        print("accu_L/5_short: {0:.3f}, accu_L/5_medium: {1:.3f}, accu_L/5_long: {2:.3f}".format(
            m_test[3], m_test[4], m_test[5]))
        print("accu_L/2_short: {0:.3f}, accu_L/2_medium: {1:.3f}, accu_L/2_long: {2:.3f}".format(
            m_test[6], m_test[7], m_test[8]))
        print("accu_L/1_short: {0:.3f}, accu_L/1_medium: {1:.3f}, accu_L/1_long: {2:.3f}\n".format(
            m_test[9], m_test[10], m_test[11]))

    torch.save(model.state_dict(),
               './compression_base.pth.tar')


def main():
    train_compression_model()


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    t = threading.Thread(target=main)
    t.start()
