import md_cnn
import md_rnn
from md_cnn import ResNet2d, ResNet1d, ResBlock_config
from md_rnn import BiResIndRNN
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


class Alignment_Compression_indrnn(nn.Module):
    def __init__(self,
                 compresser1_config: ResBlock_config,
                 compresser2_config: ResBlock_config,
                 rnn_input_size: int,
                 rnn_hidden: int,
                 rnn_blocks: int,
                 Block_configs: List[Tuple[str, ResBlock_config]]):
        super(Alignment_Compression_indrnn, self).__init__()

        self.Compresser1 = md_cnn.conv_Block1d(compresser1_config.kernel_size,
                                               compresser1_config.channel,
                                               compresser1_config.activation,
                                               compresser1_config.stride)

        self.Compresser2 = md_cnn.conv_Block1d(compresser2_config.kernel_size,
                                               compresser2_config.channel,
                                               compresser2_config.activation,
                                               compresser2_config.stride)

        self.compresser2lstm_conv = nn.Conv1d(in_channels=compresser2_config.channel[-1],
                                              out_channels=rnn_input_size,
                                              kernel_size=2)

        self.RNN = BiResIndRNN(input_size=rnn_input_size,
                               hidden_size=rnn_hidden,
                               n_blks=rnn_blocks,
                               dropout=0.5)

        rnn_out = rnn_hidden*2
        self.Upconv1 = md_cnn.UpConv1d(channels=[rnn_out,
                                                 rnn_out//2,
                                                 rnn_out//2],
                                       kernels=[5, 3],
                                       shortcut_channels=compresser2_config.channel[0])
        self.Upconv2 = md_cnn.UpConv1d(channels=[rnn_out//2,
                                                 rnn_out//4,
                                                 rnn_out//4],
                                       kernels=[5, 3],
                                       shortcut_channels=compresser1_config.channel[0])

        self.Resnet2d = ResNet2d(Block_configs)
        self.dropout = nn.Dropout(0.3)
        self.conv_fin = nn.Conv2d(in_channels=Block_configs[-1][1].channel[-1],
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

    def forward(self, x):
        shortcut1 = x
        x = self.Compresser1(x)
        shortcut2 = x
        x = self.Compresser2(x)
        x = self.compresser2lstm_conv(x)
        x = x.permute(2, 0, 1)
        x = self.RNN(x)
        x = x.permute(1, 2, 0)

        # print(x.size())
        x = self.Upconv1(x, shortcut2)
        x = self.Upconv2(x, shortcut1)

        x = self.to_2d(x)

        x = self.Resnet2d(x)
        x = self.dropout(x)
        x = self.conv_fin(x)
        x = self.activation(x)

        return x


# settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_epoch = 15
alignment_features = 71
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
rnn_input_size = alignment_features*4
rnn_hidden = 1024
rnn_block = 4

RB2d_configs = [
    ("conv", ResBlock_config([1, 1, 1], [
     rnn_hidden, rnn_hidden//3, rnn_hidden//9, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("id", ResBlock_config([3, 5], [64, 64, 64], "relu")),
    ("conv", ResBlock_config([3, 5, 3], [64, 32, 32, 16], "relu")),
    ("id", ResBlock_config([3, 5], [16, 8, 16], "relu")),
    ("id", ResBlock_config([3, 5], [16, 8, 16], "relu")),
    ("id", ResBlock_config([3, 5], [16, 8, 16], "relu"))
]


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
    model = Alignment_Compression_indrnn(compresser1_config,
                                         compresser2_config,
                                         rnn_input_size,
                                         rnn_hidden,
                                         rnn_block,
                                         RB2d_configs)
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,2.5]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.5]))
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
