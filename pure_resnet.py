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


class Alignment_Compression_Base(nn.Module):
    def __init__(self,
                 alignment_features: int,
                 init_conv_kernel: int,
                 Block_configs: List[Tuple[str, ResBlock_config]]):
        super(Alignment_Compression_Base, self).__init__()

        self.init_conv = nn.Conv2d(in_channels=alignment_features*2,
                                   out_channels=Block_configs[0][1].channel[0],
                                   kernel_size=init_conv_kernel,
                                   padding=(init_conv_kernel-1)//2)

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
        x = self.to_2d(x)
        x = self.init_conv(x)
        x = self.Resnet2d(x)
        x = self.dropout(x)
        x = self.conv_fin(x)
        x = self.activation(x)

        return x


# settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_epoch = 15
alignment_features = 71


RB2d_configs = [
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True)),
    ("id", ResBlock_config([5, 3], [60, 60, 60], "elu", True))
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
    model = Alignment_Compression_Base(alignment_features,
                                       3,
                                       RB2d_configs)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,2.5]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 2.5]))
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
