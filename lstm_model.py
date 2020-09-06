from md_cnn import ResNet2d, ResNet1d, ResBlock_config
from md_rnn import RCNN
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
from md_loss import CTmap_CrossEntropyLoss, PosWeighted_CELoss


class Language_Base(nn.Module):
    def __init__(self,
                 Bert_output_dim: int,
                 RB1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB2d_config: List[Tuple[str, ResBlock_config]]):
        super(Language_Base, self).__init__()
        self.bert = ProteinBertModel.from_pretrained('bert-base')
        self.rcnn = RCNN(Bert_output_dim,
                         RB1d_config,
                         lstm_hidden,
                         lstm_layer)
        self.resnet = ResNet2d(RB2d_config)
        # force output to have 2 channel
        self.conv2d = nn.Conv2d(RB2d_config[-1][1].channel[-1], 2, 1)

    def forward(self, inputs):
        x = self.bert(inputs)[0]
        x = self.rcnn(x)
        x = self.resnet(x)
        x = self.conv2d(x)

        return x


class Alignment_Base(nn.Module):
    def __init__(self,
                 Alignment_dim: int,
                 RB1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB2d_config: List[Tuple[str, ResBlock_config]]):

        super(Alignment_Base, self).__init__()
        self.rcnn = RCNN(Alignment_dim,
                         RB1d_config,
                         lstm_hidden,
                         lstm_layer,
                         False)
        self.resnet = ResNet2d(RB2d_config)
        self.conv2d = nn.Conv2d(RB2d_config[-1][1].channel[-1], 2, 1)

    def forward(self, inputs):
        x = self.rcnn(inputs)
        x = self.resnet(x)
        x = self.conv2d(x)

        return x


class Language_Contact(nn.Module):
    def __init__(self,
                 Bert_output_dim: int,
                 RB1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB2d_config: List[Tuple[str, ResBlock_config]]):

        super(Language_Contact, self).__init__()
        self.seq_base = Language_Base(
            Bert_output_dim, RB1d_config, lstm_hidden, lstm_layer, RB2d_config)
        self.activation = nn.Softmax(1)

    def forward(self, inputs):
        x = self.seq_base(inputs)
        y = self.activation(x)

        return y


class Alignment_Contact(nn.Module):
    def __init__(self,
                 Alignment_dim: int,
                 RB1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB2d_config: List[Tuple[str, ResBlock_config]]):

        super(Alignment_Contact, self).__init__()
        self.ali_base = Alignment_Base(Alignment_dim,
                                       RB1d_config,
                                       lstm_hidden,
                                       lstm_layer,
                                       RB2d_config)
        self.activation = nn.Softmax(1)

    def forward(self, inputs):
        x = self.ali_base(inputs)
        y = self.activation(x)

        return y


class Lang_OnTopOf_Alignment(nn.Module):
    def __init__(self,
                 Bert_output_dim: int,
                 RB1d_config: List[Tuple[str, ResBlock_config]],
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB2d_config: List[Tuple[str, ResBlock_config]],
                 Alignment_model: Alignment_Base,
                 Alignment_model_weight: str,
                 req_grad_Alignment: bool = True):
        super(Lang_OnTopOf_Alignment, self).__init__()

        self.ali_base = Alignment_model
        self.ali_base.load_state_dict(torch.load(Alignment_model_weight))
        if not req_grad_Alignment:
            for p in self.ali_base.parameters():
                p.requires_grad = False

        self.seq_base = Language_Base(Bert_output_dim,
                                      RB1d_config,
                                      lstm_hidden,
                                      lstm_layer,
                                      RB2d_config)
        self.activation = nn.Softmax(1)

    def forward(self, inputs_ali, inputs_seq):
        # seq channel
        seq_x = self.seq_base(inputs_seq)
        # aligment channel
        ali_x = self.ali_base(inputs_ali)

        # combine
        x = seq_x + ali_x
        # softmax
        y = self.activation(x)

        return y


# settings for alignment model
alignment_features = 71
lstm_hidden_ali = 256
lstm_layer_ali = 4

RB1d_confis_ali = [
    ("none", ResBlock_config(channel=[71]))
]

RB2d_configs_ali = [
    ("conv", ResBlock_config([5, 3, 5], [
     lstm_hidden_ali*4, 128, 98, 64], "relu")),
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


# settings for language model
Bert_output_dim = 768
lstm_hidden_lang = 168
lstm_layer_lang = 3

RB1d_confis_lang = [
    ("none", ResBlock_config(channel=[Bert_output_dim]))
]

RB2d_configs_lang = [
    ("conv", ResBlock_config([5, 3, 5], [
        lstm_hidden_lang*4, 128, 98, 64], "relu")),
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

# general parameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_epoch = 15
N_epoch_cop = 10


def train_align_model_along():
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
    alignment_channel_model = Alignment_Contact(alignment_features,
                                                RB1d_confis_ali,
                                                lstm_hidden_ali,
                                                lstm_layer_ali,
                                                RB2d_configs_ali)
    optimizer = optim.Adam(alignment_channel_model.parameters(), lr=0.00003)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,2.5]),
    #                                  ignore_diag=6)
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 5.00]))
    loss_fn = PosWeighted_CELoss(torch.tensor([0.25, 2.5]),
                                 {'long': 5.0, 'medium': 2.0, 'short': 0.0})

    # send model to gpu
    alignment_channel_model = alignment_channel_model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(N_epoch):
        train_an_epoch(epoch,
                       train_loader,
                       alignment_channel_model,
                       optimizer,
                       loss_fn)
        l_test, p_test, r_test, a_test, m_test = eval_an_epoch(epoch,
                                                               eval_loader,
                                                               alignment_channel_model,
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

    torch.save(alignment_channel_model.ali_base.state_dict(),
               './ali_base.pth.tar')


def train_seq_model_along():
    # set up datasets
    train_x, train_y = load_dataset(
        '/home/zheng/diskf/data/dataset/language_dataset_train.pkl')
    eval_x, eval_y = load_dataset(
        '/home/zheng/diskf/data/dataset/language_dataset_val.pkl')
    train_size = len(train_x)
    eval_size = len(eval_x)
    TrainSet = ContactMap_Set(train_x, train_y)
    EvalSet = ContactMap_Set(eval_x, eval_y)

    train_loader = DataLoader(TrainSet, 1, True, collate_fn=collate_fn_str)
    eval_loader = DataLoader(EvalSet, 1, True, collate_fn=collate_fn_str)

    # set up models
    sequentail_channel_model = Language_Contact(Bert_output_dim,
                                                RB1d_confis_lang,
                                                lstm_hidden_lang,
                                                lstm_layer_lang,
                                                RB2d_configs_lang)
    optimizer = optim.Adam(sequentail_channel_model.parameters(), lr=0.00001)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,3.0]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]))

    # send model to gpu
    sequentail_channel_model = sequentail_channel_model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(N_epoch):
        train_an_epoch(epoch,
                       train_loader,
                       sequentail_channel_model,
                       optimizer,
                       loss_fn)
        l_test, p_test, r_test, a_test, m_test = eval_an_epoch(epoch,
                                                               eval_loader,
                                                               sequentail_channel_model,
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

    torch.save(sequentail_channel_model.seq_base.state_dict(),
               './seq_base.pth.tar')


def train_seq_model_on_align_model():
    # set up datasets
    train_x, train_y = load_dataset(
        '/home/zheng/diskf/data/dataset/mixed_dataset_train.pkl')
    eval_x, eval_y = load_dataset(
        '/home/zheng/diskf/data/dataset/mixed_dataset_val.pkl')
    train_size = len(train_x)
    eval_size = len(eval_x)
    TrainSet = ContactMap_Set(train_x, train_y)
    EvalSet = ContactMap_Set(eval_x, eval_y)

    train_loader = DataLoader(TrainSet, 1, True, collate_fn=collate_fn_mix)
    eval_loader = DataLoader(EvalSet, 1, True, collate_fn=collate_fn_mix)

    # set up models
    alignment_channel = Alignment_Base(alignment_features,
                                       RB1d_confis_ali,
                                       lstm_hidden_ali,
                                       lstm_layer_ali,
                                       RB2d_configs_ali)

    mixed_model = Seq_OnTopOf_Alignment(Bert_output_dim,
                                        RB1d_confis_lang,
                                        lstm_hidden_lang,
                                        lstm_layer_lang,
                                        RB2d_configs_lang,
                                        alignment_channel,
                                        './ali_base.pth.tar',
                                        True)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, mixed_model.parameters()), lr=0.000005)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,2.5]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.21, 1.5]))

    # send model to gpu
    mixed_model = mixed_model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(N_epoch_cop):
        train_an_epoch(epoch,
                       train_loader,
                       mixed_model,
                       optimizer,
                       loss_fn)
        l_test, p_test, r_test, a_test, m_test = eval_an_epoch(epoch,
                                                               eval_loader,
                                                               mixed_model,
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

    torch.save(mixed_model.state_dict(), './seq_onTop_ali.pth.tar')


def main():
    print("Training alignment model\n\n")
    train_align_model_along()

    # print("Training language model\n\n")
    # train_seq_model_along()

    # print("Training language model on top of alignment model\n\n")
    # train_seq_model_on_align_model()


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    t = threading.Thread(target=main)
    t.start()
