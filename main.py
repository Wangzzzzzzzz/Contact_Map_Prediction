from md_util import Gathering_ResNet, ResBlock_config
from md_seq import ContactMap_feature_RCNN
from md_train import train_an_epoch, eval_an_epoch, split_train_val, ContactMap_Set, collate_fn
from torch import nn, optim
import torch 
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import sys, threading
from tape import TAPETokenizer, ProteinBertModel
from md_loss import CTmap_CrossEntropyLoss

class Sequential_Contact(nn.Module):
    def __init__(self,
                 Bert_output_dim: int,
                 conv1d_hidden: int,
                 lstm_hidden: int,
                 lstm_layer: int,
                 RB_config: List[ResBlock_config]):

        super(Sequential_Contact, self).__init__()
        self.bert = ProteinBertModel.from_pretrained('bert-base')
        self.rcnn = ContactMap_feature_RCNN(Bert_output_dim,
                                            conv1d_hidden,
                                            lstm_hidden,
                                            lstm_layer)
        self.resnet = Gathering_ResNet(RB_config)
        self.conv2d = nn.Conv2d(RB_config[-1][1].channel[3],2,1) # force output to have 2 channel
        self.activation = nn.Softmax(1)

    def forward(self, inputs):
        x = self.bert(inputs)[0]
        x = self.rcnn(x)
        x = self.resnet(x)
        x = self.conv2d(x)
        y = self.activation(x)

        return y

Bert_output_dim = 768
conv1d_hidden = 256
lstm_hidden = 128
lstm_layer = 2
RB_configs = [
    ("conv", ResBlock_config([3, 5, 1], [
        lstm_hidden*4, 256, 256, 128], "relu")),
    ("id", ResBlock_config([3, 5, 1], [128, 32, 32, 128], "relu")),
    ("id", ResBlock_config([3, 5, 1], [128, 32, 32, 128], "relu")),
    ("id", ResBlock_config([3, 5, 1], [128, 32, 32, 128], "relu")),
    ("conv", ResBlock_config([3, 5, 1], [128, 32, 32, 64], "relu")),
    ("id", ResBlock_config([3, 5, 1], [64, 16, 16, 64], "relu")),
    ("id", ResBlock_config([3, 5, 1], [64, 16, 16, 64], "relu")),
    ("id", ResBlock_config([3, 5, 1], [64, 16, 16, 64], "relu")),
    ("conv", ResBlock_config([3, 5, 1], [64, 32, 32, 16], "relu")),
    ("id", ResBlock_config([3, 5, 1], [16, 32, 32, 16], "relu")),
    ("id", ResBlock_config([3, 5, 1], [16, 32, 32, 16], "relu")),
    ("id", ResBlock_config([3, 5, 1], [16, 32, 32, 16], "relu"))
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_epoch = 50

def main():
    # set up datasets
    train_x, train_y, eval_x, eval_y = split_train_val('../data/data_set.pkl', 0.9)
    train_size = len(train_x)
    eval_size = len(eval_x)
    TrainSet = ContactMap_Set(train_x, train_y)
    EvalSet = ContactMap_Set(eval_x, eval_y)

    train_loader = DataLoader(TrainSet,1,True,collate_fn=collate_fn)
    eval_loader = DataLoader(EvalSet,1,True,collate_fn=collate_fn)


    # set up models
    sequentail_channel_model = Sequential_Contact(Bert_output_dim,
                                                conv1d_hidden,
                                                lstm_hidden,
                                                lstm_layer,
                                                RB_configs)
    optimizer = optim.Adam(sequentail_channel_model.parameters(),lr=0.000003)
    # loss_fn = CTmap_CrossEntropyLoss(weight=torch.tensor([0.1,3.0]),
    #                                  ignore_diag=6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2.0]))

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
        print("Val. Recall: {0:.3f} | Val. Acc: {1:.3f}\n".format(r_test, a_test))
        print('Specific Metrics are the following:')
        print("accu_L/10_short: {0:.3f}, accu_L/10_medium: {1:.3f}, accu_L/10_long: {2:.3f}".format(
            m_test[0], m_test[1], m_test[2]))
        print("accu_L/5_short: {0:.3f}, accu_L/5_medium: {1:.3f}, accu_L/5_long: {2:.3f}".format(
            m_test[3], m_test[4], m_test[5]))
        print("accu_L/2_short: {0:.3f}, accu_L/2_medium: {1:.3f}, accu_L/2_long: {2:.3f}".format(
            m_test[6], m_test[7], m_test[8]))
        print("accu_L/1_short: {0:.3f}, accu_L/1_medium: {1:.3f}, accu_L/1_long: {2:.3f}\n".format(
            m_test[9], m_test[10], m_test[11]))

    
    torch.save(sequentail_channel_model.state_dict(), './seq_model_20_ce.pth.tar')

if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    t = threading.Thread(target=main)
    t.start()
