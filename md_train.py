import os
import logging
import json
import numpy as np
import pickle as pkl
import torch
from torch import nn, optim
import random
from torch.utils.data import DataLoader, Dataset
from tape import ProteinBertModel, TAPETokenizer
from tqdm import tqdm
from typing import List, Dict, Tuple

tokenizer = TAPETokenizer(vocab='iupac')


def split_train_val(path_ds: str, tr_prot: float):
    # random.seed(134683)
    with open(path_ds, 'rb') as pkl_handle:
        contactmap_dataset = pkl.load(pkl_handle)
    dataset = list(contactmap_dataset.values())
    # shuffle dataset
    random.shuffle(dataset)
    ds_len = len(dataset)
    tr_len = int(tr_prot * ds_len)
    x = []
    y = []
    for item in dataset:
        x.append(item[0])
        y.append(item[1])

    return (x[0:tr_len], y[0:tr_len], x[tr_len:], y[tr_len:])


def load_dataset(path_ds: str):
    """ load the dataset"""
    with open(path_ds, 'rb') as pkl_handle:
        dataset = pkl.load(pkl_handle)
    # shuffle dataset
    random.shuffle(dataset)
    x = []
    y = []
    for item in dataset:
        if len(item) == 2:
            x.append(item[0])
            y.append(item[-1])
        elif len(item) == 3:
            x.append((item[0], item[1]))
            y.append(item[-1])
        elif len(item) == 4:
            x.append((item[0], item[1], item[2]))
            y.append(item[-1])
        else:
            raise ValueError('Too many arguments in dataset!')

    return x, y


class ContactMap_Set(Dataset):
    def __init__(self,
                 x_list: List,
                 y_list: List):

        super(ContactMap_Set, self).__init__()
        self.x = x_list
        self.y = y_list

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def collate_fn_str(sample):
    global tokenizer
    assert len(sample) == 1, "Protein Model handles batch size of 1 only"
    seq = []
    feat_2d = []
    ct_mp = []
    for (sequence, pairwise), contactmap in sample:
        seq.append(tokenizer.encode(sequence))
        feat_2d.append(pairwise)
        ct_mp.append(contactmap)

    tensor_ctmap = torch.tensor(ct_mp)
    tensor_seq = torch.tensor(seq)
    tensor_feat_2d = torch.tensor(feat_2d)

    return ((tensor_seq, tensor_feat_2d), tensor_ctmap)


def collate_fn_ali(sample):
    assert len(sample) == 1, "Protein Model handles batch size of 1 only"
    ali = []
    feat_2d = []
    ct_mp = []
    for (sequence, pairwise), contactmap in sample:
        ali.append(sequence)
        feat_2d.append(pairwise)
        ct_mp.append(contactmap)

    tensor_ctmap = torch.tensor(ct_mp)
    tensor_ali = torch.tensor(ali).float()
    tensor_feat_2d = torch.tensor(feat_2d).float()

    return ((tensor_ali, tensor_feat_2d), tensor_ctmap)


def collate_fn_mix(sample):
    assert len(sample) == 1, "Protein Model handles batch size of 1 only"
    seq = []
    ali = []
    feat_2d = []
    ct_mp = []
    for (alignment, sequence, pairwise), contactmap in sample:
        seq.append(tokenizer.encode(sequence))
        ali.append(alignment)
        feat_2d.append(pairwise)
        ct_mp.append(contactmap)

    tensor_ctmap = torch.tensor(ct_mp)
    tensor_seq = torch.tensor(seq)
    tensor_ali = torch.tensor(ali).float()
    tensor_feat_2d = torch.tensor(feat_2d).float()

    return ((tensor_ali, tensor_seq, tensor_feat_2d), tensor_ctmap)


def train_an_epoch(epoch_id: int,
                   train_loader: DataLoader,
                   model,
                   optimizer,
                   loss_fn,
                   clip_weight=None,
                   clip_weight_val=0.0,
                   clip_grad=None,
                   clip_grad_val=0.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_len = len(train_loader)
    model.train()
    with tqdm(train_loader, desc="Run Train", total=total_len) as progress:
        for batch_x, batch_y in progress:
            optimizer.zero_grad()

            if len(batch_x) == 3:
                batch_x_ali = batch_x[0]
                batch_x_seq = batch_x[1]
                batch_x_2d = batch_x[2]
                batch_x_ali = batch_x_ali.to(device)
                batch_x_seq = batch_x_seq.to(device)
                batch_x_2d = batch_x_2d.to(device)
                batch_y = batch_y.to(device)
                pred_y = model(batch_x_ali, batch_x_seq, batch_x_2d)
            elif len(batch_x) == 2:
                batch_x1d = batch_x[0]
                batch_x2d = batch_x[1]
                batch_x1d = batch_x1d.to(device)
                batch_x2d = batch_x2d.to(device)
                batch_y = batch_y.to(device)
                pred_y = model(batch_x1d, batch_x2d)
            elif not isinstance(batch_x, tuple):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred_y = model(batch_x)
            else:
                raise ValueError('Cannot handle that many arguments!')

            loss = loss_fn(pred_y, batch_y)

            progress.set_postfix(loss='{:05.3f}'.format(loss.item()))

            loss.backward()
            if not (clip_grad is None):
                clip_grad(model, clip_grad_val)
            optimizer.step()
            if not (clip_weight is None):
                clip_weight(model, clip_weight_val)

    print('epoch finished')


def metric_eval(pred_res, actu_res):
    """ 
    return the following structure:

    array[accu_L/10_short, accu_L/10_medium, accu_L/10_long,
          accu_L/5_short, accu_L/5_medium, accu_L/5_long,
          accu_L/2_short, accu_L/2_medium, accu_L/2_long,
          accu_L/1_short, accu_L/1_medium, accu_L/1_long],

    (recall_overall, precision_overall, accu_overall)
    """

    with torch.no_grad():
        lengths = actu_res.size(1)
        seq_idx = torch.arange(actu_res.size(1), device=actu_res.device)
        x_id, y_id = torch.meshgrid(seq_idx, seq_idx)

        valid_region = (abs(y_id-x_id) >= 6).long().unsqueeze(0)
        short_region = ((abs(y_id-x_id) >= 6) &
                        (abs(y_id-x_id) < 12)).long().unsqueeze(0)
        medium_region = ((abs(y_id-x_id) >= 12) &
                         (abs(y_id-x_id) < 24)).long().unsqueeze(0)
        long_region = ((abs(y_id-x_id) >= 24)).long().unsqueeze(0)
        regions = [short_region, medium_region, long_region]
        cutoffs = [10, 5, 2, 1]

        _, pred = torch.max(pred_res, 1)
        prob = pred_res[:, 1, :, :]

        # overall precision & recall & accu
        overall_prec = torch.tensor(0.0, device=actu_res.device)
        overall_reca = torch.tensor(0.0, device=actu_res.device)
        overall_accu = ((actu_res == pred) *
                        valid_region).sum().float() / valid_region.sum().float()

        for true_y, pred_y in zip(actu_res, pred):
            # overall metric
            tp = (true_y * pred_y * valid_region).sum().float()
            tn = ((1-true_y) * (1-pred_y) * valid_region).sum().float()
            fp = ((1-true_y) * pred_y * valid_region).sum().float()
            fn = (true_y * (1-pred_y) * valid_region).sum().float()

            epsilon = 1e-7

            prec = tp / (tp + fp + epsilon)
            rec = tp / (tp + fn + epsilon)

            overall_prec += prec
            overall_reca += rec

        metric_accu = []

        for cut in cutoffs:
            for reg in regions:
                total = 0
                correct = 0
                prec_item = 0
                for true_y, prob_y, pred_y in zip(actu_res, prob, pred):
                    target_reg_prob = (prob_y * reg).view(-1)
                    most_likely = target_reg_prob.topk(
                        lengths // cut, sorted=False)
                    selected = true_y.view(-1).gather(0, most_likely.indices)
                    # gather the part where
                    validation = pred_y.view(-1).gather(0, most_likely.indices)

                    correct += selected.sum().float()
                    total += selected.numel()

                metric_accu.append((correct/total).item())

        return (overall_accu, overall_prec, overall_reca), np.array(metric_accu)


def eval_an_epoch(epoch_id: int,
                  eval_loader: DataLoader,
                  model,
                  loss_fn,
                  eval_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_len = len(eval_loader)
    model.eval()
    total_loss, total_percision, total_recal, total_acc = 0.0, 0.0, 0.0, 0.0
    detailed_metric = np.zeros(12)
    with torch.no_grad():
        with tqdm(eval_loader, desc="Run Eval", total=total_len) as progress:
            for batch_x, batch_y in progress:
                batch_size = batch_y.size(0)
                if len(batch_x) == 3:
                    batch_x_ali = batch_x[0]
                    batch_x_seq = batch_x[1]
                    batch_x_2d = batch_x[2]
                    batch_x_ali = batch_x_ali.to(device)
                    batch_x_seq = batch_x_seq.to(device)
                    batch_x_2d = batch_x_2d.to(device)
                    batch_y = batch_y.to(device)
                    pred_y = model(batch_x_ali, batch_x_seq, batch_x_2d)
                elif len(batch_x) == 2:
                    batch_x1d = batch_x[0]
                    batch_x2d = batch_x[1]
                    batch_x1d = batch_x1d.to(device)
                    batch_x2d = batch_x2d.to(device)
                    batch_y = batch_y.to(device)
                    pred_y = model(batch_x1d, batch_x2d)
                elif not isinstance(batch_x, tuple):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    pred_y = model(batch_x)
                else:
                    raise ValueError('Cannot handle that many arguments!')
                loss = loss_fn(pred_y, batch_y)
                (acc, percision, recall), spe_metric = metric_eval(pred_y, batch_y)

                progress.set_postfix(perc='{:05.3f}'.format(percision.item()/batch_size),
                                     recal='{:05.3f}'.format(
                                         recall.item()/batch_size),
                                     accu='{:05.3f}'.format(acc.item()))

                total_loss += loss.item()
                total_percision += percision.item()
                total_recal += recall.item()
                total_acc += acc.item()
                detailed_metric += spe_metric

    return total_loss/total_len, total_percision/eval_size, total_recal/eval_size, total_acc/total_len, detailed_metric/total_len
