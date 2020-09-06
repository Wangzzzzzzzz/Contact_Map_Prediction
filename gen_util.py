import pickle as pkl
import sys
from typing import Dict, Tuple, List
import numpy as np
import torch


def contact_ratio(data_dir: str, exclude_neighbor=True):
    with open(data_dir, 'rb') as pkl_hdl:
        dataset = pkl.load(pkl_hdl)
    contact_cnt = 0
    total_cnt = 0
    for _, ct_map in dataset.values():
        if not exclude_neighbor:
            contact_cnt += np.sum(ct_map)
            total_cnt += ct_map.size
        else:
            lengths = ct_map.shape[0]
            idx = np.arange(lengths)
            x_id, y_id = np.meshgrid(idx, idx)
            val_reg = (abs(x_id-y_id) >= 6)
            contact_cnt += np.sum(ct_map * val_reg)
            total_cnt += np.sum(val_reg)

    return contact_cnt/total_cnt


def pos_based_ratio(data_dir: str):
    with open(data_dir, 'rb') as pkl_hd:
        dataset = pkl.load(pkl_hd)
    long_contact, long_total = 0, 0
    medium_contact, medium_total = 0, 0
    short_contact, short_total = 0, 0
    for _, true_x in dataset:
        true_x = torch.from_numpy(true_x)
        L = true_x.size(1)
        seq_idx = torch.arange(
            true_x.size(1), device=true_x.device)
        x_id, y_id = torch.meshgrid(seq_idx, seq_idx)
        short_region = ((abs(y_id-x_id) >= 6) &
                        (abs(y_id-x_id) < 12)).unsqueeze(0)
        medium_region = ((abs(y_id-x_id) >= 12) &
                         (abs(y_id-x_id) < 24)).unsqueeze(0)
        long_region = ((abs(y_id-x_id) >= 24)).unsqueeze(0)

        short_contact += (short_region*true_x).sum().int()
        medium_contact += (medium_region*true_x).sum().int()
        long_contact += (long_region*true_x).sum().int()

        short_total += short_region.sum().int()
        medium_total += medium_region.sum().int()
        long_total += long_region.sum().int()

    print("long total: {}, long contact: {}, ratio {:f}".format(
        long_total, long_contact, float(long_contact)/float(long_total)))
    print("medium total: {}, medium contact: {}, ratio {:f}".format(
        medium_total, medium_contact, float(medium_contact)/float(medium_total)))
    print("short total: {}, short contact: {}, ratio {:f}".format(
        short_total, short_contact, float(short_contact)/float(short_total)))
