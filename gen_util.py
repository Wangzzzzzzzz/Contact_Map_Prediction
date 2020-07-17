import pickle as pkl 
import sys 
from typing import Dict, Tuple, List
import numpy as np

def contact_ratio(data_dir:str, exclude_neighbor=True):
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
            x_id, y_id = np.meshgrid(idx,idx)
            val_reg = (abs(x_id-y_id) >= 6)
            contact_cnt += np.sum(ct_map * val_reg)
            total_cnt += np.sum(val_reg)
        
    return contact_cnt/total_cnt