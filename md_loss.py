import torch
from torch import nn

class CTmap_CrossEntropyLoss(nn.Module):
    def __init__(self, weight, ignore_diag:int):
        super(CTmap_CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.ignore_diag = ignore_diag

    def forward(self, pred_val, true_val):
        # make sure that the region in ignore diag
        # is same as the true val
        lengths = true_val.size(1)
        seq_idx = torch.arange(true_val.size(1), device=true_val.device)
        x_id, y_id = torch.meshgrid(seq_idx,seq_idx)

        ignore_region = (abs(y_id-x_id) < self.ignore_diag).unsqueeze(0)
        pad = torch.zeros([lengths,lengths], dtype=torch.bool, device=true_val.device).unsqueeze(0)

        pred_val = pred_val * (~ ignore_region)
        pred_val = pred_val + torch.stack([ignore_region,pad],dim=1)
        true_val[ignore_region] = 0
        
        loss = self.loss(pred_val, true_val)

        return loss


class Exclusive_CrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce, weight_exclusion = -1):
        super(Exclusive_CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight_ce)
        self.weight_ex = weight_exclusion
    
    def forward(self, pred_val, exclusive_pred_val, true_val):
        # idea is that the secondary model trained using exclusive CEloss
        # will only be evaluated for predictions that the primary model 
        # failed to make accurate prediction
        lengths = true_val.size(1)
        _, primer_pred = torch.max(exclusive_pred_val, 1)
        primer_pred_int8 = primer_pred.type(torch.int8)
        true_val_int8 = true_val.type(torch.int8)

        ignore_region = (true_val_int8 == primer_pred_int8)
        pad = torch.zeros([lengths, lengths], dtype=torch.bool,
                          device=true_val.device).unsqueeze(0)

        pred_val = pred_val * (~ ignore_region)
        pred_val = pred_val + torch.stack([ignore_region, pad], dim=1)
        true_val[ignore_region] = 0

        loss = self.loss(pred_val, true_val)

        return loss
