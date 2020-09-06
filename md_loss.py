import torch
from torch import nn
from typing import List, Tuple, Dict


class CTmap_CrossEntropyLoss(nn.Module):
    def __init__(self, weight, ignore_diag: int):
        super(CTmap_CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.ignore_diag = ignore_diag

    def forward(self, pred_val, true_val):
        # make sure that the region in ignore diag
        # is same as the true val
        lengths = true_val.size(1)
        seq_idx = torch.arange(true_val.size(1), device=true_val.device)
        x_id, y_id = torch.meshgrid(seq_idx, seq_idx)

        ignore_region = (abs(y_id-x_id) < self.ignore_diag).unsqueeze(0)
        pad = torch.zeros([lengths, lengths], dtype=torch.bool,
                          device=true_val.device).unsqueeze(0)

        pred_val = pred_val * (~ ignore_region)
        pred_val = pred_val + torch.stack([ignore_region, pad], dim=1)
        true_val[ignore_region] = 0

        loss = self.loss(pred_val, true_val)

        return loss


class Exclusive_CrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce, weight_exclusion=-1):
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


class PosWeighted_CELoss(nn.Module):
    def __init__(self, weight_class, weight_pos: Dict):
        super(PosWeighted_CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight_class, reduction="none")
        self.weight_pos = weight_pos

    def forward(self, pred_x, true_x):
        L = true_x.size(1)
        seq_idx = torch.arange(
            true_x.size(1), device=true_x.device)
        x_id, y_id = torch.meshgrid(seq_idx, seq_idx)
        short_region = ((abs(y_id-x_id) >= 6) &
                        (abs(y_id-x_id) < 12)).unsqueeze(0)
        medium_region = ((abs(y_id-x_id) >= 12) &
                         (abs(y_id-x_id) < 24)).unsqueeze(0)
        long_region = ((abs(y_id-x_id) >= 24)).unsqueeze(0)
        non_red_loss = self.loss(pred_x, true_x)

        loss = ((non_red_loss*short_region*self.weight_pos['short']).sum() +
                (non_red_loss*medium_region*self.weight_pos['medium']).sum() +
                (non_red_loss*long_region*self.weight_pos['long']).sum())/(L*L)

        return loss


class multiWeighted_CELoss(nn.Module):
    def __init__(self,
                 weight_short,
                 weight_medium,
                 weight_long):
        self.loss_short = nn.CrossEntropyLoss(
            weight=weight_short, reduction="none")
        self.loss_medium = nn.CrossEntropyLoss(
            weight=weight_medium, reduction="none")
        self.loss_long = nn.CrossEntropyLoss(
            weight=weight_long, reduction="none")

    def forward(self, pred_x, true_x):
        L = true_x.size(1)
        seq_idx = torch.arange(
            true_x.size(1), device=true_x.device)
        x_id, y_id = torch.meshgrid(seq_idx, seq_idx)
        short_region = ((abs(y_id-x_id) >= 6) &
                        (abs(y_id-x_id) < 12)).unsqueeze(0)
        medium_region = ((abs(y_id-x_id) >= 12) &
                         (abs(y_id-x_id) < 24)).unsqueeze(0)
        long_region = ((abs(y_id-x_id) >= 24)).unsqueeze(0)

        loss = ((self.loss_short(pred_x, true_x)*short_region).sum() +
                (self.loss_medium(pred_x, true_x)*medium_region).sum() +
                (self.loss_long(pred_x, true_x)*long_region).sum())/(L*L)

        return loss
