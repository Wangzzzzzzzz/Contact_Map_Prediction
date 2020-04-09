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



