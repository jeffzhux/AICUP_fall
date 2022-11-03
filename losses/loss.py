import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import ConfigDict

class MixUpLoss(nn.Module):
    def __init__(self) -> None:
        super(MixUpLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        (y_a, y_b, lam) = labels
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class InOutLoss(nn.Module):
    def __init__(self, lam=4) -> None:
        super(InOutLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lam = lam
        

    def forward(self, pred_in, labels_in, pred_out, lables_out):
        loss = self.criterion(pred_in, labels_in) + self.lam * self.criterion(pred_out, lables_out)
        return loss

class GroupLoss(nn.Module):
    '''
    Example : 
        >>> criterion = EngeyLoss(batch_size)
        >>> for idx, (imgs, labels),(out_imgs, _) in enumerate(zip(id_dataloader, ood_dataloader)):
            
            >>> imgs = torch.cat((imgs, out_imgs), 0)
            >>> imgs = imgs.cuda(non_blocking=True)
            >>> labels = labels.cuda(non_blocking=True)

            >>> logits= model(imgs)
            >>> loss = criterion(logits, labels)
    '''

    def __init__(self, cfg: ConfigDict)-> None:
        super(GroupLoss, self).__init__()
        args = cfg.copy()
        self.groups_range = args.groups_range
        self.num_of_group = len(self.groups_range)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, pred, label):
        loss = 0
        for idx, (start, end) in enumerate(self.groups_range):
            other_idx = -self.num_of_group + idx
            sub_pred = torch.cat((pred[:,start:end], pred[:,other_idx].view(-1,1)), dim=-1)
            sub_label = torch.cat((label[:,start:end], label[:,other_idx].view(-1,1)), dim=-1)
            loss += self.criterion(sub_pred, sub_label)
        return loss
