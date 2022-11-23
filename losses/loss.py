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


class OSDALoss(nn.Module):
    def __init__(self, t=0.5, **args) -> None:
        super(OSDALoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**args)
        self.t = t
        

    def forward(self, s_logits, t_logits, s_label, t_label):
        batch_size = s_logits.size(0)
       
        t_softmax = F.softmax(t_logits, dim=1)
        unknow_class_prob = t_softmax[:, -1]
        know_class_prob = 1. - unknow_class_prob

        unknow_target = torch.ones((batch_size,1), device=t_logits.device) * self.t
        know_target = 1. - unknow_target

        trans_loss = - torch.mean(unknow_target * torch.log(unknow_class_prob + 1e-6)) \
                     - torch.mean(know_target * torch.log(know_class_prob + 1e-6))

        cls_loss = self.criterion(s_logits, s_label)
        return cls_loss + trans_loss

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
