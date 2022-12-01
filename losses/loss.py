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

class KmeanClipLoss(nn.Module):
    def __init__(self, num_extra_others, **args) -> None:
        super(KmeanClipLoss, self).__init__()
        self.num_extra_others = num_extra_others
        self.criterion = nn.CrossEntropyLoss(**args)
        self.other_criterion = nn.CrossEntropyLoss(**args)
    
    def forward(self, logit, label):
        logit = logit.softmax(dim=-1)
        class_logit = logit[:, :-(self.num_extra_others+1)]
        other_logit = logit[:, -(self.num_extra_others+1):]
        logit = torch.cat((class_logit, torch.sum(other_logit, dim=-1, keepdim=True)), dim=-1)

        class_label = label[:, :-(self.num_extra_others+1)]
        other_label = label[:, -(self.num_extra_others+1):]
        label = torch.cat((class_label, torch.sum(other_label, dim=-1, keepdim=True)), dim=-1)

        loss = self.criterion(logit, label) + self.other_criterion(other_logit, other_label)
        return loss

class ClipLoss(nn.Module):
    def __init__(self, t=0.5, **args) -> None:
        super(ClipLoss, self).__init__()
        self.img_criterion = nn.CrossEntropyLoss(**args)
        self.text_criterion = nn.CrossEntropyLoss(**args)
    
    def forward(self, logit, label):
        loss = (self.img_criterion(logit, label) + self.text_criterion(logit.t(), label.t())) / 2
        return loss

class OSDALoss(nn.Module):
    def __init__(self, t=0.5, **args) -> None:
        super(OSDALoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss(**args)
        self.trans_criterion = nn.BCELoss()
        self.t = t

    def forward(self, s_logits, t_logits, s_label, t_label):
        batch_size = s_logits.size(0)
        # unknow class prob
        t_softmax = F.softmax(t_logits, dim=1)[:,-1:] # pred
        t_label = torch.ones((batch_size,1), device=t_logits.device) * self.t # label
        
        trans_loss = self.trans_criterion(t_softmax, t_label)
        cls_loss = self.cls_criterion(s_logits, s_label)
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
