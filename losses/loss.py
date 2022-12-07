import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.config import ConfigDict


class MixmatchLoss(nn.Module):
    def __init__(self) -> None:
        super(MixmatchLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, pred_l, labels_l, pred_u, labels_u):
        
        return self.criterion(pred_l, labels_l) + torch.mean((pred_u - labels_u)**2)

class SimilarityLoss(nn.Module):
    def __init__(self, lam = 5e-3, **args) -> None:
        super(SimilarityLoss, self).__init__()
        self.logit_criterion = nn.CrossEntropyLoss(**args)
        self.lam = lam
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, logits_metrix, similarity_metrix, label):
        B, D = similarity_metrix.size()
        
        logits_loss = self.logit_criterion(logits_metrix, label)
        
        on_diag = torch.diagonal(similarity_metrix).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(similarity_metrix).pow_(2).sum()
        sim_loss = (on_diag + self.lam * off_diag) / B

        return logits_loss + sim_loss

class KmeanClipLoss(nn.Module):
    def __init__(self, num_classes, num_extra_others, **args) -> None:
        super(KmeanClipLoss, self).__init__()
        self.num_class = num_classes
        self.num_extra_others = num_extra_others
        self.criterion = nn.CrossEntropyLoss(**args)
    
    def forward(self, logit, label):
        
        class_logit = logit[:, :-(self.num_extra_others+1)]
        other_logit = logit[:, -(self.num_extra_others+1):]
        logit = torch.cat((class_logit, torch.sum(other_logit, dim=-1, keepdim=True)), dim=-1)

        class_label = label[:, :-(self.num_extra_others+1)]
        other_label = label[:, -(self.num_extra_others+1):]
        label = torch.cat((class_label, torch.sum(other_label, dim=-1, keepdim=True)), dim=-1)

        loss = self.criterion(logit, label)
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
