from numpy import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixUpLoss(nn.Module):
    def __init__(self) -> None:
        super(MixUpLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        (y_a, y_b, lam) = labels
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class EnergyLoss(nn.Module):
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

    def __init__(self, m_in:int=-25, m_out:int=-7)-> None:
        super(EnergyLoss, self).__init__()
        
        self.m_in = m_in
        self.m_out = m_out

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        batch_size = labels.size(0)
        Ec_out = -torch.logsumexp(pred[batch_size:], dim=1)
        Ec_in = -torch.logsumexp(pred[:batch_size], dim=1)

        loss = self.criterion(pred[:batch_size], labels)
        loss += 0.1 * (
            torch.pow((Ec_in - self.m_in).clamp_(0), 2).mean() + 
            torch.pow((self.m_out - Ec_out).clamp_(0), 2).mean()
        )
        return loss

class OELoss(nn.Module):
    '''
    Example : 
        >>> criterion = OELoss(batch_size)
        >>> for idx, (imgs, labels),(out_imgs, _) in enumerate(zip(id_dataloader, ood_dataloader)):
            
            >>> imgs = torch.cat((imgs, out_imgs), 0)
            >>> imgs = imgs.cuda(non_blocking=True)
            >>> labels = labels.cuda(non_blocking=True)

            >>> logits= model(imgs)
            >>> loss = criterion(logits, labels)
    '''

    def __init__(self)-> None:
        super(OELoss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        batch_size = labels.size(0)

        loss = self.criterion(pred[:batch_size], labels)
        loss += 0.5 * -(
            pred[batch_size:].mean(1) - 
            torch.logsumexp(pred[batch_size:], dim=1)
        ).mean()
        return loss