from cv2 import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class CoTeachingLoss(nn.Module):
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

    def __init__(self, epochs, forget_rate=0.2, exponent=1)-> None:
        super(CoTeachingLoss, self).__init__()

        self.forget_rate = forget_rate
        self.forget_rage_schedule = np.ones(epochs) * forget_rate
        self.forget_rage_schedule[:epochs] = np.linspace(0, forget_rate**exponent, epochs)

    def forward(self, y1, y2, t, epoch):
        loss1 = F.cross_entropy(y1, t, reduction='none')
        idx1_sorted = np.argsort(loss1.data.cpu())
        loss1_sorted = loss1[idx1_sorted]

        loss2 = F.cross_entropy(y2, t, reduction='none')
        idx2_sorted = np.argsort(loss2.data.cpu())

        remeber_rate = 1-self.forget_rage_schedule[epoch]
        num_remember = int(remeber_rate * len(loss1_sorted))

        idx1_update = idx1_sorted[:num_remember]
        idx2_update = idx2_sorted[:num_remember]

        loss1_update = F.cross_entropy(y1[idx2_update], t[idx2_update])
        loss2_update = F.cross_entropy(y2[idx1_update], t[idx1_update])

        return torch.sum(loss1_update) / num_remember, torch.sum(loss2_update) / num_remember
