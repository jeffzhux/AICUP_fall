
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import ConfigDict

class OutOfDistributionBase(nn.Module):
    def __init__(self, cfg:ConfigDict):
        super(OutOfDistributionBase, self).__init__()
        self.mode = cfg['mode']
        self.threshold = cfg['threshold']
        assert self.mode in ['softmax', 'entropy'], f'except "softmax" or "entorpy", not {self.mode}'
    def forward(self, logits):
        print(logits.size())
        
        logits = F.softmax(logits, dim=-1)
        if self.mode == 'softmax':
            return torch.max(logits, -1)[0]#.ge(self.threshold)
        else:
            print(torch.special.entr(logits).sum(-1))
            return torch.special.entr(logits).sum(-1)


class EnergyOOD(nn.Module):
    def __init__(self, cfg:ConfigDict):
        super(EnergyOOD, self).__init__()
        self.t = cfg['temperature']
    def forward(self, logits):
        # print(logits.size())
        
        # print((self.t * torch.logsumexp(logits / self.t, dim=1)))
        return (self.t * torch.logsumexp(logits / self.t, dim=1))
       



