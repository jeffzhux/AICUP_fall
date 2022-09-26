from typing import List
import torch
import torch.nn as nn


class CollateFunction(nn.Module):
    def __init__(self):
        super(CollateFunction, self).__init__()
    
    def forward(self, batch: List[tuple]):
        
        images, labels = zip(*batch)
        labels = torch.stack(labels)

