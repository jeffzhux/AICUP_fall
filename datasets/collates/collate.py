from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as T

class CollateFunction(nn.Module):
    def __init__(self):
        super(CollateFunction, self).__init__()
    
    def forward(self, batch: List[tuple]):
        
        images, labels = zip(*batch)
        labels = torch.stack(labels)

class MixupCollate(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference
        https://arxiv.org/pdf/1710.09412.pdf
    '''
    def __init__(self, num_classes, alpha=1.0):
        super(MixupCollate, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, batch: List[tuple]):
        images, labels = map(list,zip(*batch))
        
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels = F.one_hot(labels, self.num_classes)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        images = lam * images + (1 - lam) * images[index, :]
        labels = lam * labels + (1 - lam) * labels[index]
        return images, labels

class TestTimeCollate(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference
        https://arxiv.org/pdf/1710.09412.pdf
    '''
    def __init__(self):
        super(TestTimeCollate, self).__init__()
        

    def forward(self, batch: List[tuple]):
        images, labels =  map(list, zip(*batch))
        images = torch.cat(images)
        labels = torch.tensor(labels)

        return images, labels

class OtherMixupCollate(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference
        https://arxiv.org/pdf/1710.09412.pdf
    '''
    def __init__(self, num_classes, alpha=1.0):
        super(OtherMixupCollate, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, batch: List[tuple]):
        images, labels, idx = map(list,zip(*batch))
        
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels = F.one_hot(labels, self.num_classes)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        images = lam * images + (1 - lam) * images[index, :]
        labels = lam * labels + (1 - lam) * labels[index]
        return images, labels, idx