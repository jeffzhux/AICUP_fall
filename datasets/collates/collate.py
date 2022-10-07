from typing import List
import torch
import torch.nn as nn
import numpy as np


class MixUp_CollateFunction(nn.Module):
    """
        Reference (12.03.2021)
        https://arxiv.org/pdf/1710.09412.pdf
    """
    def __init__(self, alpha = 1.0):

        self.lam = np.random.beta(alpha, alpha)
        super(MixUp_CollateFunction, self).__init__()
    
    def forward(self, images, labels):

        # images, labels = zip(*batch)
        # print()
        labels = torch.stack(labels)

        batch_size = images.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_images = self.lam * images + (1-self.lam) * images[index, :]
        
        label_a, label_b = labels, labels[index]

        return mixed_images, (label_a, label_b, self.lam)

