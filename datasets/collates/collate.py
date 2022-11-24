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
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels

class OSDACollate(nn.Module):
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
        super(OSDACollate, self).__init__()
        self.num_classes = num_classes
        self.mixup = MixupCollate(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollate(num_classes, alpha=cutmix_alpha)
    def forward(self, batch: List[tuple]):
        # # other is at the end of list
        st = len(batch) // 2 # source and target size
        batch.sort(key=lambda x: str(self.num_classes-1).__eq__(str(x[1])))

        img, lab = map(list,zip(*batch))
        lab = torch.tensor(lab)
        lab = F.one_hot(lab, self.num_classes).type(torch.float32)

        source_batch = list(map(lambda x, y: (x, y), img[:st], lab[:st])) # source
        target_batch = list(map(lambda x, y: (x, y), img[st:], lab[st:])) # target

        img1, lab1 = self.mixup(source_batch[:st//2])
        img2, lab2 = self.cutmix(source_batch[st//2:])
        img3, lab3 = self.mixup(target_batch[:st//2])
        img4, lab4 = self.cutmix(target_batch[st//2:])
        
        return torch.concat((img1, img2, img3, img4), dim=0), torch.concat((lab1, lab2, lab3, lab4), dim=0)

class RandomMixupCutMixCollate(nn.Module):
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
        super(RandomMixupCutMixCollate, self).__init__()
        self.num_classes = num_classes
        self.mixup = MixupCollate(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollate(num_classes, alpha=cutmix_alpha)
    def forward(self, batch: List[tuple]):
        bs = len(batch) // 2
        img1, lab1 = self.mixup(batch[:bs])
        img2, lab2 = self.cutmix(batch[bs:])
        return torch.concat((img1, img2), dim=0), torch.concat((lab1, lab2), dim=0)

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
        labels = torch.stack(labels)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        images = lam * images + (1 - lam) * images[index, :]
        labels = lam * labels + (1 - lam) * labels[index]
        return images, labels

class CutMixCollate(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference https://arxiv.org/pdf/1905.04899v2.pdf
    '''
    def __init__(self, num_classes, alpha=1.0):
        super(CutMixCollate, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
    def forward(self, batch: List[tuple]):
        images, labels = map(list,zip(*batch))
            
        images = torch.stack(images)
        labels = torch.stack(labels)

        bs = images.size(0) # batch_size
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        # create bbox
        W, H = images.size()[-2], images.size()[-1]
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        index = torch.randperm(bs)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        images[ :, :, bbx1 : bbx2, bby1 : bby2] = images[index, :, bbx1 : bbx2, bby1 : bby2]
        labels = lam * labels + (1 - lam) * labels[index]
        
        return images, labels
class TestTimeCollate(nn.Module):
    def __init__(self):
        super(TestTimeCollate, self).__init__()
        

    def forward(self, batch: List[tuple]):
        images, labels =  map(list, zip(*batch))
        images = torch.cat(images)
        labels = torch.tensor(labels)

        return images, labels
