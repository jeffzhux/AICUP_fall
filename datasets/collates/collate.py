from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T

from models.build import build_model

class CollateFunction(nn.Module):
    def __init__(self):
        super(CollateFunction, self).__init__()
    
    def forward(self, batch: List[tuple]):
        
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels

class NoiseStudentCollateFunction(nn.Module):
    def __init__(self, model_cfg, load, context_length, t):
        super(NoiseStudentCollateFunction, self).__init__()
        checkpoint = torch.load(load)
        self.model = build_model(model_cfg)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.label_token = self.model.label_token.to('cpu')

        self.context_length = context_length
        self.T = t
        
    def forward(self, batch: List[tuple]):
        
        img, lab, loc, text = map(list,zip(*batch))
        img = torch.stack(img)
        # lab = torch.tensor(lab).cuda(non_blocking=True)
        loc = torch.tensor(loc).view(-1, 2)
        text = torch.tensor(text).view(-1, self.context_length)
        self.model.eval()
        with torch.no_grad():
            pseudolabel = self.model(img, loc, text)
            pseudolabel = torch.softmax(pseudolabel, dim=1)**(1 / self.T)
        return img, pseudolabel, loc, text

class ClipCollateFunction(nn.Module):
    def __init__(self, context_length):
        super(ClipCollateFunction, self).__init__()
        self.context_length = context_length
    def forward(self, batch: List[tuple]):
        
        img, lab, loc, text = map(list,zip(*batch))
        img = torch.stack(img)
        lab = torch.tensor(lab)
        loc = torch.tensor(loc).view(-1, 2)
        text = torch.tensor(text).view(-1, self.context_length)
        return img, lab, loc, text

class ClipTestTimeCollateFunction(nn.Module):
    def __init__(self, context_length):
        super(ClipTestTimeCollateFunction, self).__init__()
        self.context_length = context_length
    def forward(self, batch: List[tuple]):
        *img, lab, loc, text = map(list,zip(*batch))
        for i in range(len(img)):
            img[i] = torch.stack(img[i])
        lab = torch.tensor(lab)
        loc = torch.tensor(loc).view(-1, 2)
        text = torch.tensor(text).view(-1, self.context_length)

        return img, lab, loc, text

class ClipFunction(nn.Module):
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
        super(ClipFunction, self).__init__()
        self.num_classes = num_classes
        self.mixup = MixupCollate(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollate(num_classes, alpha=cutmix_alpha)

    def forward(self, batch: List[tuple]):
        
        img, lab, loc, text = map(list,zip(*batch))

        batch = list(map(lambda x, y: (x, y), img, lab))
        bs = len(batch) // 2
        img1, lab1 = self.mixup(batch[:bs], len(batch), position = 0)
        img2, lab2 = self.cutmix(batch[bs:], len(batch), position = 1)

        loc = torch.tensor(loc)
        text = torch.tensor(text)
        return torch.concat((img1, img2), dim=0), torch.concat((lab1, lab2), dim=0), loc, text

class ClipUnlabelCollateFunction(nn.Module):
    def __init__(self, context_length):
        super(ClipUnlabelCollateFunction, self).__init__()
        self.context_length = context_length

    def forward(self, batch: List[tuple]):
        img1, img2, loc, text = map(list,zip(*batch))
        img1 = torch.stack(img1)
        img2 = torch.stack(img2)
        text = torch.tensor(text).view(-1, self.context_length)
        loc = torch.tensor(loc).view(-1, 2)

        return img1, img2, loc, text

class locCollateFunction(nn.Module):
    def __init__(self):
        super(locCollateFunction, self).__init__()

    def forward(self, batch: List[tuple]):
        
        img, lab, loc = map(list,zip(*batch))
        img = torch.stack(img)
        lab = torch.stack(lab)
        loc = torch.tensor(loc)
        return img, lab, loc

class locCollate(nn.Module):
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
        super(locCollate, self).__init__()
        self.num_classes = num_classes
        self.mixup = MixupCollate(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollate(num_classes, alpha=cutmix_alpha)
    def forward(self, batch: List[tuple]):
        
        img, lab, loc = map(list,zip(*batch))
        batch = list(map(lambda x, y: (x, y), img, lab))
        bs = len(batch) // 2
        img1, lab1 = self.mixup(batch[:bs])
        img2, lab2 = self.cutmix(batch[bs:])
        loc = torch.tensor(loc)
        return torch.concat((img1, img2), dim=0), torch.concat((lab1, lab2), dim=0), loc

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

    def forward(self, batch: List[tuple], cosineSimilarities_size: int = None, position:int = 0):
        images, labels = map(list,zip(*batch))
        
        images = torch.stack(images)
        labels = torch.stack(labels)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        bs = images.size(0)
        index = torch.randperm(bs)

        images = lam * images + (1 - lam) * images[index, :]
        if cosineSimilarities_size:
            labels = lam * F.one_hot(torch.arange(bs) + bs * position, num_classes=cosineSimilarities_size) \
                 + (1-lam) * F.one_hot(index + bs * position, num_classes=cosineSimilarities_size)
        else:
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
    def forward(self, batch: List[tuple], cosineSimilarities_size: int = None, position:int = 0):
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
        if cosineSimilarities_size:
            labels = lam * F.one_hot(torch.arange(bs) + bs * position, num_classes=cosineSimilarities_size) \
                 + (1-lam) * F.one_hot(index + bs * position, num_classes=cosineSimilarities_size)
        else:
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
