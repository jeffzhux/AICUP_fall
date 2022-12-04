import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self,alphastd):
        self.alphastd = alphastd
        self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])

    def __call__(self, img):
        if self.alphastd == 0.:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class RandomMixupCutMix(nn.Module):
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
        super(RandomMixupCutMix, self).__init__()
        self.num_classes = num_classes
        self.mixup = Mixup(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMix(num_classes, alpha=cutmix_alpha)
    def forward(self, images, labels):
        labels = F.one_hot(labels, self.num_classes)

        bs = images.size(0) // 2
        img1, lab1, mixup_lam, mixup_index = self.mixup(images[:bs], labels[:bs])
        img2, lab2, cutmix_lam, cutmix_index = self.cutmix(images[bs:], labels[bs:])

        return torch.cat((img1, img2), dim=0), torch.cat((lab1, lab2), dim=0), [mixup_lam, cutmix_lam], torch.cat((mixup_index, cutmix_index))

class Mixup(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference
        https://arxiv.org/pdf/1710.09412.pdf
    '''
    def __init__(self, num_classes, alpha=1.0):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        bs = images.size(0)
        index = torch.randperm(bs)

        images = lam * images + (1 - lam) * images[index, :]
        labels = lam * labels + (1 - lam) * labels[index]
        return images, labels, lam, index

class CutMix(nn.Module):
    '''
        Returns mixed inputs, pairs of targets, and lambda
        Reference https://arxiv.org/pdf/1905.04899v2.pdf
    '''
    def __init__(self, num_classes, alpha=1.0):
        super(CutMix, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
    def forward(self, images, labels, lam=None, index=None):

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

        return images, labels, lam, index