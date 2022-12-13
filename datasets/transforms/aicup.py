from typing import Tuple
from torchvision import transforms
import torchvision.transforms as T
from PIL import ImageDraw
import math
import random
from torchvision.transforms.functional import InterpolationMode
from datasets.transforms.augmentations import Lighting

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class randomAim(object):
    def __init__(self, radius = [5,50], max_num = 10):
        '''
        Attributes:
            max_num (int):
                num of aim
            radius (list):
                range of radius
        '''
        # radius
        self.radius = radius
        self.max_num = max_num
    def __call__(self, img):
        c_min = math.ceil(max(self.radius)*5/4)
        c_max = min(img.size) - c_min
        dr = ImageDraw.Draw(img)
        time = random.randint(0, self.max_num)
        for i in range(time):
            x, y = random.randint(c_min,c_max), random.randint(c_min,c_max) # center
            r = random.randint(self.radius[0], self.radius[1]) # radius
            circle_width = max(math.ceil(r/10) - random.randint(-1,1), 1)
            line_width = int(r/5) - random.randint(-1,1)
            
            dr.ellipse((x-r,y-r,x+r,y+r), width=circle_width, outline = "yellow")
            dr.line((x-r*5/4,y,x-r*3/4,y), fill='yellow', width=line_width)
            dr.line((x+r*3/4,y,x+r*5/4,y), fill='yellow', width=line_width)
            dr.line((x,y-r*5/4,x,y-r*3/4), fill='yellow', width=line_width)
            dr.line((x,y+r*3/4,x,y+r*5/4), fill='yellow', width=line_width)
        return img

def base(size: Tuple = (224,224)):
    transform = transforms.Compose([
        T.Resize(size, interpolation = InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def baseOnAim(size: Tuple = (224,224)):
    transform = transforms.Compose([
        T.RandomApply([randomAim()], p=1),
        T.RandomResizedCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def baseOnImageNet(size: Tuple = (224,224)):

    transform = transforms.Compose([
        T.RandomResizedCrop(size),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def baseOnTrivialAugment(size: Tuple = (224,224), lighting: float = 0):
    trans_list = [
        T.RandomResizedCrop(size, interpolation = InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        T.ToTensor()
    ]
    if lighting:
        trans_list.append(Lighting(lighting))
    trans_list.extend([
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std']),
        T.RandomErasing(p=0.1)
    ])
    transform = transforms.Compose(trans_list)
    return transform

def fixFineTune(resize: Tuple = (320,320), cropsize: Tuple = (224, 224)):

    transform = transforms.Compose([
        T.Resize(resize),
        T.CenterCrop(cropsize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def fixTest(resize: Tuple = (320,320), cropsize: Tuple = (224, 224)):
    transform = transforms.Compose([
        T.Resize(resize),
        T.CenterCrop(cropsize),
        T.RandomHorizontalFlip(1),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform