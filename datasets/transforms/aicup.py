from torchvision import transforms
import torchvision.transforms as T
from PIL import ImageDraw
import math
import random
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

def base():
    transform = transforms.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def baseWithAim():
    transform = transforms.Compose([
        T.RandomApply([randomAim()], p=1),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform
