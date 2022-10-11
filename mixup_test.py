# import math
# import random
# from PIL import ImageDraw,Image

# img = Image.open("./data/train/asparagus/00a326ba-7e9a-4bc1-8684-4bf404bfd6bb.jpg")
# radius = [5,50]
# max_num = 10
# c_min = math.ceil(max(radius)*5/4)
# c_max = min(img.size) - c_min
# dr = ImageDraw.Draw(img)
# time = random.randint(0, max_num)
# for i in range(time):
#     x, y = random.randint(c_min,c_max), random.randint(c_min,c_max) # center
#     r = random.randint(radius[0], radius[1]) # radius
#     circle_width = max(math.ceil(r/10) - random.randint(-1,1), 1)
#     line_width = int(r/5) - random.randint(-1,1)
    
#     dr.ellipse((x-r,y-r,x+r,y+r), width=circle_width, outline = "yellow")
#     dr.line((x-r*5/4,y,x-r*3/4,y), fill='yellow', width=line_width)
#     dr.line((x+r*3/4,y,x+r*5/4,y), fill='yellow', width=line_width)
#     dr.line((x,y-r*5/4,x,y-r*3/4), fill='yellow', width=line_width)
#     dr.line((x,y+r*3/4,x,y+r*5/4), fill='yellow', width=line_width)

# img.save('test.jpg')

from numpy import dtype, float32
import torch

def accuracy_one_hot(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    target = target > 0
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / target.sum())
    return ret

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        res = []
        if len(target.size()) == 1: # general 
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else: # one hot encoding
            target = target > 1e-3
            for k in topk:
                correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
                res.append(correct.sum().mul_(100.0/target.sum()))
        return res

# a = torch.rand(2,3)
a = torch.tensor([
    [0.3320, 0.0860, 0.9418],
    [0.8687, 0.5409, 0.6854],
    [0.8687, 0.5409, 0.6854],
])
y1 = torch.tensor([2,0,1])
# y1 = torch.nn.functional.one_hot(torch.tensor([2,0,1])).type(torch.float32)
y2 = torch.nn.functional.one_hot(torch.tensor([0,2,1])).type(torch.float32)
mix_up = (y1+y2)/2

loss = torch.nn.CrossEntropyLoss()

print(f'acc:{accuracy(a, y1, topk=(1,))}, loss : {loss(a, mix_up)}')
# print(mix_up)
# print(loss(a,mix_up))
# print((loss(a,y1)+loss(a,y2))/2)
