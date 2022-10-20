from typing import List
import torch
import torch.nn.functional as F


def otherOneHotTransform(
    target: List, 
    num_of_classes: int
    ):
    '''
        Args:
            target (List) : 答案的index
            num_of_classes (int): 類別數量
    '''
    target = torch.tensor(target)
    
    num_of_except_other_classes = num_of_classes-1
    if target == num_of_except_other_classes:
        # if target is others classes, flatten all probability
        target = torch.full((num_of_except_other_classes, ), 1 / num_of_except_other_classes)
    else:
        target = F.one_hot(target, num_of_except_other_classes)
    target = target.type(torch.FloatTensor)
    return target
