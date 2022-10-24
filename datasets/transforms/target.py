from typing import List
import torch
import torch.nn.functional as F


def OneHotTransform(
    target: List, 
    num_of_classes: int
    ):
    '''
        Args:
            target (List) : 答案的index
            num_of_classes (int): 類別數量
    '''
    target = torch.tensor(target)
    
    target = F.one_hot(target, num_of_classes)
    target = target.type(torch.FloatTensor)
    return target

def OtherOneHotTransform(
    target: List, 
    num_of_classes: int
    ):
    '''
        Args:
            target (List) : 答案的index
            num_of_classes (int): 類別數量
    '''
    target = torch.tensor(target)
    
    target = F.one_hot(target, num_of_classes)
    target = target.type(torch.FloatTensor)
    return target