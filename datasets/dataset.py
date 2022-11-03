import numpy as np

import torch
from torchvision.datasets import ImageFolder
import  torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple, List, Dict

from datasets.transforms.aicup import base, fixTest

class AICUP_ImageFolder(ImageFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        self.num_of_classes = len(self.classes)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(target)
        target = F.one_hot(target, self.num_of_classes + self.num_of_groups)

        return sample, target

class TestTimeAICUP_DataSet(ImageFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_of_trans: int = 0
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.num_of_trans = num_of_trans
        self.num_of_classes = len(self.classes)
        self.base_transform = base()
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        # targets = [target] * (self.num_of_trans + 1)
        if self.transform is not None:
            sample = torch.stack([self.base_transform(sample)]+[self.transform(sample) for i in range(self.num_of_trans)])
        if self.target_transform is not None:
            target = self.target_transform(target, self.num_of_classes)

        return sample, target

class Group_ImageFolder(ImageFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(
        self,
        root: str,
        groups: list,
        groups_range: list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None):
        
        self.groups = groups
        self.groups_range = groups_range
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.num_of_groups = len(self.groups)
        self.num_of_classes = len(self.classes)
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = []
        for g in self.groups:
            classes.extend(sorted(g))

        # classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        other_idx = [idx+self.num_of_classes for idx, s_e in enumerate(self.groups_range) if not s_e[0] <= target < s_e[1]]

        target = torch.tensor(target)
        target = F.one_hot(target, self.num_of_classes + self.num_of_groups)
        target[other_idx] = 1
        target = target.type(torch.float32)
        return sample, target