import numpy as np
import os
import csv
import torch
from torchvision.datasets import ImageFolder, VisionDataset
import  torch.nn.functional as F
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from datasets.tokenizer import area_vocab
from datasets.transforms.aicup import base
from torch.utils.data import Dataset
from PIL import Image

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
        target = F.one_hot(target, self.num_of_classes)
        target = target.type(torch.float32)
        return sample, target

class OOD_ImageFolder(ImageFolder):
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

        self.num_of_classes = 2
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
        target = torch.where(target == self.class_to_idx['others'], 1, 0)
        target = F.one_hot(target, self.num_of_classes)
        target = target.type(torch.float32)
        return sample, target

class TestTimeAICUP_DataSet(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        base_transform: Optional[Callable] = base,
        num_of_trans: int = 0):

        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.num_of_trans = num_of_trans
        self.num_of_classes = len(self.classes)
        self.base_transform = base_transform
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

class loc_Dataset(Dataset):
    def __init__(self, sample, num_of_classes, transforms=None):
        super(loc_Dataset, self).__init__()
        self.sample = sample
        self.transform = transforms
        self.num_of_classes = num_of_classes

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        path, target, loc = self.sample[index]
        
        sample = self.pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(target)
        target = F.one_hot(target, self.num_of_classes)
        target = target.type(torch.float32)

        return sample,  target, loc
    
    def __len__(self):
        return len(self.sample)


class loc_ImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        loc_file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.loc_file_path = loc_file_path
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_of_classes = len(self.classes)

    def has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        if 'others' in classes:
            classes.remove('others')
            classes.append('others')
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        filename_to_loc = {}
        with open(self.loc_file_path, mode = 'r') as inp:
            reader = csv.reader(inp)
            for i, v in enumerate(reader):
                if i > 0:
                    filename_to_loc[v[1]] = [
                        (float(v[7]) - 21.896823) / (25.299653 - 21.896823),
                        (float(v[6]) - 120.035198) / (122.007112 - 120.035198)
                        ]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index, filename_to_loc[fname]
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target, loc = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(target)
        target = F.one_hot(target, self.num_of_classes)
        target = target.type(torch.float32)

        return sample, target, loc



class Kmean_ImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        num_extra_others: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_of_classes = len(self.classes)
        self.num_extra_others = num_extra_others

        self.extra_others_targets = [(idx, 0) for idx, s in enumerate(self.samples) if s[1] == self.class_to_idx['others']]

    def has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        if 'others' in classes:
            classes.remove('others')
            classes.append('others')
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        filename_to_loc = {}
        with open('./data/ID/tag_locCoor.csv', mode = 'r') as inp:
            reader = csv.reader(inp)
            for i, v in enumerate(reader):
                if i > 0:
                    filename_to_loc[v[1]] = [
                        (float(v[7]) - 21.896823) / (25.299653 - 21.896823),
                        (float(v[6]) - 120.035198) / (122.007112 - 120.035198)
                        ]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index, filename_to_loc[fname]
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, loc = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        others_idx, others_targets = zip(*self.extra_others_targets)
        target += others_targets[index-others_idx[0]] if index in others_idx else 0
        target = torch.tensor(target)
        target = F.one_hot(target, (self.num_of_classes + self.num_extra_others))
        target = target.type(torch.float32)

        return sample, target, loc, 

class Clip_ImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        loc_file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.loc_file_path = loc_file_path
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        self.num_of_classes = len(self.classes)

    def has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        if 'others' in classes:
            classes.remove('others')
            classes.append('others')
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        filename_to_loc = {}
        with open(self.loc_file_path, mode = 'r') as inp:
            reader = csv.reader(inp)
            for i, v in enumerate(reader):
                if i > 0:
                    filename_to_loc[v[1]] = [
                        (float(v[7]) - 21.896823) / (25.299653 - 21.896823),
                        (float(v[6]) - 120.035198) / (122.007112 - 120.035198),
                        [area_vocab[v[4]], area_vocab[v[5]]]
                        ]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index, filename_to_loc[fname][:2], filename_to_loc[fname][2]
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, loc, text = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # target = torch.tensor(target)
        # target = F.one_hot(target, self.num_of_classes)
        # target = target.type(torch.float32)

        return sample, target, loc, text

class ClipUnlabel_ImageFolder(Clip_ImageFolder):
    def __init__(self, **args):
        super().__init__(**args)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target, loc, text = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample1, sample2 = self.transform(sample), self.transform(sample)

        return sample1, sample2, loc, text
