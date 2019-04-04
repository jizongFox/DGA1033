# coding=utf8
from __future__ import print_function, division
import sys
import os
import random
import re
from itertools import repeat
from pathlib import Path
from typing import Callable, Dict, List, Match, Pattern, TypeVar, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from pathlib import Path
from admm_research import ModelMode

# sys.path.insert(0,str(Path(__file__).parent))
default_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])


def make_dataset(root, mode, subfolder="WeaklyAnnotations"):
    assert mode in ['train', 'val', 'test']
    items = []

    if mode == 'train':
        try:
            train_img_path = os.path.join(root, 'train', 'Img')
            images = os.listdir(train_img_path)
        except FileNotFoundError:
            train_img_path = os.path.join(root, 'train', 'img')
            images = os.listdir(train_img_path)
        try:
            train_mask_path = os.path.join(root, 'train', 'GT')
            labels = os.listdir(train_mask_path)
        except FileNotFoundError:
            train_mask_path = os.path.join(root, 'train', 'gt')
            labels = os.listdir(train_mask_path)

        train_mask_weak_path = os.path.join(root, 'train', subfolder)
        labels_weak = os.listdir(train_mask_weak_path)

        images.sort()
        labels.sort()
        labels_weak.sort()

        for it_im, it_gt, it_w in zip(images, labels, labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt),
                    os.path.join(train_mask_weak_path, it_w))
            items.append(item)

    elif mode == 'val':
        try:
            train_img_path = os.path.join(root, 'val', 'Img')
            images = os.listdir(train_img_path)
        except FileNotFoundError:
            train_img_path = os.path.join(root, 'val', 'img')
            images = os.listdir(train_img_path)
        try:
            train_mask_path = os.path.join(root, 'val', 'GT')
            labels = os.listdir(train_mask_path)
        except FileNotFoundError:
            train_mask_path = os.path.join(root, 'val', 'gt')
            labels = os.listdir(train_mask_path)

        train_mask_weak_path = os.path.join(root, 'val', subfolder)

        labels_weak = os.listdir(train_mask_weak_path)

        images.sort()
        labels.sort()
        labels_weak.sort()

        for it_im, it_gt, it_w in zip(images, labels, labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt),
                    os.path.join(train_mask_weak_path, it_w))
            items.append(item)
    else:
        try:
            train_img_path = os.path.join(root, 'val', 'Img')
        except FileNotFoundError:
            train_img_path = os.path.join(root, 'val', 'img')
        try:
            train_mask_path = os.path.join(root, 'val', 'GT')
        except FileNotFoundError:
            train_mask_path = os.path.join(root, 'val', 'gt')
        train_mask_weak_path = os.path.join(root, 'test', subfolder)

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labels_weak = os.listdir(train_mask_weak_path)

        images.sort()
        labels.sort()
        labels_weak.sort()

        for it_im, it_gt, it_w in zip(images, labels, labels_weak):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt),
                    os.path.join(train_mask_weak_path, it_w))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):

    def __init__(self, root_dir, mode, subfolder='WeaklyAnnotations', transform=None, augment=None, equalize=False,
                 metainfoGenerator_dict: dict = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name = mode + '_dataset'
        self.root_dir = root_dir
        self.transform = transform
        self.subfolder = subfolder
        self.imgs = make_dataset(root_dir, mode, subfolder)
        self.augment = augment
        self.equalize = equalize
        self.training = ModelMode.TRAIN
        if metainfoGenerator_dict is not None:
            from . import metainfoGenerator
            self.metainGenerator = getattr(metainfoGenerator, metainfoGenerator_dict['name']) \
                (**{k: v for k, v in metainfoGenerator_dict.items() if k != 'name'})

    def __len__(self):
        return int(len(self.imgs))

    def set_mode(self, mode):
        assert isinstance(mode, (str, ModelMode)), 'the type of mode should be str or ModelMode, given %s' % str(mode)

        if isinstance(mode, str):
            self.training = ModelMode.from_str(mode)
        else:
            self.training = mode

    def __getitem__(self, index):
        img_path, mask_path, mask_weak_path = self.imgs[index]
        img = Image.open(img_path).convert('L')  # .convert('RGB')
        mask = Image.open(mask_path)  # .convert('RGB')
        mask_weak = Image.open(mask_weak_path).convert('L')

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augment is not None and self.training == ModelMode.TRAIN:
            img, mask, mask_weak = self.augment(img, mask, mask_weak)

        self.transform = self.transform if self.transform is not None else default_transform
        img = self.transform['Img'](img)
        mask = self.transform['mask'](mask)
        mask_weak = self.transform['mask'](mask_weak)
        if getattr(self, 'metainGenerator'):
            meta_info = self.metainGenerator(mask)
            return [img, mask, mask_weak, img_path], meta_info

        return [img, mask, mask_weak, img_path], None


def id_(x):
    return x


A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", torch.Tensor, np.ndarray)


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageDataset, grp_regex, shuffle=False) -> None:
        imgs: List[str] = dataset.imgs
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

        print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename[0]).stem for filename in imgs]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(1) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(imgs)
        print(f"Found {len(unique_patients)} unique patients out of {len(imgs)} images")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        # print(self.idx_map)
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(imgs)

        print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)
