# coding=utf8
from __future__ import print_function, division
import os, sys, random, re
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from admm_research.method import ModelMode
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional, TypeVar, Iterable
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
import torch, numpy as np

default_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')
        train_mask_weak_path = os.path.join(root, 'train', 'WeaklyAnnotations')

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

    elif mode == 'val':
        train_img_path = os.path.join(root, 'val', 'Img')
        train_mask_path = os.path.join(root, 'val', 'GT')
        train_mask_weak_path = os.path.join(root, 'val', 'WeaklyAnnotations')

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
    else:
        train_img_path = os.path.join(root, 'test', 'Img')
        train_mask_path = os.path.join(root, 'test', 'GT')
        train_mask_weak_path = os.path.join(root, 'test', 'WeaklyAnnotations')

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

    def __init__(self, root_dir, mode, transform=None, augment=None, equalize=False):
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
        self.imgs = make_dataset(root_dir, mode)
        self.augment = augment
        self.equalize = equalize
        self.training = ModelMode.TRAIN

    def __len__(self):
        return int(len(self.imgs)/5)

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
        mask = (mask >= 0.8).long()
        mask_weak = self.transform['mask'](mask_weak)
        mask_weak = (mask_weak >= 0.8).long()

        return [img, mask, mask_weak, img_path]

    def mask_pixelvalue2OneHot(self, mask):
        possible_pixel_values = [0.000000, 0.33333334, 0.66666669, 1.000000]
        mask_ = mask.clone()
        for i, p in enumerate(possible_pixel_values):
            mask_[(mask < p + 0.1) & (mask > p - 0.1)] = i
        mask_ = mask_.long()
        return mask_


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
