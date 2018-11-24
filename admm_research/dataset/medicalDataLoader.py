# coding=utf8
from __future__ import print_function, division
import os, sys
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from admm_research.method import ModelMode

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
        img = self.transform['img'](img)
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
