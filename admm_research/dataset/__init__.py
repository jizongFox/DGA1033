import random

from PIL import ImageOps
from torchvision import transforms
import torch
from torch import nn
import numpy as np
from admm_research.utils import Colorize
from .medicalDataLoader import MedicalImageDataset, PatientSampler

color_transform = Colorize()

dataset_root = {}


def _registre_data_root(name, root, alis=None):
    if name in dataset_root.keys():
        raise ('The {} has been taken in the dictionary.'.format(name))
    dataset_root[name] = root
    if alis is not None and alis not in dataset_root.keys():
        dataset_root[alis] = root


class ToLabel(nn.Module):

    def __init__(self, mapping=None):
        super().__init__()
        self.mapping = mapping

    def forward(self, input):
        input_np = np.array(input).astype(np.float32)
        if self.mapping is not None:
            input_np = np.vectorize(self.mapping.get)(input_np)
        input_torch = torch.from_numpy(input_np).long()
        return input_torch


def segment_transform(size,mapping=None):
    from PIL import Image
    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(size,interpolation=Image.NEAREST),
        # transforms.ToTensor()
        ToLabel(mapping)
    ])
    prior_transform = transforms.Compose([
        transforms.Resize(size,interpolation=Image.NEAREST),
        transforms.ToTensor()
        # ToLabel(mapping)
    ])
    return {'Img': img_transform,
            'mask': mask_transform,
            'prior':prior_transform}


def augment(img, mask, weak_mask):
    if random.random() > 0.5:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)
        weak_mask = ImageOps.flip(weak_mask)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)
        weak_mask = ImageOps.mirror(weak_mask)
    if random.random() > 0.5:
        angle = random.random() * 90 - 45
        img = img.rotate(angle)
        mask = mask.rotate(angle)
        weak_mask = weak_mask.rotate(angle)
    # if random.random() > 0.8:
    #     (w, h) = img.size
    #     (w_, h_) = mask.size
    #     assert (w == w_ and h == h_), 'The size should be the same.'
    #     crop = random.uniform(0.85, 0.95)
    #     W = int(crop * w)
    #     H = int(crop * h)
    #     start_x = w - W
    #     start_y = h - H
    #     x_pos = int(random.uniform(0, start_x))
    #     y_pos = int(random.uniform(0, start_y))
    #     img = img.crop((x_pos, y_pos, x_pos + W, y_pos + H))
    #     mask = mask.crop((x_pos, y_pos, x_pos + W, y_pos + H))
    #     weak_mask = weak_mask.crop((x_pos, y_pos, x_pos + W, y_pos + H))

    return img, mask, weak_mask


_registre_data_root('ACDC_2D', './admm_research/dataset/ACDC-2D-All', 'cardiac')
_registre_data_root('PROSTATE', './admm_research/dataset/PROSTATE', 'prostate')
_registre_data_root('PROSTATE-Aug', './admm_research/dataset/PROSTATE-Aug', 'prostate_aug')


def get_dataset_root(dataname):
    if dataname in dataset_root.keys():
        return dataset_root[dataname]
    else:
        raise ('There is no such dataname, given {}'.format(dataname))


def loader_interface(dataconfig_dict, loader_config_dict):
    assert dataconfig_dict['dataset_name'] in ('cardiac', 'prostate', 'prostate_aug'),dataconfig_dict['dataset_name']
    if dataconfig_dict['dataset_name'] == 'cardiac':
        from .ACDC_helper import ACDC_dataloader
        return ACDC_dataloader(dataconfig_dict, loader_config_dict,)
    elif dataconfig_dict['dataset_name'] in ('prostate','prostate_aug'):
        from .Prostate_helper import PROSTATE_dataloader
        return PROSTATE_dataloader(dataconfig_dict, loader_config_dict )
    else:
        raise NotImplementedError
