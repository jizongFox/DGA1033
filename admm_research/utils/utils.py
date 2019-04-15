from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchnet.meter import AverageValueMeter
from torchvision.utils import make_grid
from tqdm import tqdm

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

tqdm_ = partial(tqdm, ncols=25,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss_numpy(input, target):
    # with torch.no_grad:
    smooth = 1.
    iflat = input.reshape(input.shape[0], -1)
    tflat = target.reshape(input.shape[0], -1)
    intersection = (iflat * tflat).sum(1)

    foreground_iou = float(
        ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)).mean())

    iflat = 1 - iflat
    tflat = 1 - tflat
    intersection = (iflat * tflat).sum(1)
    background_iou = float(
        ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)).mean())

    return [background_iou, foreground_iou]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.shape[0], -1)
    tflat = target.view(input.shape[0], -1)
    intersection = (iflat * tflat).sum(1)

    foreground_dice = float(
        ((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())

    iflat = 1 - iflat
    tflat = 1 - tflat
    intersection = (iflat * tflat).sum(1)
    background_dice = float(
        ((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean())

    return [background_dice, foreground_dice]


def evaluate_dice(val_dataloader, network, save=False):
    b_dice_meter = AverageValueMeter()
    f_dice_meter = AverageValueMeter()
    network.eval()
    with torch.no_grad():
        images = []
        for i, (image, mask, weak_mask, pathname) in enumerate(val_dataloader):
            if mask.sum() == 0 or weak_mask.sum() == 0:
                continue
            image, mask, weak_mask = image.to(device), mask.to(device), weak_mask.to(device)
            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            [b_iou, f_iou] = dice_loss(predicted_mask, mask)
            b_dice_meter.add(b_iou)
            f_dice_meter.add(f_iou)
            if save:
                images = save_images(images, image, proba, mask, weak_mask)
    network.train()
    if save:
        grid = make_grid(images, nrow=4)
        return [[b_dice_meter.value()[0], f_dice_meter.value()[0]], grid]
    else:
        return [[b_dice_meter.value()[0], f_dice_meter.value()[0]], None]


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
        # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:

                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print(1)
        return color_image


def show_image_mask(*args):
    imgs = [x for x in args if type(x) != str]
    title = [x for x in args if type(x) == str]
    num = len(imgs)
    plt.figure()
    if len(title) >= 1:
        plt.title(title[0])

    for i in range(num):
        plt.subplot(1, num, i + 1)
        try:
            plt.imshow(imgs[i].cpu().data.numpy().squeeze())
        except:
            plt.imshow(imgs[i].squeeze())
    plt.tight_layout()
    plt.show()


# fns
from torch import Tensor, einsum
from functools import partial
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, Union, Optional, Dict, Any

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", a).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)

    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


import collections


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


import warnings
from pathlib import Path
from skimage.io import imsave


def save_images(segs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, iter: int, seg_num=None) -> None:
    '''save_images saves Tensor in 0-C setting to save in .png format.
    :param segs: 0-C Tensor with shape b,w,h
    :param names: names of b
    :param root: saved path
    :param mode: the string after root, usually as a mode
    :param iter: int interatition
    :param seg_num: default as None is there is no specified case
    :return: None
    '''
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            if seg_num is None:
                save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
            else:
                save_path = Path(root, f"iter{iter:03d}", mode, seg_num, name).with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy())


# argparser
import argparse
from functools import reduce
from copy import deepcopy as dcopy


def yaml_parser() -> dict:
    parser = argparse.ArgumentParser('Augment parser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])

    args: argparse.Namespace = parser.parse_args()  # type: ignore
    args_dict: dict = _parser(args.strings)  # type: ignore
    # pprint(args)
    return args_dict


def _parser(strings: List[str]) -> List[dict]:
    assert isinstance(strings, list)
    ## no doubled augments
    assert set(map_(lambda x: x.split('=')[0], strings)).__len__() == strings.__len__(), 'Augment doubly input.'
    args: List[Optional[Dict[Any, Any]]] = [_parser_(s) for s in strings]
    args = reduce(lambda x, y: dict_merge(x, y, True), args)
    return args


def _parser_(input_string: str) -> Optional[dict]:
    if input_string.__len__() == 0:
        return None
    assert input_string.find('=') > 0, f"Input args should include '=' to include the value"
    keys, value = input_string.split('=')[:-1][0].replace(' ', ''), input_string.split('=')[1].replace(' ', '')
    keys = keys.split('.')
    keys.reverse()
    for k in keys:
        d = {}
        d[k] = value
        value = dcopy(d)
    return dict(value)


## dictionary functions
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_merge(dct: dict, merge_dct: dict, re=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # dct = dcopy(dct)
    if merge_dct is None:
        if re:
            return dct
        else:
            return
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            try:
                dct[k] = type(dct[k])(eval(merge_dct[k])) if type(dct[k]) in (bool, list) else type(dct[k])(
                    merge_dct[k])
            except:
                dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)
