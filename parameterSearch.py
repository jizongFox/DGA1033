import argparse
import warnings
from pathlib import Path
from typing import Iterable, Union, Tuple

import easydict
import numpy as np
import pandas as pd
import torch
import yaml
from skimage.io import imsave

from admm_research.dataset import loader_interface
from admm_research.method.gc import _multiprocess_Call
from admm_research.metrics2 import DiceMeter
from admm_research.utils import class2one_hot, flatten_dict


def get_args():
    parser = argparse.ArgumentParser(description='user input')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--method', type=str, default='method3')
    args = parser.parse_args()
    return args


parser_args = get_args()
with open('config.yaml') as f:
    config = yaml.load(f, )
config['Dataset']['dataset_name'] = parser_args.name

train_loader, val_loader = loader_interface(config['Dataset'], config['Dataloader'], group_train=True)


def save_images(segs: np.ndarray, names: Iterable[str], root: Union[str, Path], ) -> None:
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            save_path = Path(root, name).with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg)


def test_configuration(args, parser_args):
    train_loader_ = enumerate(train_loader)
    diceMeter = DiceMeter(method='2d', report_axises=[1], C=2)
    bdiceMeter = DiceMeter(method='3d', report_axises=[1], C=2)
    for i, ((img, gt, wgt, path), bounds) in train_loader_:
        # if gt.sum() == 0 or wgt.sum() == 0:
        #     continue
        results: np.ndarray = _multiprocess_Call(
            imgs=img.squeeze(1).numpy(),
            scores=0.5 * np.ones(shape=[img.shape[0], 2, img.shape[2], img.shape[3]]),
            us=np.zeros_like(img.squeeze(1)),
            gts=gt.numpy(),
            weak_gts=wgt.numpy(),
            lamda=args.lamda,
            sigma=args.sigma,
            kernelsize=args.kernal_size,
            bounds=bounds[:, :, 1].numpy(),
            method=parser_args.method
        )
        diceMeter.add(class2one_hot(torch.Tensor(results), C=2).float(), gt)
        bdiceMeter.add(class2one_hot(torch.Tensor(results), C=2).float(), gt)
        # save_images(results, root=f'parameterSearch/{parser_args.name}/{str(args)}', names=[Path(p).stem for p in path])
    result = flatten_dict({'2d': diceMeter.summary(), '3d': bdiceMeter.summary()}, sep='')
    return result


def main():
    report_results = []
    # sigmas = [0.0001, 0.0005, 0.001]
    # kernal_sizes = [3, 5]
    # lamdas = [0.1, 0, 1, 10]
    sigmas = [0.005]
    kernal_sizes = [5]
    lamdas = [0.5]
    config_list = []
    for s in sigmas:
        for k in kernal_sizes:
            for l in lamdas:
                config_dicts = easydict.EasyDict({'kernal_size': k, 'lamda': l, "sigma": s})
                config_list.append(config_dicts)
    for args_config in config_list:
        final_dice = test_configuration(args_config, parser_args)
        report_results.append({**args_config, **final_dice})
        save_dir: Path = Path(f'parameterSearch/{parser_args.name}/{parser_args.method}_results.csv')
        save_dir.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(report_results).to_csv(save_dir)


if __name__ == '__main__':
    main()
