from admm_research.dataset import MedicalImageDataset, segment_transform, get_dataset_root
from torch.utils.data import DataLoader
import torch, pandas as pd
from admm_research.utils import tqdm_,class2one_hot
from admm_research.dataset import loader_interface
from admm_research.method.gc import _multiprocess_Call
from multiprocessing import Pool
import argparse, os
from pathlib import Path
import yaml
import numpy as np
from admm_research.metrics2 import DiceMeter
import easydict


def get_args():
    parser = argparse.ArgumentParser(description='user input')
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    return args


args = get_args()
with open('config.yaml') as f:
    config = yaml.load(f, )
config['Dataset']['dataset_name'] = args.name

train_loader, _ = loader_interface(config['Dataset'], config['Dataloader'])


def test_configuration(args):
    train_loader_ = enumerate(train_loader)
    diceMeter = DiceMeter(method='2d', report_axises=[1], C=2)
    for i, ((img, gt, wgt, path), _) in train_loader_:
        if gt.sum() == 0 or wgt.sum() == 0:
            continue
        results = _multiprocess_Call(
            imgs=img.squeeze(1).numpy(),
            scores=0.5*np.ones(shape=[img.shape[0], 2, img.shape[2], img.shape[3]]),
            us=np.zeros_like(img.squeeze(1)),
            gts=gt.numpy(),
            weak_gts=wgt.numpy(),
            lamda=args.lamda,
            sigma=args.sigma,
            kernelsize=args.kernal_size
        )
        diceMeter.add(class2one_hot(torch.Tensor(results),C=2).float(),gt)
    return diceMeter.summary()



def main(user_choice):
    report_results =[]
    sigmas = [0.001, 0.01, 0.02,  0.1, 10 ]
    kernal_sizes = [3, 5, 7]
    lamdas = [0, 0.1, 1, 2, 5, 10]
    config_list = []
    for s in sigmas:
        for k in kernal_sizes:
            for l in lamdas:
                    config_dicts = easydict.EasyDict({'kernal_size': k, 'lamda': l, "sigma": s})
                    config_list.append(config_dicts)
    for args_config in config_list:
        final_dice = test_configuration(args_config)
        report_results.append({**args_config,**final_dice})
        save_dir:Path =Path(f'parameter_search/{args.name}/results.csv')
        save_dir.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(report_results).to_csv(save_dir)


if __name__ == '__main__':
    main(get_args())
