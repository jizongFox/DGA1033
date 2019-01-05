from admm_research.dataset import MedicalImageDataset, segment_transform, get_dataset_root
from torch.utils.data import DataLoader
import torch, pandas as pd
from admm_research.utils import graphcut_with_FG_seed_and_BG_dlation, AverageMeter, tqdm_
from multiprocessing import Pool
import argparse, os
from pathlib import Path


def mmp(function, args):
    return Pool().map(function, args)


class args:

    def __init__(self) -> None:
        super().__init__()
        self.name = 'cardiac'
        self.kernal_size = 5
        self.lamda = 10
        self.sigma = 0.01
        self.dilation_level = 10

    def update(cls, dict):
        for k, v in dict.items():
            setattr(cls, k, v)
        return cls


def build_datasets(dataset_name):
    root = get_dataset_root(dataset_name)
    trainset = MedicalImageDataset(root, 'train', transform=segment_transform((256, 256)), augment=None)
    valset = MedicalImageDataset(root, 'val', transform=segment_transform((256, 256)), augment=None)
    trainLoader = DataLoader(trainset, batch_size=1)
    valLoader = DataLoader(valset, batch_size=1)
    return trainLoader, valLoader


def test_one(args):
    device = torch.device('cpu')

    train_loader, val_loader = build_datasets(args.name)
    fd_meter = AverageMeter()
    train_loader_ = tqdm_(enumerate(train_loader))
    for i, (img, gt, wgt, path) in train_loader_:
        if gt.sum() == 0 or wgt.sum() == 0:
            continue
        img, gt, wgt = img.to(device), gt.to(device), wgt.to(device)
        [_, fd] = graphcut_with_FG_seed_and_BG_dlation(img.cpu().numpy().squeeze(), wgt.cpu().numpy().squeeze(),
                                                       gt.cpu().numpy().squeeze(), args.kernal_size, args.lamda,
                                                       args.sigma,
                                                       args.dilation_level)
        fd_meter.update(fd)
        train_loader_.set_postfix({'fd': fd_meter.avg})

    return {**{k: v for k, v in vars(args).items() if k.find('__') < 0}, **{'fd': fd_meter.avg}}


def main(user_choice):
    sigmas = [0.001, 0.01, 0.02, 0.03, 0.1]
    kernal_sizes = [3, 5, 7]
    lamdas = [0, 0.1, 1, 2, 5, 10]
    dilation_levels = [0, 3, 5, 7, 9, 12, 14]
    if user_choice.debug:
        sigmas = [0.001,]
        kernal_sizes = [3]
        lamdas = [0,]
        dilation_levels = [0,]

    config_list = []
    for s in sigmas:
        for k in kernal_sizes:
            for l in lamdas:
                for d in dilation_levels:
                    config_dicts = {'name': user_choice.name, 'kernal_size': k, 'lamda': l, "sigma": s,
                                    "dilation_level": d}
                    config_list.append(config_dicts)
    print(f'>> {config_list.__len__()} are found to test, the results are saved in {user_choice.output_dir}.')

    args_list = [args().update(d) for d in config_list]
    results = mmp(test_one, args_list)
    results = pd.DataFrame(results, columns=results[0].keys())
    results.index.name = 'opt'
    outdir = Path(user_choice.output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    results.to_csv(os.path.join(outdir.name, '%s.csv' % user_choice.name))


def input_args():
    parser = argparse.ArgumentParser(description='user input')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true', help='help with debug')

    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':
    main(input_args())
