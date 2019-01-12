from admm_research.dataset import MedicalImageDataset, segment_transform, get_dataset_root
from torch.utils.data import DataLoader
import torch, pandas as pd
from admm_research.utils import graphcut_with_FG_seed_and_BG_dlation, AverageMeter, tqdm_
from multiprocessing import Pool
import argparse, os
from pathlib import Path
from skimage.io import imsave
from functools import partial
import warnings
from admm_research.utils import dice_loss_numpy


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


def build_datasets(dataset_name, foldername, equalize):
    root = get_dataset_root(dataset_name)
    trainset = MedicalImageDataset(root, 'train', transform=segment_transform((256, 256)), augment=None,
                                   foldername=foldername, equalize=equalize)
    trainLoader = DataLoader(trainset, batch_size=1)
    return trainLoader, None


def test_one(args, userchoice):
    print(f'>> args: {vars(args)}')
    args_name = '_'.join(['%s_%s' % (k, str(v)) for k, v in vars(args).items()])
    train_loader, _ = build_datasets(args.name, userchoice.folder_name, userchoice.equalize)
    fd_meter = AverageMeter()
    train_loader_ = tqdm_(enumerate(train_loader))
    for i, (img, gt, wgt, path) in train_loader_:
        gamma, [_, fd] = graphcut_with_FG_seed_and_BG_dlation(img.cpu().numpy().squeeze(), wgt.cpu().numpy().squeeze(),
                                                              gt.cpu().numpy().squeeze(), args.kernal_size, args.lamda,
                                                              args.sigma,
                                                              args.dilation_level)

        fd_meter.update(fd)

        save_img(gamma, userchoice.output_dir, args.name, userchoice.folder_name, args_name, path[0])

        train_loader_.set_postfix({'fd': fd_meter.avg})

    return {**{k: v for k, v in vars(args).items() if k.find('__') < 0}, **{'fd': fd_meter.avg, 'arg_name': args_name}}


def baseline(userchoice):
    device = torch.device('cpu')
    train_loader, _ = build_datasets(userchoice.name, userchoice.folder_name, userchoice.equalize)
    fd_meter = AverageMeter()
    train_loader_ = tqdm_(enumerate(train_loader))
    for i, (img, gt, wgt, path) in train_loader_:
        img, gt, wgt = img.to(device), gt.to(device), wgt.to(device)
        [_, df] = dice_loss_numpy(wgt.data.numpy(), gt.data.numpy())
        fd_meter.update(df)

    return fd_meter.avg


def save_img(gamma, output_dir, dataname, folder, args_name, path):
    save_name = Path(output_dir, dataname, folder, args_name, Path(path).name)
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        imsave(save_name, gamma)


def main(user_choice):
    sigmas = user_choice.sigmas
    kernal_sizes = user_choice.kernel_sizes
    lamdas = user_choice.lambdas
    dilation_levels = user_choice.dilation_levels
    if user_choice.debug:
        sigmas = [0.001, ]
        kernal_sizes = [5]
        lamdas = [1]
        dilation_levels = [5]

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
    test_one_ = partial(test_one, userchoice=user_choice)
    b_line = baseline(user_choice)
    # print('>> baseline:%.4f' % b_line)
    results = mmp(test_one_, args_list)

    results = [{**result, **{'baseline': b_line}} for result in results]

    results = pd.DataFrame(results, columns=results[0].keys())
    results.index.name = 'opt'
    outdir = Path(user_choice.output_dir, user_choice.name, user_choice.folder_name)
    outdir.mkdir(exist_ok=True, parents=True)
    results.to_csv(os.path.join(str(outdir), '%s.csv' % user_choice.name))
    return parse_results(os.path.join(str(outdir), '%s.csv' % user_choice.name))


def input_args():
    parser = argparse.ArgumentParser(description='user input')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--folder_name', type=str, default='WeaklyAnnotations')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true', help='help with debug')
    parser.add_argument('--sigmas', type=float, nargs='+', help='sigmal range', required=True)
    parser.add_argument('--lambdas', type=float, nargs='+', help='lambda range', required=True)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='lambda range', required=True)
    parser.add_argument('--dilation_levels', type=int, nargs='+', help='dilation_levels', required=True)
    parser.add_argument('--equalize', action='store_true')
    args_ = parser.parse_args()
    return args_


def parse_results(in_path):
    file = pd.read_csv(in_path, index_col=0)
    sorted_file = file.sort_values(by=['fd'], ascending=False)
    print(sorted_file.head(5))
    # return {k: list(v.values())[0] for k, v in sorted_file.head(1).to_dict().items()}


if __name__ == '__main__':
    received_args = input_args()
    main(received_args)
