## TODO the script is to general final visual results for a given trained checkpoint.
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pathlib
from admm_research.arch import get_arch
from admm_research.dataset import get_dataset_root, MedicalImageDataset, augment, segment_transform
from admm_research.utils import tqdm_, dice_loss, AverageMeter, map_
from typing import List
from PIL import Image

resize = (384, 384)
margin = 50


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='draw final image mask')
    parser.add_argument('--checkpoint', required=True, type=str, default=None)
    parser.add_argument('--dataset', required=True, type=str, default='cardiac')
    parser.add_argument('--arch', required=True, type=str, help='name of the used architecture')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cpu', action='store_true', help='use cpu as backend')
    parser.add_argument('--save', action='store_true', help='save images')
    parser.add_argument('--show', action='store_true', help='show ground truth in the figure')
    parser.add_argument('--show_gt', action='store_true', help='show ground truth in the figure')
    parser.add_argument('--zoomin', action='store_true', help='show ground truth in the figure')

    args = parser.parse_args()
    print(args)
    return args


def inference(args: argparse.Namespace) -> None:
    ## load model
    assert args.dataset in ('cardiac', 'promise')
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f'Checkpoint given {args.checkpoint} does not exisit.'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ## report checkpoint:
    print(
        f">>>checkpoint {checkpoint_path} loaded. \n"
        f"Best epoch: {checkpoint['epoch']}, best val-2D dice: {round(checkpoint['dice'], 4)}")

    ## load model
    net: torch.nn.Module = get_arch(args.arch, {'num_classes': args.num_classes})
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    net.train()
    # net.eval()

    ## build dataloader
    root_dir = get_dataset_root(args.dataset)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((256, 256)), augment=None)
    val_loader = DataLoader(val_dataset, batch_size=1)

    val_loader = tqdm_(val_loader)
    dice_meter = AverageMeter()
    for i, (imgs, gts, wgts, paths) in enumerate(val_loader):
        imgs, gts, wgts = imgs.to(device), gts.to(device), wgts.to(device)
        pred_masks = net(imgs).max(1)[1]
        dice_meter.update(dice_loss(pred_masks, gts)[1], gts.shape[0])
        save_images(imgs, pred_masks, gts, paths, args)

        val_loader.set_postfix({'val 2d-dice': dice_meter.avg})
    print(f'\nrecalculated dice: {round(dice_meter.avg, 4)}')


def save_images(imgs: torch.Tensor, preds: torch.Tensor, gts: torch.Tensor, paths: List[str], args: argparse.Namespace):
    if args.save == False:
        return
    imgs, preds, gts = imgs.cpu(), preds.cpu(), gts.cpu()
    output_path: pathlib.PosixPath = Path(args.checkpoint).parents[0] / 'visualize'
    output_zi_path = None
    output_path.mkdir(exist_ok=True, parents=True)
    if args.zoomin:
        output_zi_path: pathlib.PosixPath = Path(args.checkpoint).parents[0] / 'visualize_zoomin'
        output_zi_path.mkdir(exist_ok=True, parents=True)

    for img, pred, gt, path in zip(imgs, preds, gts, paths):
        output_name = Path(path).name
        fig, fig_zoomin = save_image(img, pred, gt, args)
        fig.savefig(output_path / output_name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        if args.zoomin:
            fig_zoomin.savefig(output_zi_path / output_name, bbox_inches='tight', pad_inches=0)
            plt.close(fig_zoomin)


def save_image(img: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, args: argparse.Namespace):
    assert img.shape[0] == 1
    img = img.squeeze().numpy()
    pred = pred.squeeze().numpy()
    gt = gt.squeeze().numpy()

    fig = draw_figure(img, pred, gt, args)

    ## zoom_in
    if gt.sum() > 0:
        x_min, x_max = np.where(gt.sum(0) > 0)[0].min(), np.where(gt.sum(0) > 0)[0].max()
        y_min, y_max = np.where(gt.sum(1) > 0)[0].min(), np.where(gt.sum(1) > 0)[0].max()
        img = img[max(y_min - margin, 0):min(y_max + margin, len(img[1, :])),
              max(x_min - margin, 0):min(x_max + margin, len(img[:, 1]))]
        gt = gt[max(y_min - margin, 0):min(y_max + margin, len(gt[1, :])),
             max(x_min - margin, 0):min(x_max + margin, len(gt[:, 1]))]
        pred = pred[max(y_min - margin, 0):min(y_max + margin, len(pred[1, :])),
               max(x_min - margin, 0):min(x_max + margin, len(pred[:, 1]))]
        fig_zoomin = draw_figure(img, pred, gt, args)

        return fig, fig_zoomin
    else:
        return fig, fig


def draw_figure(img: np.ndarray, pred: np.ndarray, gt: np.ndarray, args: argparse.Namespace):
    img: np.ndarray = np.array(Image.fromarray(img * 255.0).resize(resize)) / 255.0
    gt: np.ndarray = np.array(Image.fromarray(gt * 255.0).resize(resize)) / 255.0
    pred: np.ndarray = np.array(Image.fromarray(pred * 255.0).resize(resize)) / 255.0

    # You cannot point out the number here.
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.contourf(pred, levels=[0.49, 0.5], colors='red')
    if args.show_gt:
        ax.contourf(gt, levels=[0.49, 0.5], colors='green')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    if args.show:
        plt.show()

    return fig


if __name__ == '__main__':
    inference(get_args())
