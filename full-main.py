# coding=utf8
import os
import sys
import warnings

sys.path.insert(-1, os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.criterion import CrossEntropyLoss2d, MSE_2D
import admm_research.dataset.medicalDataLoader as medicalDataLoader
from utils.enet import Enet
from tqdm import tqdm
from utils.utils import Colorize, dice_loss
from torchnet.meter import AverageValueMeter
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import click

torch.set_num_threads(2)
warnings.filterwarnings('ignore')
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
cuda_device = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

filename = os.path.basename(__file__).split('.')[0]

if not os.path.exists(os.path.join('results', filename)):
    os.mkdir(os.path.join('results', filename))

batch_size = 1
batch_size_val = 1
num_workers = 1
max_epoch = 400
data_dir = 'dataset/ACDC-2D-All'

color_transform = Colorize()
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

train_set = medicalDataLoader.MedicalImageDataset('train', data_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=True, equalize=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)

val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

val_iou_tables = []
train_iou_tables = []


def val(val_dataloader, network, save=False):
    # network.eval()
    dice_meter_b = AverageValueMeter()
    dice_meter_f = AverageValueMeter()

    dice_meter_b.reset()
    dice_meter_f.reset()
    images = []
    with torch.no_grad():
        for i, (image, mask, _, _) in enumerate(val_dataloader):
            if mask.sum() == 0:
                continue
            image, mask = image.to(device), mask.to(device)

            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            iou = dice_loss(predicted_mask, mask)

            dice_meter_f.add(iou[1])
            dice_meter_b.add(iou[0])

            if save:
                images = save_images(images, image, mask, proba[:, 1], predicted_mask)
    if save:
        grid = make_grid(images, nrow=4)
        return [[dice_meter_b.value()[0], dice_meter_f.value()[0]], grid]
    else:
        return [[dice_meter_b.value()[0], dice_meter_f.value()[0]], None]

    # network.train()


def save_images(images, img, mask, prob, segm):
    if len(images) >= 30 * 4:
        return images
    images.extend([img[0].float(), mask[0].float(), prob.float(), segm.float()])
    return images


@click.command()
@click.option('--lr', default=5e-4, help='learning rate')
@click.option('--loss_function', default='CE', type=click.Choice(['CE', 'MSE']))
def main(lr, loss_function):
    from datetime import datetime
    writer = SummaryWriter('log/' + str(lr) + '_' + str(loss_function) + '_' + datetime.now().strftime('%b%d_%H-%M-%S'))

    neural_net = Enet(2)
    neural_net.to(device)
    criterion = CrossEntropyLoss2d(weight=torch.Tensor([0.5, 2])).to(device) if loss_function == 'CE' else MSE_2D()
    optimizer = torch.optim.Adam(params=neural_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.25)
    highest_iou = -1

    plt.ion()
    for epoch in range(max_epoch):
        scheduler.step()
        for param_group in optimizer.param_groups:
            _lr = param_group['lr']
        for i, (img, full_mask, _, _) in tqdm(enumerate(train_loader)):
            if full_mask.sum() == 0: continue;
            img, full_mask = img.to(device), full_mask.to(device)
            optimizer.zero_grad()
            output = neural_net(img)
            loss = criterion(output, full_mask.squeeze(1))
            loss.backward()
            optimizer.step()

        ## evaluate the model:
        [train_ious, train_grid] = val(train_loader, neural_net, save=True)
        writer.add_scalars('data/train_dice', {'bdice': train_ious[0], 'fdice': train_ious[1]}, global_step=epoch)
        writer.add_image('train_grid', train_grid, epoch)
        train_ious.insert(0, _lr)
        train_iou_tables.append(train_ious)
        [val_ious, val_grid] = val(val_loader, neural_net, save=True)
        writer.add_scalars('data/test_dice', {'bdice': val_ious[0], 'fdice': val_ious[1]}, global_step=epoch)
        writer.add_image('val_grid', val_grid, epoch)
        val_ious.insert(0, _lr)
        val_iou_tables.append(val_ious)
        print(
            '%d epoch: training fiou is: %.5f and val fiou is %.5f, with learning rate of %.6f' % (
                epoch, train_ious[2], val_ious[2], _lr))
        try:
            pd.DataFrame(train_iou_tables, columns=['learning rate', 'background', 'foregound']).to_csv(
                'results/%s/train_lr_%f_%s.csv' % (filename, lr, loss_function))
            pd.DataFrame(val_iou_tables, columns=['learning rate', 'background', 'foregound']).to_csv(
                'results/%s/val_lr_%f_%s.csv' % (filename, lr, loss_function))

        except Exception as e:
            print(e)

        if val_ious[2] > highest_iou:
            print('The highest val fiou is %f' % val_ious[2])
            highest_iou = val_ious[2]
            try:
                torch.save(neural_net.state_dict(),
                           'full_checkpoint/pretrained_%.5f_%s.pth' % (val_ious[2], loss_function))
            except:
                os.mkdir('full_checkpoint')
                torch.save(neural_net.state_dict(),
                           'full_checkpoint/pretrained_%.5f_%s.pth' % (val_ious[2], loss_function))


if __name__ == "__main__":
    main()
