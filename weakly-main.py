# coding=utf8
import os
import sys
from torchvision.utils import save_image, make_grid
import pandas as pd

sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from ADMM import weakly_ADMM_network, weakly_ADMM_without_sizeConstraint, weakly_ADMM_without_gc
from utils.enet import Enet
from utils.network import UNet
from utils.utils import Colorize, evaluate_dice

from tqdm import tqdm
import click, sqlite3

torch.set_num_threads(1)

filename = os.path.basename(__file__).split('.')[0]
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

create_table_script = '''
CREATE TABLE IF NOT EXISTS dice_table(id INTEGER PRIMARY KEY, c_time text , 
net_arch TEXT, method_name TEXT, innerloop integer, lambda float , sigma float , kernel_size integer, eps float ,epoch integer, dataset TEXT, F_dice float,comment TEXT )
'''
db = sqlite3.connect('dataset/statistic_results')
try:
    db.execute(create_table_script)
    db.commit()
except Exception as e:
    print(e)
cursor = db.cursor()

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.001
max_epoch = 100
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
data_aug = False

train_set = medicalDataLoader.MedicalImageDataset('train', data_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=data_aug)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)


@click.command()
@click.option('--netarch', default='unet', type=click.Choice(['enet', 'unet']))
@click.option('--baseline', default='ADMM_weak', type=click.Choice(['ADMM_weak', 'ADMM_weak_gc', 'ADMM_weak_size']))
@click.option('--inneriter', default=1, help='iterative time in an inner admm loop')
@click.option('--lamda', default=1.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.01, help='sigma in the boundary term of the graphcut')
@click.option('--kernelsize', default=7, help='kernelsize of the graphcut')
@click.option('--assign_size_to_each', default=True, help='to apply individual loss')
@click.option('--eps', default=0.05, help='default eps for testing')
@click.option('--comments', default='test', type=click.Choice(['test','official']))
def main(netarch, baseline, inneriter, lamda, sigma, kernelsize, assign_size_to_each, eps, comments):

    best_val_score = -1
    ##==================================================================================================================
    if netarch == 'enet':
        neural_net = Enet(2)
    elif netarch == 'unet':
        neural_net = UNet(2)
    else:
        raise ValueError

    neural_net.to(device)
    if baseline == 'ADMM_weak':
        net = weakly_ADMM_network(neural_net, lr, sigma=sigma, lamda=lamda, assign_size_to_each=assign_size_to_each,
                                  eps=eps)
    elif baseline == 'ADMM_weak_gc':
        net = weakly_ADMM_without_sizeConstraint(neural_net, lr, lamda=lamda, sigma=sigma, kernelsize=kernelsize)
    elif baseline == 'ADMM_weak_size':
        net = weakly_ADMM_without_gc(neural_net, lr, assign_size_to_each=assign_size_to_each, eps=eps)
    else:
        raise ValueError

    plt.ion()
    for iteration in range(max_epoch):

        [train_ious, _] = evaluate_dice(train_loader, net.neural_net, save=False)
        [val_ious, _] = evaluate_dice(val_loader, net.neural_net, save=False)

        if best_val_score< val_ious[1]:
            # if the path does not exisit.
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
                if not os.path.exists(os.path.join('checkpoints','weakly')):
                    os.mkdir(os.path.join('checkpoints','weakly'))
            torch.save(neural_net.state_dict(),os.path.join('checkpoints','weakly','%s_fdice_%.4f.pth'%(netarch,val_ious[1])))
            best_val_score= val_ious[1]

        try:
            cursor.execute(
                '''INSERT INTO dice_table(c_time,net_arch,method_name,innerloop,lambda,sigma,kernel_size,eps,epoch,dataset,F_dice,comment) VALUES(DATETIME('now','localtime'),?,?,?,?,?,?,?,?,?,?,?)''',
                (netarch, baseline, inneriter, lamda, sigma, kernelsize, eps, iteration, 'train', train_ious[1],comments))
            cursor.execute(
                '''INSERT INTO dice_table(c_time,net_arch,method_name,innerloop,lambda,sigma,kernel_size,eps,epoch,dataset,F_dice,comment) VALUES(DATETIME('now','localtime'),?,?,?,?,?,?,?,?,?,?,?)''',
                (netarch, baseline, inneriter, lamda, sigma, kernelsize, eps, iteration, 'val', val_ious[1],comments))
            db.commit()

        except Exception as e:
            print(e)

        if iteration % 20 == 0:
            net.learning_rate_decay(0.95)

        for j, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
            if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
                continue
            img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)

            for i in range(inneriter):
                net.update_1((img, weak_mask), full_mask)
                net.update_2()
            net.reset()


if __name__ == "__main__":
    np.random.seed(1)
    torch.random.manual_seed(1)
    main()
