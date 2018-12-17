# coding=utf-8
import os, sys
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import admm_research.dataset.medicalDataLoader as medicalDataLoader
from admm_research.dataset import segment_transform
from admm_research.arch import Enet
from tqdm import tqdm

sys.path.insert(-1, os.getcwd())
warnings.filterwarnings('ignore')

torch.set_num_threads(1)

filename = os.path.basename(__file__).split('.')[0]
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.001
max_epoch = 200
data_dir = '/Users/jizong/workspace/DGA1033/admm_research/dataset/ACDC-2D-All'
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
data_aug = False

train_set = medicalDataLoader.MedicalImageDataset(data_dir,'train', transform=segment_transform((200, 200)),augment=None)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = medicalDataLoader.MedicalImageDataset(data_dir,'val',transform=segment_transform((200, 200)),augment=None)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

##==================================================================================================================
neural_net = Enet(2)
torch.manual_seed(41)
neural_net.to(device)
from admm_research.method.ADMM_inequality import  ADMM

admm = ADMM(neural_net)

plt.ion()
for iteration in range(max_epoch):
    for j, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
        if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
            continue
        img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)
        admm.forward(img, full_mask, weak_mask.squeeze(1).long())
