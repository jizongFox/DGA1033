import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
import matplotlib.pyplot as plt
from utils.criterion import CrossEntropyLoss2d
from utils.utils import dice_loss

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')


def val(val_dataloader, network):
    # network.eval()
    f_dice_meter = AverageValueMeter()
    f_dice_meter.reset()
    with torch.no_grad():
        for i, (image, mask, _, _) in enumerate(val_dataloader):
            if mask.sum()==0:
                continue
            image, mask = image.to(device), mask.to(device)
            proba = F.softmax(network(image), dim=1)
            predicted_mask = proba.max(1)[1]
            [_,fiou] = dice_loss(predicted_mask, mask)
            f_dice_meter.add(fiou)
            # plt.imshow(predicted_mask.squeeze())
            # plt.colorbar()
            # plt.show()
    # network.train()
    print('val iou:  %.8f' % f_dice_meter.value()[0])
    return f_dice_meter.value()[0]

def pretrain(train_dataloader, val_dataloader_, network, lr=5e-4, split_ratio=0.05,path=None):
    highest_iou = -1
    class config:
        lr = 1e-3
        epochs = 1000
        path = 'semi_pretrain_checkpoint'

    pretrain_config = config()
    pretrain_config.lr=lr

    if path:
        pretrain_config.path = path

    if not os.path.exists(pretrain_config.path):
        os.mkdir(pretrain_config.path)

    network.to(device)
    criterion_ = CrossEntropyLoss2d()
    optimiser_ = torch.optim.Adam(network.parameters(), pretrain_config.lr, weight_decay=1e-5)
    loss_meter = AverageValueMeter()

    fiou_tables = []

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser_,milestones=[200,400,600,800],gamma=0.25)
    for iteration in range(pretrain_config.epochs):
        loss_meter.reset()
        scheduler.step()

        for i, (img, mask, weak_mask, _) in tqdm(enumerate(train_dataloader)):
            img, mask = img.to(device), mask.to(device)
            optimiser_.zero_grad()
            output = network(img)
            loss = criterion_(output, mask.squeeze(1))
            loss.backward()
            optimiser_.step()
            loss_meter.add(loss.item())



        print('train_loss: %.6f' % loss_meter.value()[0])

        val_iou = val(val_dataloader_, network)
        fiou_tables.append(val_iou)
        if val_iou > highest_iou:
            highest_iou = val_iou
            torch.save(network.state_dict(),
                       os.path.join(pretrain_config.path, 'model_%.4f_split_%.3f.pth' % (val_iou, split_ratio)))
            print('pretrained model saved with %.4f.' % highest_iou)

    return fiou_tables