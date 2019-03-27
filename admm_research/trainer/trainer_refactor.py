from abc import ABC, abstractmethod
from pathlib import Path

import os
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from admm_research import LOGGER, config_logger
from admm_research import ModelMode
from admm_research.method.ADMM_refactor import AdmmBase
from admm_research.utils import tqdm_
from admm_research.metrics2 import DiceMeter, AverageValueMeter, AggragatedMeter


class Base(ABC):
    src = './runs'
    des = './archive'

    @abstractmethod
    def start_training(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, dataloader, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _main_loop(self, dataloader, epoch, mode, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def load_checkpoint(self, *args, **kwargs):
    #     raise NotImplementedError


class ADMM_Trainer(Base):

    def __init__(self, ADMM_method: AdmmBase, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 criterion: nn.Module, save_dir: str = 'tmp', max_epcoh: int = 3, checkpoint=None,
                 whole_config_dict=None) -> None:
        super().__init__()
        self.admm = ADMM_method
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.max_epoch = max_epcoh
        self.begin_epoch = 0
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        config_logger(self.save_dir)
        self.device = self.admm.device
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        if whole_config_dict:
            self.whole_config = whole_config_dict
            with open(self.save_dir / 'config.yaml', 'w') as f:
                yaml.dump(whole_config_dict, f, default_flow_styple=True)
        self.to(self.device)

    def to(self, device):
        self.admm.to(device)
        self.criterion.to(device)

    def schedulerstep(self):
        self.admm.model.schedulerStep()

    def start_training(self):
        ## define aggregate recorder.
        train_aggregate_dicemeter = AggragatedMeter()

        for epoch in range(self.begin_epoch, self.max_epoch):
            train_dice = self._main_loop(self.train_dataloader, epoch, mode=ModelMode.TRAIN)
            train_aggregate_dicemeter.add(train_dice)
            print(train_aggregate_dicemeter.summary())
            # with torch.no_grad():
            #     f_dice, thr_dice = self._evaluate(self.val_dataloader, mode='3Ddice')

        self.writer.cleanup()

    def _main_loop(self, dataloader, epoch, mode, *args, **kwargs):
        dataloader.dataset.set_mode(mode)
        self.admm.set_mode(mode)
        assert self.admm.model.training == True
        assert dataloader.dataset.training == ModelMode.TRAIN
        # define recorder for one epoch
        train_dice = DiceMeter(method='2d',report_axises=[1],C=2)

        for i, ((img, gt, wgt, _), size) in tqdm_(enumerate(dataloader)):
            img, gt, wgt = img.to(self.device), gt.to(self.device), wgt.to(self.device)
            self.admm.set_input(img, gt, wgt, size[:, :, 1])
            self.admm.update(self.criterion)
            train_dice.add(self.admm.score,gt)

        LOGGER.info('%s %d complete' % (mode.value, epoch))
        return train_dice.detailed_summary()


    def _evaluate(self, dataloader, *args, **kwargs):
        pass

    def checkpoint(self, dice, epoch):
        try:
            getattr(self, 'best_dice')
        except:
            self.best_dice = -1

        if dice >= self.best_dice:
            self.best_dice = dice
            dict = {}
            dict['model'] = self.admm.save_dict
            dict['epoch'] = epoch
            dict['dice'] = dice
            torch.save(dict, os.path.join(self.writer_name, 'best.pth'))
