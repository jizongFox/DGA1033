import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import yaml
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import DataLoader

from admm_research import ModelMode
from admm_research import config_logger
from admm_research.method.ADMM_refactor import AdmmBase
from admm_research.metrics2 import DiceMeter, AggragatedMeter, ListAggregatedMeter
from admm_research.utils import tqdm_, flatten_dict, class2one_hot


class Base(ABC):
    src = './runs'
    des = './archive'

    @abstractmethod
    def start_training(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def _evaluate(self, dataloader, *args, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def _main_loop(self, dataloader, epoch, mode, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def load_checkpoint(self, *args, **kwargs):
    #     raise NotImplementedError


class ADMM_Trainer(Base):

    def __init__(self, ADMM_method: AdmmBase, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 criterion: nn.Module, save_dir: str = 'tmp', max_epoch: int = 3, checkpoint=None, use_tqdm=True,
                 whole_config_dict=None) -> None:
        super().__init__()
        self.admm = ADMM_method
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.begin_epoch = 0
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        config_logger(self.save_dir)
        self.device = self.admm.device

        self.checkpoint = None
        if checkpoint is not None:
            try:
                self.checkpoint = checkpoint
                self.load_checkpoint(checkpoint)
            except Exception as e:
                print(f'Loading checkpoint failed with {e}.')

        self.whole_config = None
        if whole_config_dict is not None:
            self.whole_config = whole_config_dict
            with open(self.save_dir / 'config_ACDC.yaml', 'w') as f:
                yaml.dump(whole_config_dict, f, )
        # this is for mac os system
        # self.admm.device = torch.device('cpu')
        self.device = self.admm.device

        self.to(self.device)
        self.use_tqdm = use_tqdm

    def to(self, device):
        self.admm.to(device)
        self.criterion.to(device)

    def schedulerstep(self):
        self.admm.model.schedulerStep()
        self.admm.step()

    def start_training(self):
        METERS = edict()
        METERS.tra_3d_dice = AggragatedMeter()
        METERS.tra_gc_dice = AggragatedMeter()
        METERS.tra_sz_dice = AggragatedMeter()
        METERS.val_2d_dice = AggragatedMeter()
        METERS.val_3d_dice = AggragatedMeter()
        wholeMeter = ListAggregatedMeter(names=list(METERS.keys()), listAggregatedMeter=list(METERS.values()))
        # try to load the saved meters
        if self.checkpoint is not None:
            try:
                wholeMeter.load_state_dict(
                    torch.load(Path(self.checkpoint) / 'last.pth', map_location=torch.device('cpu'))['meter'])
            except Exception as e:
                print(f'Loading meter historical record failed with {e}.')

        Path(self.save_dir, 'meters').mkdir(exist_ok=True)

        for epoch in range(self.begin_epoch + 1, self.max_epoch + 1):
            tra_3d_dice, tra_gc_dice, tra_sz_dice = self._main_loop(
                self.train_dataloader,
                epoch,
                mode=ModelMode.TRAIN
            )
            with torch.no_grad():
                val_2d_dice, val_3d_dice = self._eval_loop(
                    val_dataloader=self.val_dataloader,
                    epoch=epoch,
                    mode=ModelMode.EVAL
                )

            # save results:
            for k, v in METERS.items():
                v.add(eval(k))
            for k, v in METERS.items():
                v.summary().to_csv(Path(self.save_dir, 'meters', f'{k}.csv'))
            wholeMeter.summary().to_csv(Path(self.save_dir, f'wholeMeter.csv'))
            self.schedulerstep()
            self.save_checkpoint(dice=val_3d_dice.get('DSC1'), epoch=epoch, meters=wholeMeter)

    def summary(self):
        from summary import main as summary_main
        import argparse
        # proc = Popen(f'python summary.py --folder={self.save_dir}', shell=True, stdout=PIPE, stderr=PIPE)
        # out, err = proc.communicate()
        # proc.wait()
        # return  out.decode('utf-8')[out.decode('utf-8').find(RESULT_FLAG)+len(RESULT_FLAG):]
        results = summary_main(
            args=argparse.Namespace(**{'folder': self.save_dir,
                                       'checkpoint_name': 'best.pth',
                                       'use_cpu': False}
                                    )
        )
        return results

    def _main_loop(self, dataloader: DataLoader, epoch: int, mode: ModelMode, *args, **kwargs):
        dataloader.dataset.set_mode(mode)
        self.admm.set_mode(mode)
        assert self.admm.model.training == True
        assert dataloader.dataset.training == ModelMode.TRAIN
        # define recorder for one epoch
        train_dice = DiceMeter(method='3d', report_axises=[1], C=2)
        gc_dice = DiceMeter(method='3d', report_axises=[1], C=2)
        size_dice = DiceMeter(method='3d', report_axises=[1], C=2)
        dataloader_ = tqdm_(dataloader) if self.use_tqdm else dataloader

        for i, ((img, gt, wgt, path), size) in enumerate(dataloader_):
            img, gt, wgt = img.to(self.device), gt.to(self.device), wgt.to(self.device)
            self.admm.set_input(img, gt, wgt, size[:, :, 1], path)
            self.admm.update(self.criterion)
            train_dice.add(self.admm.score, gt)
            try:
                gc_dice.add(class2one_hot(torch.from_numpy(self.admm.gamma), C=2).float().to(self.device), gt)
            except AttributeError:
                pass
            try:
                size_dice.add(class2one_hot(self.admm.s.float(), C=2).float().to(self.device), gt)
            except AttributeError:
                pass
            if self.use_tqdm:
                report_dict = flatten_dict(
                    {'tra': train_dice.detailed_summary(), 'gc': gc_dice.summary(), 'sz': size_dice.summary()})
                dataloader_.set_postfix({k: v for k, v in report_dict.items() if v > 1e-6})
        if self.use_tqdm:
            report_dict: dict
            string_dict = f', '.join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
            print(f'Training   epoch: {epoch} -> {string_dict}')
        return train_dice.summary(), gc_dice.summary(), size_dice.summary()

    def _eval_loop(self, val_dataloader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL):
        val_dataloader.dataset.set_mode(mode)
        self.admm.set_mode(mode)
        assert self.admm.model.training == False
        assert val_dataloader.dataset.training == ModelMode.EVAL
        # define recorder for one epoch
        val_dice = DiceMeter(method='2d', report_axises=[1], C=2)
        val_bdice = DiceMeter(method='3d', report_axises=[1], C=2)
        val_dataloader_ = tqdm_(val_dataloader) if self.use_tqdm else val_dataloader
        for i, ((img, gt, wgt, _), size) in enumerate(val_dataloader_):
            img, gt, wgt = img.to(self.device), gt.to(self.device), wgt.to(self.device)
            pred = self.admm.model.predict(img, logit=False)
            val_dice.add(pred_logit=pred, gt=gt)
            val_bdice.add(pred, gt)
            if self.use_tqdm:
                report_dict = flatten_dict({'': val_dice.summary(), 'b': val_bdice.summary()}, sep='')
                val_dataloader_.set_postfix(report_dict)
        if self.use_tqdm:
            report_dict = flatten_dict({'': val_dice.summary(), 'b': val_bdice.summary()}, sep='')
            string_dict = f', '.join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
            print(f'Validating epoch: {epoch} -> {string_dict}')
        return val_dice.summary(), val_bdice.summary()

    def save_checkpoint(self, dice, epoch, meters):
        try:
            getattr(self, 'best_dice')
        except:
            self.best_dice = -1
        save_best = False
        if dice >= self.best_dice:
            self.best_dice = dice
            save_best = True

        dict = {}
        # dict['model'] = self.admm.model.state_dict
        dict['epoch'] = epoch
        dict['meter'] = meters.state_dict
        dict['ADMM'] = self.admm.state_dict
        dict['best'] = self.best_dice
        torch.save(dict, os.path.join(self.save_dir, 'last.pth'))
        if save_best:
            torch.save(dict, os.path.join(self.save_dir, 'best.pth'))

    # todo modify
    def load_checkpoint(self, checkpoint):

        state_dict = torch.load(Path(checkpoint) / 'last.pth', map_location=torch.device('cpu'))

        self.admm.load_state_dict(state_dict['ADMM'])

        self.begin_epoch = state_dict['epoch']
        self.best_dice = state_dict['best']
        print(f'loaded checkpoint: {checkpoint} '
              f'Best score:{self.best_dice:.3f} with run epoch: {self.begin_epoch}')
