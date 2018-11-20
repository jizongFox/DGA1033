from abc import ABC, abstractmethod
from admm_research import flags, LOGGER
from admm_research.method import AdmmGCSize
from admm_research.utils import extract_from_big_dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from admm_research.method import ModelMode
import torch


class Base(ABC):
    trainer_hparam_keys = []

    def __init_subclass__(cls):
        """ Make sure every subclass have arch_hparam_keys """
        if not 'trainer_hparam_keys' in cls.__dict__:
            raise NotImplementedError(
                'Attribute \'trainer_hparam_keys\' has not been overriden in class \'{}\''.format(cls))

    @classmethod
    @abstractmethod
    def setup_arch_flags(cls):
        """ Setup the arch_hparams """
        pass

    @abstractmethod
    def start_training(self):
        pass

    @abstractmethod
    def _evaluate(self, dataloader):
        pass

    @abstractmethod
    def _main_loop(self, dataloader, epoch, mode):
        pass


class ADMM_Trainer(Base):
    lr_scheduler_hparam_keys = ['max_epoch', 'milestones', 'gamma']
    trainer_hparam_keys = ['device', 'printfreq', 'num_admm_innerloop', 'num_workers',
                           'batch_size'] + Base.trainer_hparam_keys

    @classmethod
    def setup_arch_flags(cls):
        flags.DEFINE_integer('max_epoch', default=100, help='number of max_epoch')
        flags.DEFINE_multi_integer('milestones', default=[50, 100, 150, 200], help='miletones for lr_decay')
        flags.DEFINE_float('gamma', default=0.2, help='gamma for lr_decay')
        flags.DEFINE_string('device', default='cpu', help='cpu or cuda?')
        flags.DEFINE_integer('printfreq', default=5, help='how many output for an epoch')
        flags.DEFINE_integer('num_admm_innerloop', default=2, help='how many output for an epoch')
        flags.DEFINE_integer('num_workers', default=1, help='how many output for an epoch')
        flags.DEFINE_integer('batch_size', default=1, help='how many output for an epoch')

    def __init__(self, ADMM_method: AdmmGCSize, datasets: list, criterion, hparams: dict) -> None:
        super().__init__()
        self.admm = ADMM_method
        self.criterion = criterion
        self.hparams = hparams
        lr_scheduler_hparams = extract_from_big_dict(hparams, ADMM_Trainer.lr_scheduler_hparam_keys)
        self.max_epoch = lr_scheduler_hparams['max_epoch']
        lr_scheduler_hparams.pop('max_epoch')
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.admm.optim, **lr_scheduler_hparams)
        self.device = torch.device(hparams['device'])
        self.admm.to(self.device)
        self.criterion.to(self.device)
        self.train_loader, self.val_loader = self._build_dataset(datasets, self.hparams)

    def start_training(self):
        for epoch in range(self.hparams['max_epoch']):
            self.lr_scheduler.step()
            self._main_loop(self.train_loader, epoch)
            with torch.no_grad():
                f_dice = self._evaluate(self.train_loader)
                LOGGER.info('At epoch {}, train acc is {:3f}%, under EVAL mode'.format(epoch, f_dice * 100))
                f_dice = self._evaluate(self.val_loader)
                LOGGER.info('At epoch {}, val acc is {:3f}%, under EVAL mode'.format(epoch, f_dice * 100))
            if epoch >= self.hparams['stop_dilation']:
                self.admm.is_dilation = False

    def _main_loop(self, dataloader, epoch, mode=ModelMode.TRAIN):
        dataloader.dataset.set_mode(mode)
        self.admm.set_mode(mode)

        for i, (img, gt, wgt, _) in tqdm(enumerate(dataloader)):
            if wgt.sum() == 0 and gt.sum() != 0:
                continue

            img, gt, wgt = img.to(self.device), gt.to(self.device), wgt.to(self.device)
            self.admm.reset(img)
            for j in range(self.hparams['num_admm_innerloop']):  #
                self.admm.update((img, gt, wgt), self.criterion)
                try:
                    self.admm.show('gamma', fig_num=2)
                except Exception as e:
                    print(e)
                try:
                    self.admm.show('s', fig_num=3)
                except Exception as e:
                    print(e)
        LOGGER.info('%s %d complete' % (mode.value, epoch))

    def _evaluate(self, dataloader):
        [_, fdice] = self.admm.evaluate(dataloader)
        return fdice

    @staticmethod
    def _build_dataset(datasets, hparams):
        train_set, val_set = datasets
        train_loader = DataLoader(train_set,
                                  num_workers=hparams['num_workers'],
                                  shuffle=True,
                                  batch_size=hparams['batch_size']
                                  )
        val_loader = DataLoader(val_set,
                                num_workers=hparams['num_workers'],
                                shuffle=False,
                                batch_size=hparams['batch_size']
                                )
        return train_loader, val_loader
