from abc import ABC, abstractmethod
from admm_research import LOGGER, config_logger
from admm_research.method import AdmmGCSize
from admm_research.utils import Writter_tf, tqdm_
from torch.utils.data import DataLoader
from admm_research import ModelMode
import torch, os, shutil, numpy as np, pandas as pd
from admm_research.dataset import PatientSampler
from pathlib import Path


class Base(ABC):
    trainer_hparam_keys = ['save_dir']
    src = './runs'
    des = './archive'

    def __init_subclass__(cls):
        """ Make sure every subclass have arch_hparam_keys """
        if not 'trainer_hparam_keys' in cls.__dict__:
            raise NotImplementedError(
                'Attribute \'trainer_hparam_keys\' has not been overriden in class \'{}\''.format(cls))

    @classmethod
    @abstractmethod
    def setup_arch_flags(cls):
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

    @abstractmethod
    def checkpoint(self, **kwargs):
        pass


class ADMM_Trainer(Base):
    lr_scheduler_hparam_keys = ['max_epoch', 'milestones', 'gamma']
    trainer_hparam_keys = ['device', 'printfreq', 'num_admm_innerloop', 'num_workers',
                           'batch_size', 'vis_during_training'] + Base.trainer_hparam_keys

    @classmethod
    def setup_arch_flags(cls):
        super().setup_arch_flags()
        flags.DEFINE_integer('max_epoch', default=200, help='number of max_epoch')
        flags.DEFINE_multi_integer('milestones', default=[20, 40, 60, 80, 100, 120, 140, 160],
                                   help='miletones for lr_decay')
        flags.DEFINE_float('gamma', default=0.9, help='gamma for lr_decay')
        flags.DEFINE_string('device', default='cpu', help='cpu or cuda?')
        flags.DEFINE_integer('printfreq', default=5, help='how many output for an epoch')
        flags.DEFINE_integer('num_admm_innerloop', default=2, help='how many output for an epoch')
        flags.DEFINE_integer('num_workers', default=1, help='how many output for an epoch')
        flags.DEFINE_integer('batch_size', default=1, help='how many output for an epoch')
        flags.DEFINE_boolean('vis_during_training', default=False, help='matplotlib plot image during training')

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
        if hparams['save_dir'] != 'None':
            self.writer_name = os.path.join(ADMM_Trainer.src, hparams['save_dir'])
            if Path(self.writer_name).exists():
                shutil.rmtree(self.writer_name)

        else:
            self.writer_name = os.path.join(ADMM_Trainer.src,
                                            self.generate_current_time() + '_' + self.generate_random_str())
        self.writer = Writter_tf(self.writer_name, self.admm.torchnet, num_img=30)
        config_logger(self.writer_name)
        self.save_hparams()

    def start_training(self):
        metrics = np.zeros((self.max_epoch, 3))

        LOGGER.info('begin training with max_epoch = %d' % self.hparams['max_epoch'])
        for epoch in range(self.hparams['max_epoch']):
            self.lr_scheduler.step()
            self._main_loop(self.train_loader, epoch)
            with torch.no_grad():
                f_dice, _ = self._evaluate(self.train_loader, mode='2Ddice')
                self.writer.add_scalar('train/2Ddice', f_dice, epoch)
                # self.writer.add_images(self.train_loader, epoch, device=self.device, opt='tensorboard', omit_image=True)
                LOGGER.info('At epoch {}, 2d train dice is {:.3f}%, under EVAL mode'.format(epoch, f_dice * 100))
                metrics[epoch, 0] = f_dice

                f_dice, thr_dice = self._evaluate(self.val_loader, mode='3Ddice')
                self.writer.add_scalar('val/2Ddice', f_dice, epoch)
                self.writer.add_scalar('val/3Ddice', thr_dice, epoch)
                self.writer.add_images(self.val_loader, epoch, device=self.device, opt='save', omit_image=False)
                LOGGER.info('At epoch {}, 2d val dice is {:.3f}%, under EVAL mode'.format(epoch, f_dice * 100))
                LOGGER.info('At epoch {}, 3d val dice is {:.3f}%, under EVAL mode'.format(epoch, thr_dice * 100))
                metrics[epoch, [1, 2]] = [f_dice, thr_dice]

            try:
                if epoch >= self.hparams['stop_dilation_epoch']:
                    self.admm.is_dilation = False
                    LOGGER.info('At epoch {}, Stop_dilation begins'.format(epoch))
            except:
                pass

            self.checkpoint(f_dice, epoch)
            pd.DataFrame(metrics, columns=['tra_2D_dice', 'val_2D_dice', 'val_3D_dice']).to_csv(
                os.path.join(self.writer_name, 'metrics.csv'), index_label='epoch')

        ## clean up
        self.writer.cleanup()

    def _main_loop(self, dataloader, epoch, mode=ModelMode.TRAIN):
        dataloader.dataset.set_mode(mode)
        self.admm.set_mode(mode)
        assert self.admm.torchnet.training == True
        assert dataloader.dataset.training == ModelMode.TRAIN

        for i, (img, gt, wgt, _) in tqdm_(enumerate(dataloader)):

            if self.hparams['ignore_negative']:
                if wgt.sum() <= 0 or gt.sum() <= 0:
                    continue

            img, gt, wgt = img.to(self.device), gt.to(self.device), wgt.to(self.device)
            self.admm.reset(img, gt, wgt)
            for j in range(self.hparams['num_admm_innerloop']):  #
                self.admm.update_1((img, gt, wgt))
                if self.hparams['vis_during_training']:
                    self.visualize_during_Training()
                self.admm.update_2(self.criterion)

        LOGGER.info('%s %d complete' % (mode.value, epoch))

    def _evaluate(self, dataloader, mode=ModelMode.EVAL):

        [_, fdice, threeD_dice] = self.admm.evaluate(dataloader, mode)
        return fdice, threeD_dice

    @staticmethod
    def _build_dataset(datasets, hparams):
        train_set, val_set = datasets

        if val_set.root_dir.find('ACDC') > 0:
            val_sampler = PatientSampler(val_set, "(patient\d+_\d+)_\d+", shuffle=False)
        else:
            val_sampler = PatientSampler(val_set, "(Case\d+_\d+)_\d+", shuffle=False)
        val_batch_size = 1
        train_loader = DataLoader(train_set,
                                  num_workers=hparams['num_workers'],
                                  shuffle=True,
                                  batch_size=hparams['batch_size']
                                  )
        val_loader = DataLoader(val_set,
                                num_workers=hparams['num_workers'],
                                batch_sampler=val_sampler,
                                batch_size=val_batch_size
                                )
        return train_loader, val_loader

    @staticmethod
    def generate_random_str(randomlength=16):
        """
        生成一个指定长度的随机字符串
        """
        import random
        random_str = ''
        base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
        length = len(base_str) - 1
        for i in range(randomlength):
            random_str += base_str[random.randint(0, length)]
        return random_str

    @staticmethod
    def generate_current_time():
        from time import strftime, localtime
        ctime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        return ctime

    def save_hparams(self):
        import pandas as pd
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(self.hparams.items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        file_name = os.path.join(self.writer_name, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        pd.Series(self.hparams).to_csv(os.path.join(self.writer_name, 'opt.csv'))

    def visualize_during_Training(self):
        try:
            self.admm.show('gamma', fig_num=2)
        except Exception as e:
            # print(e)
            pass
        try:
            self.admm.show('s', fig_num=3)
        except Exception as e:
            # print(e)
            pass
        try:
            self.admm.show('Y', fig_num=3)
        except Exception as e:
            # print(e)
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
