from admm_research import LOGGER, flags, app
from admm_research.dataset import MedicalImageDataset, segment_transform, augment
from admm_research.method import AdmmGCSize, get_method
from admm_research.loss import get_loss_fn
from admm_research.arch import get_arch
from admm_research.trainer import ADMM_Trainer
from admm_research.utils import extract_from_big_dict
import torch

torch.set_num_threads(1)


def run(argv):
    del argv

    hparams = flags.FLAGS.flag_values_dict()
    print(hparams)
    root_dir = 'admm_research/dataset/ACDC-2D-All'

    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((200, 200)), augment=None)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((200, 200)), augment=None)

    arch_hparams = extract_from_big_dict(hparams, AdmmGCSize.arch_hparam_keys)
    torchnet = get_arch(arch_hparams['arch'], arch_hparams)
    # torchnet.load_state_dict(torch.load('checkpoints/weakly/enet_fdice_0.8906.pth',map_location=lambda storage, loc: storage))
    admm = get_method(hparams['method'], torchnet, **hparams)
    criterion = get_loss_fn('partial_ce')
    trainer = ADMM_Trainer(admm, [train_dataset, val_dataset], criterion, hparams)
    trainer.start_training()


if __name__ == '__main__':
    AdmmGCSize.setup_arch_flags()
    ADMM_Trainer.setup_arch_flags()
    app.run(run)
