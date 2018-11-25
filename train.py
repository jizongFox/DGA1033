from admm_research import LOGGER, flags, app, config_logger
from admm_research.dataset import MedicalImageDataset, segment_transform, augment, get_dataset_root
from admm_research.method import AdmmGCSize, get_method
from admm_research.loss import get_loss_fn
from admm_research.arch import get_arch
from admm_research.trainer import ADMM_Trainer
from admm_research.utils import extract_from_big_dict
import torch
import warnings

warnings.filterwarnings('ignore')
torch.set_num_threads(1)


def run(argv):
    del argv

    hparams = flags.FLAGS.flag_values_dict()

    root_dir = get_dataset_root(hparams['dataroot'])

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
    flags.DEFINE_string('dataroot', default='cardiac', help='the name of the dataset')
    AdmmGCSize.setup_arch_flags()
    ADMM_Trainer.setup_arch_flags()
    app.run(run)
