from admm_research import LOGGER, flags,app
from admm_research.dataset import MedicalImageDataset, segment_transform, augment
from admm_research.method import AdmmGCSize
from admm_research.loss import get_loss_fn
from admm_research.arch import get_arch
from admm_research.trainer import ADMM_Trainer
import torch


def run(argv):
    del argv
    hparams = flags.FLAGS.flag_values_dict()

    root_dir = 'admm_research/dataset/ACDC-2D-All'

    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((128, 128)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((128, 128)), augment=None)


    torchnet = get_arch('enet', **{'num_classes': 2})
    torchnet.load_state_dict(torch.load('/Users/jizong/workspace/DGA1033/checkpoints/weakly/enet_fdice_0.8906.pth',
                                        map_location=lambda storage, loc: storage))
    admm = AdmmGCSize(torchnet,hparams)
    weight = torch.Tensor([0, 1])
    criterion = get_loss_fn('cross_entropy', weight=weight)
    trainer = ADMM_Trainer(admm,[train_dataset,val_dataset],criterion,hparams)
    trainer.start_training()

if __name__ == '__main__':
    AdmmGCSize.setup_arch_flags()
    ADMM_Trainer.setup_arch_flags()
    app.run(run)
