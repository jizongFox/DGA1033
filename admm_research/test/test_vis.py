from admm_research.utils import Writter_tf
from admm_research.dataset import MedicalImageDataset,segment_transform,augment
from admm_research.arch import get_arch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def test_visualization():
    root_dir = '../dataset/ACDC-2D-All'
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((200, 200)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((200, 200)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    torchnet = get_arch('enet', {'num_classes': 2})
    writer = SummaryWriter()
    vis = Visualize_tf(writer)
    vis.show_imgs(train_loader)

if __name__ == '__main__':
    test_visualization()




