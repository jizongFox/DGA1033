from admm_research.method import AdmmSize
from admm_research.dataset import MedicalImageDataset, segment_transform, augment
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage


def test_dataloader():
    root_dir = '../dataset/ACDC-2D-All'
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((128, 128)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((128, 128)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # for i, (Img, GT, wgt, _) in enumerate(train_loader):
    #     ToPILImage()(Img[0]).show()
    #     if i == 5:
    #         train_loader.dataset.set_mode('eval')
    #     ToPILImage()(Img[0]).show()
    #     if i == 10:
    #         break
    #
    # for i, (img, gt, wgt, _) in enumerate(val_loader):
    #     ToPILImage()(img[0]).show()
    #     if i == 5:
    #         val_loader.dataset.set_mode('eval')
    #     ToPILImage()(img[0]).show()
    #     if i == 10:
    #         break
    assert train_dataset.__len__() == train_dataset.imgs.__len__()


def test_prostate_dataloader():
    root_dir = '../dataset/PROSTATE'
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((128, 128)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((128, 128)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # for i, (Img, GT, wgt, _) in enumerate(train_loader):
    #     ToPILImage()(Img[0]).show()
    #     if i == 5:
    #         train_loader.dataset.set_mode('eval')
    #     ToPILImage()(Img[0]).show()
    #     if i == 10:
    #         break
    #
    # for i, (img, gt, wgt, _) in enumerate(val_loader):
    #     ToPILImage()(img[0]).show()
    #     if i == 5:
    #         val_loader.dataset.set_mode('eval')
    #     ToPILImage()(img[0]).show()
    #     if i == 10:
    #         break
    assert train_dataset.__len__() == train_dataset.imgs.__len__()
