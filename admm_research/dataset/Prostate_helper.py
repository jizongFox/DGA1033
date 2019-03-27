from .medicalDataLoader import MedicalImageDataset
from .medicalDataLoader import PatientSampler
from . import segment_transform
from . import get_dataset_root
from . import augment
from torch.utils.data import DataLoader
__all__ = ['PROSTATE_dataloader']

def build_datasets(dataset_name, use_data_aug, metainfoGenerator_dict):
    assert dataset_name=='prostate'
    root_dir = get_dataset_root(dataset_name)
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((256, 256)),
                                        augment=augment if use_data_aug else None,metainfoGenerator_dict=metainfoGenerator_dict)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((256, 256)), augment=None,metainfoGenerator_dict=metainfoGenerator_dict)

    return train_dataset, val_dataset

def build_dataloader(train_set, val_set, num_workers, batch_size, shuffle=True):
    val_sampler = PatientSampler(val_set, "(Case\d+_\d+)_\d+", shuffle=shuffle)
    train_loader = DataLoader(train_set,
                              num_workers=num_workers,
                              shuffle=True,
                              batch_size=batch_size
                              )
    val_loader = DataLoader(val_set,
                            num_workers=num_workers,
                            batch_sampler=val_sampler,
                            batch_size=1
                            )
    return train_loader, val_loader

def PROSTATE_dataloader(dataset_dict, dataloader_dict):
    train_set, val_set = build_datasets(**dataset_dict)
    train_loader, val_loader = build_dataloader(train_set=train_set, val_set=val_set, **dataloader_dict)
    return train_loader, val_loader
