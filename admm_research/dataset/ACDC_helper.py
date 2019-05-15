from torch.utils.data import DataLoader

from . import augment
from . import get_dataset_root
from . import segment_transform
from .medicalDataLoader import MedicalImageDataset
from .medicalDataLoader import PatientSampler

__all__ = ['ACDC_dataloader']


def build_datasets(dataset_name, use_data_aug=False, subfolder='WeaklyAnnotations', metainfoGenerator_dict={},
                   choosen_class='LV', *args, **kwargs):
    assert dataset_name == 'cardiac'
    if choosen_class == 'LV':
        mapping = {0: 0, 85: 0, 170: 0, 255: 1}
    elif choosen_class == "RV":
        mapping = {0: 0, 85: 1, 170: 0, 255: 0}
    else:
        raise AttributeError(choosen_class)

    root_dir = get_dataset_root(dataset_name)
    train_dataset = MedicalImageDataset(root_dir, 'train', subfolder=subfolder,
                                        transform=segment_transform((256, 256), mapping=mapping),
                                        augment=augment if use_data_aug else None,
                                        metainfoGenerator_dict=metainfoGenerator_dict, *args, **kwargs)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((256, 256), mapping=mapping),
                                      augment=None,
                                      metainfoGenerator_dict=metainfoGenerator_dict, *args, **kwargs)

    return train_dataset, val_dataset


def build_dataloader(train_set, val_set, num_workers, batch_size, shuffle=True, group_train=False):
    val_sampler = PatientSampler(val_set, "(patient\d+_\d+)_\d+", shuffle=False)

    train_loader = DataLoader(train_set,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              batch_size=batch_size
                              )
    if group_train:
        tra_sampler = PatientSampler(train_set, "(patient\d+_\d+)_\d+", shuffle=True)
        train_loader = DataLoader(train_set,
                                  num_workers=num_workers,
                                  batch_size=1,
                                  batch_sampler=tra_sampler,
                                  )
    val_loader = DataLoader(val_set,
                            num_workers=num_workers,
                            batch_sampler=val_sampler,
                            batch_size=1
                            )
    return train_loader, val_loader


def ACDC_dataloader(dataset_dict, dataloader_dict):
    train_set, val_set = build_datasets(**dataset_dict)
    train_loader, val_loader = build_dataloader(train_set=train_set, val_set=val_set,
                                                **dataloader_dict)
    return train_loader, val_loader
