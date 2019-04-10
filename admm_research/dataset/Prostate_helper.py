from .medicalDataLoader import MedicalImageDataset
from .medicalDataLoader import PatientSampler
from . import segment_transform
from . import get_dataset_root
from . import augment
from torch.utils.data import DataLoader

__all__ = ['PROSTATE_dataloader']


def build_datasets(dataset_name, use_data_aug, subfolder='WeaklyAnnotations', metainfoGenerator_dict={}):
    assert dataset_name in ('prostate', 'prostate_aug')
    root_dir = get_dataset_root(dataset_name)
    train_dataset = MedicalImageDataset(root_dir, 'train', subfolder=subfolder,
                                        transform=segment_transform((256, 256), mapping={0: 0, 1: 1, 255: 1}),
                                        augment=augment if use_data_aug else None,
                                        metainfoGenerator_dict=metainfoGenerator_dict)
    val_dataset = MedicalImageDataset(root_dir, 'val', subfolder=subfolder,
                                      transform=segment_transform((256, 256), mapping={0: 0, 1: 1, 255: 1}),
                                      augment=None,
                                      metainfoGenerator_dict=metainfoGenerator_dict)

    return train_dataset, val_dataset


def build_dataloader(train_set, val_set, num_workers, batch_size, shuffle=False, group_train=False):
    try:
        val_sampler = PatientSampler(val_set, "(Case\d+_\d+)_\d+", shuffle=shuffle)
    except AttributeError:
        val_sampler = PatientSampler(val_set, "\d+_(Case\d+_\d+)_\d+", shuffle=shuffle)

    train_loader = DataLoader(train_set,
                              num_workers=num_workers,
                              shuffle=True,
                              batch_size=batch_size
                              )

    if group_train:
        try:
            tra_sampler = PatientSampler(train_set, "(Case\d+_\d+)_\d+", shuffle=shuffle)
        except AttributeError:
            tra_sampler = PatientSampler(train_set, "\d+_(Case\d+_\d+)_\d+", shuffle=shuffle)
        train_loader = DataLoader(train_set,
                                  num_workers=num_workers,
                                  batch_size=1,
                                  batch_sampler=tra_sampler
                                  )

    val_loader = DataLoader(val_set,
                            num_workers=num_workers,
                            batch_sampler=val_sampler,
                            batch_size=1
                            )
    return train_loader, val_loader


def PROSTATE_dataloader(dataset_dict, dataloader_dict):
    # prostate_aug should not be coupled with group_train:
    if dataloader_dict.get('dataset_name') == 'prostate_aug':
        assert dataloader_dict.get('train_group', False) == False
    train_set, val_set = build_datasets(**dataset_dict)
    train_loader, val_loader = build_dataloader(train_set=train_set, val_set=val_set,
                                                **dataloader_dict)
    return train_loader, val_loader
