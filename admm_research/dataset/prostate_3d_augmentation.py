# This script is for augmenting the image, gt and prior so that
# it can be used in a static data augmentation way.

from pathlib import Path
from PIL import Image
from deepclustering.augment.sychronized_augment import SequentialWrapper
from torchvision.transforms import ToPILImage, RandomRotation, Compose, RandomHorizontalFlip

from admm_research.dataset import loader_interface

img_transform = Compose([
    ToPILImage(),
    RandomRotation(45),
    RandomHorizontalFlip()
])
target_transform = img_transform

sequential_transform = SequentialWrapper(img_transform, target_transform)

dataset_config = {
    'dataset_name': 'prostate',
    'subfolder': 'prior',
    'use_data_aug': False,
    'is_subfolder_mask': False,
    'metainfoGenerator_dict':
        {
            'name': 'IndividualBoundGenerator',
            'eps': 0.1
        }
}
dataloader_config = {
    'num_workers': 1,
    'batch_size': 1,
    'shuffle': True,
    'group_train': True
}
save_dir = Path('/home/jizong/Workspace/DGA1033/admm_research/dataset/PROSTATE-Aug-3D/train_')
save_dir.mkdir(exist_ok=True, parents=True)
save_img_dir = save_dir / 'Img'
save_gt_dir = save_dir / 'GT'
save_prior_dir = save_dir / 'prior'


def save_image(plt_img: Image, path: Path, name: str, randomseed):
    name_ = name.replace('_0_', f'_{randomseed}_')
    path.mkdir(exist_ok=True, parents=True)
    save_path: Path = path / name_
    plt_img.save(str(save_path) + '.png')


def save_images(imglist, path, name_list, randomseed=0):
    assert len(imglist) == len(name_list)
    for img, name in zip(imglist, name_list):
        save_image(img, path, name, randomseed)


train_loader, _ = loader_interface(dataconfig_dict=dataset_config, loader_config_dict=dataloader_config)

for randon_seed in (4,):
    for i, ((imgs, gts, priors, paths), _) in enumerate(train_loader):
        imgs_ = sequential_transform([x for x in imgs.squeeze()], if_is_target=[True for _ in range(imgs.__len__())],
                                     random_seed=randon_seed)

        save_images(imgs_, save_img_dir, [Path(x).stem for x in paths], randon_seed)

        priors_ = sequential_transform([x for x in priors.squeeze()],
                                       if_is_target=[True for _ in range(priors.__len__())],
                                       random_seed=randon_seed)

        save_images(priors_, save_prior_dir, [Path(x).stem for x in paths], randon_seed)
        gts_ = sequential_transform([x for x in gts.squeeze().float()],
                                    if_is_target=[True for _ in range(gts.__len__())],
                                    random_seed=randon_seed)
        save_images(gts_, save_gt_dir, [Path(x).stem for x in paths], randon_seed)
