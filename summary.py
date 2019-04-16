import argparse
from pprint import pprint
from typing import Union, Dict, Tuple

import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from admm_research import ModelMode
from admm_research.dataset import loader_interface
from admm_research.metrics2 import DiceMeter
from admm_research.models import Segmentator
from admm_research.utils import flatten_dict
from admm_research.utils import save_images

RESULT_FLAG="Final Results"


def get_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog='summary',
        usage='Generate acc/dice etc.',
        description='This summary.py generate masks and final dice scores.'
    )
    parser.add_argument('--folder', type=str, required=True,
                        help='File path containing the best.pth and yaml configuration.')
    parser.add_argument('--checkpoint_name', type=str, default='best.pth',
                        help='Checkpoint name to load in the folder (default: "best.pth").')
    parser.add_argument('--use_cpu', action='store_true', help='Force to use CPU even when cuda is available.')
    parser.add_argument('--run-by-cmd', action='store_true', help="Disable all display when running within CMD.")
    args: argparse.Namespace = parser.parse_args()
    print(args)
    return args


def read_config(config_path: Union[Path, str] = '') -> dict:
    """
    Return configuration dict given the input config_path
    :param config_path: type: Union[Path,str]
    :return dict
    """
    assert isinstance(config_path, (Path, str))
    if isinstance(config_path, str):
        config_path: Path = Path(config_path)  # type: ignore
    assert isinstance(config_path, Path)
    assert config_path.exists(), f"{str(config_path)} doesn't exist, check the input argument."
    # read yaml file.
    import yaml
    with open(config_path, 'r') as f:
        try:
            config: dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConnectionError(exc)
    return config


def main(args: argparse.Namespace) -> None:
    filepath: Path = Path(args.folder)
    assert filepath.exists()
    assert len(list(filepath.glob('*.yaml'))) == 1, f'The yaml config file should be unique, ' \
        f'given {list(filepath.glob("*.yaml"))}.'
    config_file_path: Path = list(filepath.glob('*.yaml'))[0]
    config_dict: dict = read_config(config_file_path)
    pprint(config_dict)
    print(f'-> Load val_dataloader')
    _, val_loader = loader_interface(config_dict['Dataset'], config_dict['Dataloader'])
    val_loader.dataset.set_mode('eval')
    checkpoint: dict = torch.load(str(filepath / args.checkpoint_name), map_location=torch.device('cpu'))
    print(f'-> Load networks')
    model = Segmentator(config_dict['Arch'], config_dict['Optim'], config_dict['Scheduler'])
    model = model.load_state_dict(checkpoint['ADMM']['model'])
    model.to(torch.device('cuda' if torch.cuda.is_available() and (not args.use_cpu) else 'cpu'))
    model.eval()
    print(f'Best score: {checkpoint["best"]:.3f}')
    print(f'-> Evaluating:')
    with torch.no_grad():
        dice: dict
        bdice: dict
        dice, bdice, *_ = evaluate_loop(model, val_loader, args)
    print(f'\n-> Folder "{args.folder}" Results (loaded: {checkpoint["best"]:.3f}):')
    results: str = ','.join([f'{k}:{v:.3f}' for k, v in dice.items()])
    print(f'2d DSC: {results}')
    results: str = ','.join([f'{k}:{v:.3f}' for k, v in bdice.items()])  # type:ignore
    print(f'3d DSC: {results}')
    print(f'{RESULT_FLAG}{results}')


def evaluate_loop(model: Segmentator, val_loader: DataLoader, args: argparse.Namespace) -> Tuple[
    Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    assert not model.training
    assert val_loader.dataset.training == ModelMode.EVAL
    dice_Meter = DiceMeter(method='2d', C=model.arch_params['num_classes'])
    bdice_Meter = DiceMeter(method='3d', C=model.arch_params['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() and (not args.use_cpu) else 'cpu')
    val_loader_ = tqdm(val_loader, leave=True, ncols=15)
    for i, ((img, gt, weak, path), _) in enumerate(val_loader_):
        img, gt = img.to(device), gt.to(device)
        pred = model.predict(img, logit=False)
        save_images(pred.max(1)[1], root=Path(args.folder), mode='best', iter=1000, names=[Path(x).stem for x in path])
        dice_Meter.add(pred, gt)
        bdice_Meter.add(pred, gt)
        val_loader_.set_postfix(flatten_dict({'': dice_Meter.summary(), 'b': bdice_Meter.summary()}, sep=''))
    return dice_Meter.detailed_summary(), bdice_Meter.detailed_summary(), dice_Meter.log, bdice_Meter.log


if __name__ == '__main__':
    main(get_parser())
