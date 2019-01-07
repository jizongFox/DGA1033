## TODO the script is to general final visual results for a given trained checkpoint.
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from admm_research.arch import get_arch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='draw final image mask')
    parser.add_argument('--checkpoont', required=True, type=str, default=None)
    parser.add_argument('--arch', required=True, type=str, help='name of the used architecture')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cpu', action='store_true', default='use cpu as backend')
    args = parser.parse_args()
    print(args)
    return args


def inference(args: argparse.Namespace) -> None:
    ## load model
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f'Checkpoint given {args.checkpoint} does not exisit.'

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    net: torch.nn.Module = get_arch(args.arch, args.num_classes)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    ## load data
