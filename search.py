import argparse
import random
import string
from pprint import pprint
from typing import *

import numpy as np
import pandas as pd
import yaml
from pathlib2 import Path

from admm_research.utils import dict_merge, string_parser_

RUN_HP_RANGES = {
    'ADMM_Method.ADMMLoopNum': {1, 2},
    'ADMM_Method.OptimInnerLoopNum': {1, 2},
    'ADMM_Method.balance_scheduler_dict.begin_epoch': {0, 20},
    'ADMM_Method.balance_scheduler_dict.max_value': [0.5, 0.75],
    'Optim.lr': {0.0001, 0.0005, 0.001, 0.002},
    'ADMM_Method.lamda': {100},
    'ADMM_Method.sigma': {0.0001},
    'ADMM_Method.kernel_size': {3},
}

GC_HP_RANGES = {
    'Arch.name': 'dummy',
    'ADMM_Method.ADMMLoopNum': {1},
    'ADMM_Method.OptimInnerLoopNum': {0},
    'ADMM_Method.lamda': {0, 10, 10, 100, 0.1},
    'ADMM_Method.sigma': {0.00001, 0.0001, 0.001, 0.01},
    'ADMM_Method.kernel_size': {3, 5},
    'ADMM_Method.gc_use_prior': {True, False},
    'Optim.lr': {0},
    'Trainer.max_epoch': {1}
}

ADMM_HP_RANGES = {
    'ADMM_Method.ADMMLoopNum': {1, 2},
    'ADMM_Method.OptimInnerLoopNum': {1, 2, 4},
    'ADMM_Method.kernel_size': {3},
    'Optim.lr': {0.00005, 0.0001, 0.001, },
    'Trainer.max_epoch': {200},
    'ADMM_Method.p_v': {0},
    'Scheduler.step_size': {50}
}


def generate_next_hparam(hp_range, sample_time=100) -> Generator:
    def sequential_choose(one_parameter_dict: dict):
        k = list(one_parameter_dict.keys())[0]
        v = one_parameter_dict.get(k)
        assert isinstance(v, (set, tuple))
        if isinstance(v, set):
            return {f'{k}': f'{np.random.choice(list(v))}'}
        elif isinstance(v, tuple):
            return {f'{k}': f'{np.random.choice(v)}'}

    def random_choose(one_parameter_dict: dict):
        k = list(one_parameter_dict.keys())[0]
        v = one_parameter_dict.get(k)
        assert isinstance(v, list)
        assert v.__len__() == 2, v.__len__()
        assert v[0] < v[1], f'{v[0]} should be lower than {v[1]}'
        return {f'{k}': f'{np.random.uniform(v[0], v[1], 1)[0]}'}

    def choose(one_parameter_dict: dict):
        k = list(one_parameter_dict.keys())[0]
        v = one_parameter_dict.get(k)

        assert isinstance(v, (list, tuple, set, str))
        if isinstance(v, list):
            return random_choose(one_parameter_dict)
        elif isinstance(v, (tuple, set)):
            return sequential_choose(one_parameter_dict)
        elif isinstance(v, str):
            return {f'{k}': f'{v}'}

    for _ in range(sample_time):
        results: dict = {}
        for k, v in hp_range.items():
            results = dict_merge(results, choose({k: v}), re=True)
        yield results


def parse_hyparm(string_dict: Dict[str, Any]) -> dict:
    assert isinstance(string_dict, dict)
    result = {}
    for k, v in string_dict.items():
        parsed_dict = string_parser_(f'{k}={v}')
        result = dict_merge(result, parsed_dict, re=True)
    return result


def random_save_dir(exp_path: str) -> dict:
    return {'Trainer': {'save_dir': exp_path + '/' + ''.join(random.choice(string.ascii_uppercase + string.digits)
                                                             for _ in range(8))
                        }
            }


def search(args: argparse.Namespace, HP_RANGES: dict) -> None:
    from main import main as main_function
    hp_generator = generate_next_hparam(HP_RANGES, sample_time=args.sample_time)
    save_dir: Path = Path(args.exp_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    search_results = []

    while True:
        try:
            hp_config = hp_generator.__next__()
            hp_config_dict: dict = parse_hyparm(hp_config)
            save_dir: dict = random_save_dir(args.exp_dir)  # type: ignore
            merged_dict = dict_merge(dict_merge(BASE_CONFIG, hp_config_dict, re=True), save_dir, True)
            summary_result, whole_results = main_function(merged_dict)
            if args.method == 'gc':
                search_results.append(
                    {**hp_config, **{'save_dir': save_dir['Trainer']['save_dir']},
                     **{'gc': whole_results.summary()['tra_gc_dice_DSC1'][0]},
                     **summary_result}
                )
            else:
                search_results.append(
                    {**hp_config, **save_dir, **summary_result}
                )

            pd.DataFrame(search_results).to_csv(Path(args.exp_dir) / 'Prostate_search_result.csv')
        except StopIteration:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Config', required=True, type=str, help="Default yaml file path.")
    parser.add_argument('--exp_dir', '-d', required=True, default=None,
                        help='Path to exp')
    parser.add_argument('--method', '-m', default='run', type=str, help="Evaluating GC hyperparamter.")
    parser.add_argument('--sample_time', '-t', type=int, default=10, help='Sample time, default = 5.')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    with open(args.Config, 'r') as stream:
        BASE_CONFIG = yaml.safe_load(stream)
    print('->Base configuration:')
    pprint(BASE_CONFIG)

    if args.method == 'run':
        HP_RANGES = RUN_HP_RANGES
    elif args.method == 'gc':
        HP_RANGES = GC_HP_RANGES
    elif args.method == 'admm':
        HP_RANGES = ADMM_HP_RANGES
    else:
        raise NotImplementedError(f'{args.method} not implemented.')
    search(args, HP_RANGES)

# for GC_search with RV,
# `python search.py --method=gc -t=40 -d runs/RV_prior/GC_search`
