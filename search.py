import argparse
import os
import random
import subprocess
import sys
import string
import logging

LOGGER = logging.getLogger('random_hparam')
LOGGER.setLevel(logging.INFO)
LOGGER.handlers = [logging.StreamHandler(stream=sys.stdout)]

FIX_HP = {
    'nb_ings': 4, 'nb_props': 4, 'effect_ratio': 0,
    'max_formula_ing': 4, 'max_select': 2, 'seller_range': 0,
    'alchem_nb_knowledge': 5, 'num_steps': 1500,
    'seller': 'simple_seq2seq', 'alchem': 'simple_seq2seq',
    'batch_size': 16, 'gamma': 0.9
}


HP_RANGES = {
    'reward_fn': ['better_best_init', 'better_last_accept', 'better_last_best'],
    'alchem_lr': (0.0003, 0.0003),
    'seller_lr': 'alchem_lr',
    'alchem_v_coeff': (0.1, 0.2),
    'seller_v_coeff': 'alchem_v_coeff',
    'alchem_ent_coeff': [0.001, 0.002, 0.003, 0.004, 0.005],
    'seller_ent_coeff': 'alchem_ent_coeff',
}



def generate_next_hparam(hparam_ranges):
    """ From the dictionary generate next hyparameter and save_dir """
    hparams = dict()
    while len(hparams) < len(hparam_ranges):
        for key, range in hparam_ranges.items():
            if key in hparams:
                continue

            if isinstance(range, list):
                hparams[key] = random.choice(range)
            elif isinstance(range, tuple):
                low, high = range[0], range[1]
                hparams[key] = random.uniform(low, high)
            elif isinstance(range, str):
                value = hparams.get(range, None)
                if value:
                    hparams[key] = value
    return hparams


def random_id():
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(8))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_dir', required=True, default=None,
                        help='Path to exp')
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    record_file_path = os.path.join(exp_dir, 'record.csv')
    first_line = None
    if not os.path.exists(record_file_path):
        first_line = ['exp_id'] + list(HP_RANGES.keys())
        first_line = '\t'.join(first_line)
    file_handler = logging.FileHandler(record_file_path)
    file_handler.setFormatter("")
    LOGGER.addHandler(file_handler)

    if first_line is not None:
        LOGGER.info(first_line)

    base_cmd = ['python', 'aichemist_research/emergent/train.py']
    while True:
        hparams = generate_next_hparam(HP_RANGES)
        exp_name = str(random_id())
        save_dir = os.path.join(exp_dir, exp_name)
        cmd = base_cmd + ['-save_dir', save_dir]
        for key, val in FIX_HP.items():
            cmd += ['-{}'.format(key), str(val)]
        for key, val in hparams.items():
            cmd += ['-{}'.format(key), str(val)]
        print(' '.join(cmd))

        exp_line = [exp_name] + [str(val) for val in hparams.values()]
        exp_line = '\t'.join(exp_line)
        LOGGER.info(exp_line)
        subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == '__main__':
    main(get_args())
