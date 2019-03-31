import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

def get_args():
    parser = argparse.ArgumentParser(description='getting final results')
    parser.add_argument('--folder',type=str,required=True, help= 'folder path, only one folder')
    parser.add_argument('--file',type=str, required=True, help='csv to report.')

    return parser.parse_args()


def main(args):
    file_paths = list(Path(args.folder).glob(f'**/{args.file}'))
    print('Found file lists:')
    pprint(file_paths)



if __name__ == '__main__':
    main(get_args())

