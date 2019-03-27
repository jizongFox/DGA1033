import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot curves given folders, files, and column names')
    parser.add_argument('--folders', type=str, nargs='+', help='input folders', required=True)
    parser.add_argument('--file', type=str, required=True, help='csv name')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    pass


if __name__ == '__main__':
    main(get_args())
