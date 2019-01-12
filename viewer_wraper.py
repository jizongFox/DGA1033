import sys, os
import argparse
from pathlib import Path
import pandas as pd
import subprocess
import copy


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='viewer wrapper to automatically parse the folder paths')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--n_columns', type=int, default=4)
    parser.add_argument('--img_source', type=str, required=True)
    args = parser.parse_args()
    # print(args)
    return args


def main(args: argparse.Namespace) -> str:
    csv_path: Path = Path(args.csv_path)
    assert csv_path.exists(), csv_path
    record = pd.read_csv(csv_path, index_col=0)
    sorted_record = record.sort_values(by=['fd'], ascending=False).head(args.n_columns)
    sorted_record_ = copy.deepcopy(sorted_record)
    del sorted_record_['arg_name']
    # print(sorted_record_)

    columns_paths = [Path(csv_path.parent, x) for x in sorted_record['arg_name']]
    for path in columns_paths:
        assert path.exists()
    columns_paths.reverse()
    columns_path_str = ' '.join([str(x) for x in columns_paths])
    # print(columns_path_str)
    img_source = args.img_source
    assert Path(img_source).exists()

    gt_path = img_source.replace('Img', 'GT')
    assert Path(gt_path).exists()
    folder_path = gt_path.replace('GT', csv_path.parents[0].name)
    assert Path(folder_path).exists()

    predefineColums_str = ' '.join([gt_path, folder_path])

    parameters = f'''{predefineColums_str} {columns_path_str}  --img_source={img_source}'''
    call_command = "python3.6 viewer.py %s" % parameters
    # subprocess.Popen(['/bin/zsh','-c',call_command])
    # subprocess.Popen(call_command, shell=True, executable='/bin/zsh')
    # subprocess.Popen(['/bin/bash', '-c', call_command])
    print(call_command)
    return call_command


if __name__ == '__main__':
    main(get_args())
