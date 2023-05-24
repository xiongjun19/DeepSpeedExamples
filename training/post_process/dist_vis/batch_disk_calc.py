# coding=utf8

import os
import argparse
import calc_disk 
from tqdm import tqdm


def main(args):
    dir_path = args.input
    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    f_arr = os.listdir(dir_path)
    for f in tqdm(f_arr):
        if f.endswith(".org_data_trans.txt"):
            f_path = os.path.join(dir_path, f)
            out_f_name = f.split(".")[0]
            out_file = os.path.join(out_path, out_f_name)
            calc_disk.main(f_path, out_file)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='the log path')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output excel file')
    t_args = parser.parse_args()
    main(t_args)

