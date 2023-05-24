# coding=utf8

import os
import argparse
import dist_vis_parser


def main(args):
    dir_path = args.input
    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    f_arr = os.listdir(dir_path)
    for f in f_arr:
        if f.endswith(".sqlite"):
            f_path = os.path.join(dir_path, f)
            out_file = os.path.join(out_path, f.strip(".sqlite"))
            dist_vis_parser.main(f_path, out_file)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='the log path')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output excel file')
    t_args = parser.parse_args()
    main(t_args)

