# coding=utf8

import json
import seaborn as sns
import calc_helper


sns.set()


def read_data(f_path):
    with open(f_path) as _in:
        obj = json.load(_in)
        return obj


def _calc_and_save(size_arr, out_path):
    res_dict = calc_helper.calc_batch(size_arr)
    with open(out_path, 'w') as _out:
        json.dump(res_dict, _out)


def main(in_path, out_path):
    data_dict = read_data(in_path)
    h2d_dict = data_dict['1']['bytes']
    d2h_dict = data_dict['2']['bytes']
    _calc_and_save(d2h_dict, out_path + "_d2h_")
    _calc_and_save(h2d_dict, out_path + "_h2d_")


if __name__ == '__main__':
    import sys
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
