# coding=utf8

import json
import seaborn as sns
import calc_helper
import pandas as pd
import matplotlib.pyplot as plt


sns.set()


def read_data(f_path):
    with open(f_path) as _in:
        obj = json.load(_in)
        return obj


def _calc_and_save(size_arr, out_path, title='disk_read'):
    res_dict = calc_helper.calc_batch(size_arr)
    _vis_and_save(res_dict, out_path, title)


def _vis_and_save(_dict, out_path, title):
    col_num = 1
    row_num = 2
    figs, axes = plt.subplots(row_num, col_num, figsize=(24, 24))
    figs.suptitle(title)
    df = pd.DataFrame.from_dict(_dict)
    f1_df = df[df['bandwidth(GB/s)'].isin([5, 20, 80, 160])]
    sns.barplot(ax=axes[0], data=f1_df, x='latency(ns)', y='time(ms)', hue='bandwidth(GB/s)')
    f2_df = df[df['latency(ns)'].isin([100, 200, 400, 800, 1600])]
    sns.barplot(ax=axes[1], data=f2_df, x='bandwidth(GB/s)', y='time(ms)', hue='latency(ns)')
    plt.savefig(out_path)


def main(in_path, out_path):
    data_dict = read_data(in_path)
    read_dict = data_dict['read']['bytes']
    _calc_and_save(read_dict, out_path + "_read.jpg", title='disk read')
    write_dict = data_dict.get('write')
    if write_dict is not None:
        _calc_and_save(write_dict['bytes'], out_path + "_write.jpg", title='disk write')


if __name__ == '__main__':
    import sys
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
