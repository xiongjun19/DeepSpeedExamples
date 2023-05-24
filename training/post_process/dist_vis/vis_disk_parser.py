# coding=utf8

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set()


def read_data(f_path):
    with open(f_path) as _in:
        obj = json.load(_in)
        return obj


def plot_hist(_dict, out_path, key, factor=4):
    df = pd.DataFrame.from_dict(_dict)
    out_path = out_path + str(key) + ".jpg"
    _max_val = max(df[key])
    min_val = min(df[key])
    bins_arr = [max(min_val-1, 0)]
    t_val = factor * min_val
    while t_val < _max_val:
        bins_arr.append(t_val)
        t_val = factor * t_val
    bins_arr.append(_max_val+1)
    new_data = pd.cut(df[key], bins=bins_arr).value_counts(normalize=True)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=new_data.index, y=new_data.values)
    plt.savefig(out_path)


def main(in_path, out_path):
    data_dict = read_data(in_path)
    read_dict = data_dict['read']
    plot_hist(read_dict, out_path + "_read_", 'bytes')
    write_dict = data_dict.get('write')
    if write_dict is not None:
        plot_hist(write_dict, out_path + "_write_", 'bytes')


if __name__ == '__main__':
    import sys
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
