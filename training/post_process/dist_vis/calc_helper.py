# coding=utf8

latency_arr = [(i + 1) * 100 for i in range(40)]  # ns
bandwidth_arr = [(i + 1) * 5 for i in range(40)]  # GB/s


def calc_time(size, latency, bandwidth):
    # size in KB
    # latency in ns
    # bandwidth in GB /s
    res = size / (bandwidth * (1024 ** 2)) * (10 ** 9) + latency
    return res / (10 ** 6)  # convert to ms


def calc_batch(size_arr):
    res = {
            'latency(ns)': [],
            'bandwidth(GB/s)': [],
            'time(ms)': [],
            }
    for lat in latency_arr:
        for bandwidth in bandwidth_arr:
            tmp_sum = 0.
            for size in size_arr:
                tmp_sum += calc_time(size, lat, bandwidth)
            res['latency(ns)'].append(lat)
            res['bandwidth(GB/s)'].append(bandwidth)
            res['time(ms)'].append(tmp_sum)
    return res
