import os
# from utils import save_csv, read_csv
import numpy as np
import math
from multiprocessing import Pool, freeze_support, cpu_count
import os


def one_chip(file_name, start_line, end_line, chip_num, chip_idx, chip_size):
    print(chip_idx)
    lines = os.popen('sed -n {:d},{:d}p {:s}'.format(start_line,
                     end_line, file_name)).read().split('\n')
    if lines[-1] == '':
        lines = lines[0:-1]
    matrix = []
    assert len(lines) == end_line - start_line + 1
    assert (start_line - 1) % chip_size == 0
    assert chip_idx == (start_line - 1) // chip_size
    for i in range(len(lines)):
        matrix.append(list(map(int, lines[i].split(','))))
    res = np.zeros((1, chip_num), dtype=np.int32)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            j_idx = j // chip_size
            if j_idx == chip_idx:
                continue
            res[0, j_idx] += matrix[i][j]
    print('///', chip_idx, 'Done !')
    return start_line // chip_size, res


def one_chip_new(file_name, chip_num, chip_idx, chip_size):
    res = np.zeros((1, chip_num), dtype=np.int32)
    matrix = np.loadtxt(file_name, dtype=np.int32, comments='//', delimiter=',').sum(0)
    for j in range(chip_num):
        if j == chip_idx:
            continue
        j_s, j_e = j * chip_size, (j+1) * chip_size
        res[0, j] += matrix[j_s:j_e].sum()
    print('///', chip_idx, 'Done !')
    return chip_idx, res


def main_mt():
    file_name = '/data1/dataset/sch/data/sch/gpt_lut/'
    with open(file_name + 'name.log', 'r') as nf:
        all_name = nf.readlines()
    core_num = 217008
    chip_size = 160
    # chip_num = 1400
    chip_num = math.ceil(core_num / chip_size)
    res = []
    freeze_support()
    # pool = Pool(1)
    pool = Pool(cpu_count() * 1 // 2)
    for idx, name in enumerate(all_name):
        if name[-1:] == '\n':
            name = name[:-1] 
        pool.apply_async(func=one_chip_new,
                            args=(file_name + name, chip_num, idx, chip_size,),
                            callback=res.append)
    pool.close()
    pool.join()
    pool.terminate()
    res = sorted(res)
    new_lut = np.concatenate([item[1] for item in res])
    np.savetxt('./gpt_xl_t1000_processed.csv', new_lut, fmt='%d', delimiter=',') #39145
    print('***** Save Done !')


def main():
    # os.system('unzip -d /data/sch/gpt_lut/ /data/sch/gpt_lut/gpt_xl_t1000.zip')
    # origin_csv = np.loadtxt('/data1/dataset/sch/data/sch/gpt_lut/gpt_xl_t1000.csv', dtype=np.int32, comments='//', delimiter=',')
    f = open('/data1/dataset/sch/data/sch/gpt_lut/gpt_xl_t1000.csv', 'r')
    # line = list(map(int, f.readline().split(',')))
    # # print('*** Read first done')
    core_num = 217008
    chip_num = math.ceil(core_num / 160)
    assert chip_num <= 1400
    post_processed_csv = np.zeros((1400, 1400), dtype=np.int32)
    print('*** initial done')
    for i in range(core_num):
        line = list(map(int, f.readline().split(',')))
        i_chip = i // 160
        if i % 160 == 0:
            print(i_chip)
        for j in range(core_num):
            j_chip = j // 160
            if i_chip == j_chip:
                continue
            post_processed_csv[i_chip, j_chip] += line[j]
    print('process done')
    np.savetxt('/data/sch/gpt_lut/gpt_xl_t1000_processed.csv',
               post_processed_csv, fmt='%d', delimiter=',')  # 39019
    print('save done')


if __name__ == '__main__':
    main_mt()
