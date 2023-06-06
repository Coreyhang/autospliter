import numpy as np
from itertools import product
from math import ceil, inf, floor
from utils import CordTool, save_csv, read_csv
from draw import draw
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support, cpu_count
import time
from copy import deepcopy


class GPTSpliter:
    def __init__(self, n_layers=0, n_heads=0, d_head=0, d_model=0, T=0, solution_num=100, max_memory=144):
        super(GPTSpliter, self).__init__()
        assert d_model == n_heads * d_head
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.T = T

        self.max_memory = max_memory  # 144KB
        self.solution_num = solution_num

        self.l1_param = {}
        self.l2_param = {}
        self.l1_lut = None
        self.l2_lut = None
        self.l1_l2_lut = None
        self.l2_l1_lut = None
        self.lut = None

        self.cal_split_param_1()
        self.cal_split_param_2()
        # param2 = self.select(self.l2_param, mode='cheaper')
        # time2 = self.l2_param[param2]
        # param1 = self.select(self.l1_param, mode='timely', expe_time=time2)
        self.param1 = self.select(self.l1_param, mode='cheaper')
        self.time1 = self.l1_param[self.param1]
        self.param2 = self.select(self.l2_param, mode='timely', expe_time=self.time1)
        self.time2 = self.l2_param[self.param2]
        print('**** param1 = {}, core num = {}, time = {}'.format(
            self.param1, self.param1[0] * self.param1[1] * self.param1[2] * 3 * n_heads, self.time1))
        print('**** param2 = {}, core num = {}, time = {}'.format(
            self.param2, self.param2[0] * self.param2[1] * self.param2[2], self.time2))
        self.gen_lut_1(self.n_heads, *self.param1)
        self.gen_lut_2(*self.param2)
        self.gen_lut()
        print('\n**** total core num = {}\n'.format(self.n_layers * (
                    self.param1[0] * self.param1[1] * self.param1[2] * 3 * n_heads + self.param2[0] * self.param2[1] *
                    self.param2[2])))

    @staticmethod
    def gen_lut_1_head(shead_idx, shead, sow, siw, sh, T, d_head):
        lut = np.zeros((3 * sow * siw * sh, shead * 3 * sow * siw * sh), dtype=np.int32)
        cord_tool = CordTool(size=(shead, 3, sow, siw, sh))
        for j in range(3):  # q k v
            for ow in range(sow):  # sow
                for iw in range(siw):  # siw
                    for h in range(sh):  # sh
                        src_index = cord_tool.flatten((0, j, ow, iw, h))  # 0 is necessary
                        # process
                        # q = x * Wq, k = x * Wk, v = x * Wv
                        # transfer for partial sum (siw)
                        for dst_iw in range(siw):
                            if dst_iw == iw:
                                continue
                            transfer_size = ceil(T / sh / siw) * ceil(d_head / sow)
                            lut[src_index, cord_tool.flatten((shead_idx, j, ow, dst_iw, h))] += transfer_size
                        # partial sum
                        # transfer :
                        # q & k send to each other
                        if j != 2:
                            dst_j = 1 if j == 0 else 0
                            transfer_size = ceil(T / sh / siw / 2) * ceil(d_head / sow)
                            lut[src_index, cord_tool.flatten((shead_idx, dst_j, ow, iw, h))] += transfer_size
                            # att = q * k
                            # transfer for partial sum (sow)
                            for dst_ow in range(sow):
                                if dst_ow == ow:
                                    continue
                                transfer_size = ceil(T / sh / siw / 2) * ceil(T / sh / siw / 2) / sow
                                lut[src_index, cord_tool.flatten((shead_idx, j, dst_ow, iw, h))] += transfer_size
                            # partial sum
                            for iter_num in range(2 * sh * siw - 1):
                                # transfer k
                                if h == sh - 1 and (iw == siw - 1 or iw == 0):
                                    dst_j = 0 if j == 1 else 1
                                    dst_iw = 0
                                    dst_h = 0
                                elif h % 2 == 0:
                                    dst_j = j
                                    if iw == siw - 1:
                                        dst_iw = iw
                                        dst_h = h + 1
                                    else:
                                        dst_iw = iw + 1
                                        dst_h = h
                                else:
                                    dst_j = j
                                    if iw == 0:
                                        dst_iw = iw
                                        dst_h = h + 1
                                    else:
                                        dst_iw = iw - 1
                                        dst_h = h
                                transfer_size = ceil(T / sh / siw / 2) * ceil(d_head / sow)
                                lut[src_index, cord_tool.flatten(
                                    (shead_idx, dst_j, ow, dst_iw, dst_h))] += transfer_size
                                # att = q * k
                                # transfer for partial sum (sow)
                                for dst_ow in range(sow):
                                    if dst_ow == ow:
                                        continue
                                    transfer_size = ceil(T / sh / siw / 2) * ceil(T / sh / siw / 2) / sow
                                    lut[src_index, cord_tool.flatten((shead_idx, j, dst_ow, iw, h))] += transfer_size
                                # partial sum
                            # mask & softmax
                        # transfer for att * v :
                        # k -> v; v -> q
                        if j != 0:
                            dst_j = 0 if j == 2 else 2
                            transfer_size = ceil(T / sh / siw) * ceil(d_head / sow / 2)
                            lut[src_index, cord_tool.flatten((shead_idx, dst_j, ow, iw, h))] += transfer_size
                        # att * v
                        # transfer for partail sum (siw * sh)
                        if j != 1:
                            transfer_size = ceil(T / 2 / sow / siw / sh) * ceil(d_head / sow / 2)
                            for dst_iw, dst_h in product(range(siw), range(sh)):
                                if dst_iw == iw and dst_h == h:
                                    continue
                                lut[src_index, cord_tool.flatten((shead_idx, j, ow, dst_iw, dst_h))] += transfer_size
                            # partial sum
                            # switch v
                            for iter_num in range(2 * sow):
                                if ow == sow - 1:
                                    dst_j = 0 if j == 2 else 2
                                    dst_ow = 0
                                else:
                                    dst_j = j
                                    dst_ow = ow + 1
                                transfer_size = ceil(T / sh / siw) * ceil(d_head / sow / 2)
                                lut[src_index, cord_tool.flatten((shead_idx, dst_j, dst_ow, iw, h))] += transfer_size
                                # att * v
                                # transfer for partial sum (siw * sh)
                                transfer_size = ceil(T / 2 / sow / siw / sh) * ceil(d_head / sow / 2)
                                for dst_iw, dst_h in product(range(siw), range(sh)):
                                    if dst_iw == iw and dst_h == h:
                                        continue
                                    lut[src_index, cord_tool.flatten(
                                        (shead_idx, j, ow, dst_iw, dst_h))] += transfer_size
                                # partial sum
                        # y = y-att * wo :
                        # transfer of y-att (each got 1/3)
                        if j != 1:
                            transfer_size = ceil(T / 6 / sow) * ceil(d_head / siw / sh)
                            lut[src_index, cord_tool.flatten((shead_idx, 1, ow, iw, h))] += transfer_size
                        # y-att * wo
                        # transfer for partial sum (siw * sh * shead)
                        transfer_size = ceil(T / 3 / sow) * ceil(d_head / 3 / sow) / siw / sh / shead
                        for dst_iw, dst_h, dst_head in product(range(siw), range(sh), range(shead)):
                            if dst_iw == iw and dst_h == h and dst_head == shead_idx:
                                continue
                            lut[src_index, cord_tool.flatten((dst_head, j, ow, dst_iw, dst_h))] += transfer_size
                        # partial sum
                        # switch wo
                        transfer_size = ceil(d_head / sh / siw) * ceil(d_head / sow / 3)
                        for iter_num in range(3 * sow * shead):
                            if ow == sow - 1:
                                dst_j = 0 if j == 2 else j + 1
                                dst_ow = 0
                            else:
                                dst_j = j
                                dst_ow = ow + 1
                            lut[src_index, cord_tool.flatten((shead_idx, dst_j, dst_ow, iw, h))] += transfer_size
                            # y-att * wo
                            # transfer for partial sum (siw * sh * shead)
                            transfer_size = ceil(T / 3 / sow) * ceil(
                                d_head / 3 / sow) / siw / sh / shead
                            for dst_iw, dst_h, dst_head in product(range(siw), range(sh), range(shead)):
                                if dst_iw == iw and dst_h == h and dst_head == shead_idx:
                                    continue
                                lut[src_index, cord_tool.flatten(
                                    (dst_head, j, ow, dst_iw, dst_h))] += transfer_size
                            # partial sum
                        # finish
        return shead_idx, lut

    @staticmethod
    def gen_lut_2_head(sow_idx, sow, siw, sh, T, d_model):
        lut = np.zeros((siw * sh, sow * siw * sh), dtype=np.int32)
        cord_tool = CordTool(size=(sow, siw, sh))
        for iw in range(siw):  # siw
            for h in range(sh):  # sh
                src_index = cord_tool.flatten((0, iw, h))
                # MLP1
                # transfer for partial sum (siw)
                for dst_iw in range(siw):
                    if dst_iw == iw:
                        continue
                    transfer_size = ceil(T / sh / siw) * ceil(4 * d_model / sow)
                    lut[src_index, cord_tool.flatten((sow_idx, dst_iw, h))] += transfer_size
                # partial sum
                # ReLU
                # transfer (siw)
                for dst_iw in range(siw):
                    if dst_iw == iw:
                        continue
                    transfer_size = ceil(T / sh / siw) * ceil(4 * d_model / sow)
                    lut[src_index, cord_tool.flatten((sow_idx, dst_iw, h))] += transfer_size
                # MLP2
                # transfer for partial sum (sow)
                for dst_ow in range(sow):
                    if dst_ow == sow_idx:
                        continue
                    transfer_size = ceil(T / sh / sow) * ceil(d_model / siw)
                    lut[src_index, cord_tool.flatten((dst_ow, iw, h))] += transfer_size
                # partial sum
                # shortcut add
        return sow_idx, lut

    def cal_split_param_1(self):
        """calculate multi-head self attention split parameter"""
        res = {}
        shead, siw, sow, sh = self.n_heads, 0, 0, 0
        while len(res.keys()) < self.solution_num:  # find more than 100 feasible solutions
            siw += 1
            sow += 1
            sh += 1
            for ow, iw, h in product([siw, siw + 1], [sow, sow + 1], [sh, sh + 1]):
                if iw == siw + 1 and ow == sow + 1 and h == sh + 1:
                    continue
                time_mat = 0
                # q = x * Wq, k = x * Wk, v = x * Wv
                i1_size = ceil(self.T / h) * ceil(self.d_model / iw)
                o1_size = ceil(self.T / h) * ceil(self.d_head / ow)
                w1_size = ceil(self.d_model / iw) * ceil(self.d_head / ow)
                time_mat += ceil(self.T / h) * ceil(self.d_head / ow) * ceil(self.d_model / iw) / 32 / 4  # mat
                time_mat += ceil(self.T / h) * ceil(self.d_head / ow) * (iw - 1) / 32  # sum
                # att = q * k^T / sqrt(n_head)
                i2_size = ceil(self.T / h / iw) * ceil(self.d_head / ow)
                o2_size = ceil(self.T / 2 / h / iw) ** 2
                o2_ps = ceil(self.T / 2 / ow) * ceil(self.T / iw / h)
                time_mat += ((ceil(self.T / h / iw / 2) ** 2) * ceil(self.d_head / ow) / 32 / 4) * (2 * iw * h)  # mat
                time_mat += (ceil(self.T / h / iw / ow / 2) * ceil(self.T / 2 / h / iw) * (ow - 1) / 32) * (
                        2 * iw * h)  # sum
                # mask & softmax
                lut_size = (2 ** 8) * 3
                time_mat += ceil(self.T / 2 / ow) * ceil(self.T / iw / h) * 3
                time_mat += ceil(self.T / 2 / ow) * ceil(self.T / iw / h) / 32
                # att * v
                i3_size = ceil(self.T / 2 / ow) * ceil(self.T / iw / h) + \
                          ceil(self.T / iw / h) * ceil(self.d_head / 2 / ow)
                o3_size = ceil(self.T / 2 / ow) * ceil(self.d_head / 2 / ow)
                o3_ps = ceil(self.T / iw / h) * ceil(self.T / 2 / ow)
                time_mat += (ceil(self.T / ow / 2) * ceil(self.T / iw / h) * ceil(self.d_head / 2 / ow) / 32 / 4) * (
                        2 * ow)  # mat
                time_mat += (ceil(self.T / 2 / ow / iw / h) * ceil(self.d_head / 2 / ow) * (iw * h - 1) / 32) * (
                        2 * ow)  # sum
                # y = y-att * wo :
                i4_size = ceil(self.T / 2 / ow) * ceil(self.d_head / iw / h)
                w4_size = ceil(self.d_head / iw / h) * ceil(self.d_head / 3 / ow)
                o4_size = ceil(self.T / 3 / ow) * ceil(self.d_head / 3 / ow)
                o4_ps = ceil(self.T / 3 / h) * ceil(self.d_head / iw / ow)
                time_mat += (ceil(self.T / ow / 3) * ceil(self.d_head / iw / h) * ceil(
                    self.d_head / 3 / ow) / 32 / 4) * (3 * ow * self.n_heads)  # mat
                time_mat += (ceil(self.T / 3 / ow / iw / h / self.n_heads) * ceil(
                    self.d_head / 3 / ow) * (iw * h * self.n_heads - 1) / 32) * (3 * ow * self.n_heads)  # sum
                mem_use = w1_size + lut_size + w4_size + i1_size + max(o1_size, i2_size, o2_size, o2_ps, i3_size,
                                                                       o3_size, o3_ps, i4_size, o4_size, o4_ps)
                if mem_use < self.max_memory * 1024:
                    res[(ow, iw, h)] = time_mat
        self.l1_param = res

    def cal_split_param_2(self):
        """calculate MLP split parameter"""
        res = {}
        siw, sow, sh = 0, 0, 0
        while len(res.keys()) < self.solution_num:  # find more than 100 feasible solutions
            siw += 1
            sow += 1
            sh += 1
            for ow, iw, h in product([siw, siw + 1], [sow, sow + 1], [sh, sh + 1]):
                if iw == siw + 1 and ow == sow + 1 and h == sh + 1:
                    continue
                time_mat = 0
                # MLP1
                i1_size = ceil(self.T / h) * ceil(self.d_model / iw)
                o1_size = ceil(self.T / h) * ceil(4 * self.d_model / ow)
                w1_size = ceil(self.d_model / iw) * ceil(4 * self.d_model / ow)
                o1_ps = ceil(self.T / h / iw) * ceil(self.d_model * 4 / ow)
                time_mat += ceil(self.T / h) * ceil(self.d_model / iw) * ceil(4 * self.d_model / ow) / 32 / 4  # mat
                time_mat += ceil(self.T / h / iw) * ceil(4 * self.d_model / ow) * (iw - 1) / 32  # sum
                time_mat += ceil(self.T / h / iw) * ceil(4 * self.d_model / ow) / 32  # ReLU
                # MLP2
                i2_size = ceil(self.T / h) * ceil(4 * self.d_model / ow)
                o2_size = ceil(self.T / h) * ceil(4 * self.d_model / iw)
                w2_size = ceil(4 * self.d_model / ow) * ceil(self.d_model / iw)
                o2_ps = ceil(self.T / h / ow) * ceil(self.d_model / iw)
                time_mat += ceil(self.T / h) * ceil(4 * self.d_model / ow) * ceil(self.d_model / iw) / 32 / 4  # mat
                time_mat += ceil(self.T / h / ow) * ceil(self.d_model / iw) * (ow - 1) / 32  # sum
                time_mat += ceil(self.T / h / ow) * ceil(self.d_model / iw) / 32  # shortcut add
                mem_use = w1_size + w2_size + i1_size + max(o1_size, o1_ps, i2_size, o2_size, o2_ps)
                if mem_use < self.max_memory * 1024:
                    res[(ow, iw, h)] = time_mat
        self.l2_param = res

    def gen_lut_1(self, shead, sow, siw, sh):
        """generate LUT table for multi-head self attention"""
        print('**** Start Generating LUT 1, Using {}/{} CPU ...'.format(cpu_count() * 3 // 4, cpu_count()))
        start_time = time.time()
        res = []
        freeze_support()
        pool = Pool(cpu_count() * 3 // 4)
        for i in range(shead):  # s_heads
            pool.apply_async(func=GPTSpliter.gen_lut_1_head,
                             args=(i, shead, sow, siw, sh, self.T, self.d_head,),
                             callback=res.append)
        pool.close()
        pool.join()
        pool.terminate()
        res = sorted(res)
        self.l1_lut = np.concatenate([item[1] for item in res])
        print('**** Generating LUT 1 done, time = {:.4f}'.format(time.time() - start_time))

    def gen_lut_2(self, sow, siw, sh):
        """generate LUT table for MLP"""
        print('**** Start Generating LUT 2, Using {}/{} CPU ...'.format(cpu_count() * 3 // 4, cpu_count()))
        start_time = time.time()
        res = []
        freeze_support()
        pool = Pool(cpu_count() * 3 // 4)
        for ow in range(sow):  # sow
            pool.apply_async(func=GPTSpliter.gen_lut_2_head,
                             args=(ow, sow, siw, sh, self.T, self.d_model,),
                             callback=res.append)
        pool.close()
        pool.join()
        pool.terminate()
        res = sorted(res)
        self.l2_lut = np.concatenate([item[1] for item in res])
        print('**** Generating LUT 2 done, time = {:.4f}'.format(time.time() - start_time))

    @staticmethod
    def gen_lut_inter_part1(shead_idx, shead1, sow1, siw1, sh1, T, d_model, sh2, siw2, sow2):
        t_part1 = 3 * sh1 * shead1
        d_part1 = siw1 * sow1
        lut = np.zeros((3 * sow1 * siw1 * sh1, sh2 * siw2 * sow2), dtype=np.int32)
        cord1 = CordTool(size=(shead1, 3, sow1, siw1, sh1))
        cord2 = CordTool(size=(sow2, siw2, sh2))
        cord1_t = CordTool(size=(3, sh1, shead1))
        cord1_d = CordTool(size=(siw1, sow1))
        total_size = T * d_model
        for j in range(3):  # q k v
            for ow in range(sow1):  # sow
                for iw in range(siw1):  # siw
                    for h in range(sh1):  # sh
                        src_t = cord1_t.flatten((j, h, shead_idx))
                        src_d = cord1_d.flatten((iw, ow))
                        src_cord = cord1.flatten((0, j, ow, iw, h))
                        for h2 in range(sh2):
                            for iw2 in range(siw2):
                                transfer_size = \
                                    max(min((src_t + 1) / t_part1, (h2 + 1) / sh2) - max(src_t / t_part1, h2 / sh2),
                                        0) * \
                                    max(min((src_d + 1) / d_part1, (iw2 + 1) / siw2) - max(src_d / d_part1, iw2 / siw2),
                                        0)
                                if transfer_size > 0:
                                    dst_cord = cord2.flatten((0, iw2, h2))
                                    lut[src_cord, dst_cord::(siw2 * sh2)] = total_size * transfer_size
        return shead_idx, lut

    @staticmethod
    def gen_lut_inter_part2(sow_idx, sow2, siw2, sh2, T, d_model, shead1, sow1, siw1, sh1):
        t_part2 = sh2 * sow2
        d_part2 = siw2
        lut = np.zeros((sh2 * siw2, shead1 * 3 * sow1 * siw1 * sh1), dtype=np.int32)
        cord1 = CordTool(size=(shead1, 3, sow1, siw1, sh1))
        cord2 = CordTool(size=(sow2, siw2, sh2))
        cord2_t = CordTool(size=(sh2, sow2))
        total_size = T * d_model
        for h2 in range(sh2):
            for iw2 in range(siw2):
                src_t = cord2_t.flatten((h2, sow_idx))
                src_d = iw2
                src_cord = cord2.flatten((0, iw2, h2))
                # for j in range(3):  # q k v
                #     for ow1 in range(sow1):  # sow
                for iw1 in range(siw1):  # siw
                    for h1 in range(sh1):  # sh
                        transfer_size = \
                            max(min((src_t + 1) / t_part2, (h1 + 1) / sh1) - max(src_t / t_part2, h1 / sh1),
                                0) * \
                            max(min((src_d + 1) / d_part2, (iw1 + 1) / siw1) - max(src_d / d_part2, iw1 / siw1),
                                0)
                        if transfer_size > 0:
                            dst_cord = cord1.flatten((0, 0, 0, iw1, h1))
                            lut[src_cord, dst_cord::(siw1 * sh1)] = total_size * transfer_size
        return sow_idx, lut

    def gen_lut_inter_group(self):
        print('**** Start Generating inter group LUT, Using {}/{} CPU ...'.format(cpu_count() * 3 // 4, cpu_count()))
        core_num1 = self.param1[0] * self.param1[1] * self.param1[2] * 3 * self.n_heads
        core_num2 = self.param2[0] * self.param2[1] * self.param2[2]
        sh1, shead1, siw1, sow1 = self.param1[2], self.n_heads, self.param1[1], self.param1[0]
        sh2, siw2, sow2 = self.param2[2], self.param2[1], self.param2[0]
        # part 1 : multi-head self attention -> MLP
        print('**** Start Generating LUT 1 -> LUT 2 ...')
        start_time = time.time()
        res = []
        freeze_support()
        pool = Pool(cpu_count() * 3 // 4)
        for i in range(self.n_heads):  # s_heads
            # GPTSpliter.gen_lut_inter_part1(i, shead1, sow1, siw1, sh1, self.T, self.d_model, sh2, siw2, sow2)
            pool.apply_async(func=GPTSpliter.gen_lut_inter_part1,
                             args=(i, shead1, sow1, siw1, sh1, self.T, self.d_model, sh2, siw2, sow2,),
                             callback=res.append)
        pool.close()
        pool.join()
        pool.terminate()
        res = sorted(res)
        self.l1_l2_lut = np.concatenate([item[1] for item in res])
        print('**** Generating LUT 1 -> LUT 2 done, time = {:.4f}'.format(time.time() - start_time))
        # part 2 : MLP -> multi-head self attention
        print('**** Start Generating LUT 2 -> LUT 1 ...')
        start_time = time.time()
        res = []
        freeze_support()
        pool = Pool(cpu_count() * 3 // 4)
        for ow in range(sow2):  # sow
            # GPTSpliter.gen_lut_inter_part2(ow, sow2, siw2, sh2, self.T, self.d_model, shead1, sow1, siw1, sh1)
            pool.apply_async(func=GPTSpliter.gen_lut_inter_part2,
                             args=(ow, sow2, siw2, sh2, self.T, self.d_model, shead1, sow1, siw1, sh1,),
                             callback=res.append)
        pool.close()
        pool.join()
        pool.terminate()
        res = sorted(res)
        self.l2_l1_lut = np.concatenate([item[1] for item in res])
        print('**** Generating LUT 2 -> LUT 1 done, time = {:.4f}'.format(time.time() - start_time))

    def gen_lut(self):
        self.gen_lut_inter_group()
        core_num1 = self.param1[0] * self.param1[1] * self.param1[2] * 3 * self.n_heads
        core_num2 = self.param2[0] * self.param2[1] * self.param2[2]
        block_core_num = core_num2 + core_num1
        self.lut = np.zeros((block_core_num * self.n_layers, block_core_num * self.n_layers), dtype=np.int32)
        lut1 = np.concatenate([self.l1_lut, self.l1_l2_lut], axis=1)
        lut2 = np.concatenate([self.l2_lut, self.l2_l1_lut], axis=1)
        print('**** Start Generating LUT ...')
        start_time = time.time()
        for layer in range(self.n_layers):
            # part 1 : multi-head self attention
            self.lut[layer * block_core_num:layer * block_core_num + core_num1,
            layer * block_core_num:(layer + 1) * block_core_num] = lut1
            # part 2 : MLP
            if layer == self.n_layers - 1:
                self.lut[layer * block_core_num + core_num1:(layer + 1) * block_core_num,
                layer * block_core_num + core_num1:(layer + 1) * block_core_num] = self.l2_lut
            else:
                self.lut[layer * block_core_num + core_num1:(layer + 1) * block_core_num,
                layer * block_core_num + core_num1:(layer + 1) * block_core_num + core_num1] = lut2
        print('**** Generating LUT done, time = {:.4f}'.format(time.time() - start_time))

    @staticmethod
    def select(param: dict, mode: str, expe_time=0):
        if mode == 'faster':
            min_time = inf
            min_key = (inf, inf, inf)
            for key, value in param.items():
                if min_time > value:
                    min_key = key
                    min_time = value
            return min_key
        elif mode == 'cheaper':
            min_key = (inf, inf, inf)
            for key, value in param.items():
                if key[0] * key[1] * key[2] < min_key[0] * min_key[1] * min_key[2]:
                    min_key = key
            return min_key
        elif mode == 'timely':
            # find min_time
            min_time = inf
            for value in param.values():
                if min_time > value:
                    min_time = value
            #
            assert expe_time > 0
            fit_time = inf
            min_key = (inf, inf, inf)
            for key, value in param.items():
                if value == min_time:
                    break
                if abs(value - expe_time) < abs(fit_time - expe_time):
                    min_key = key
                    fit_time = value
            return min_key
        elif mode == 'plot':
            plt.bar(list(range(len(param.values()))), list(param.values()))
            plt.show()
        else:
            raise NotImplementedError
