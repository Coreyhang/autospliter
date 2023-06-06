import numpy as np


def param_cal(n_layers=0, n_heads=0, d_head=0, d_model=0):
    # we only focus on encoder part
    # multi-head self attention
    param = n_heads * (d_model * d_head * 3)
    param += n_heads * d_head * d_model
    # feed forward
    param += d_model * 4 * d_model
    param += 4 * d_model * d_model

    return n_layers * param


def save_csv(data, file_name: str):
    np.savetxt('/data/sch/gpt_lut/' + file_name + '.csv', data, fmt='%d', delimiter=',')


def read_csv(file_name: str):
    return np.loadtxt('temp/' + file_name + '.csv', dtype=np.int32, comments='//', delimiter=',')


class CordTool:
    def __init__(self, size: tuple):
        super(CordTool, self).__init__()
        self.size = size
        self.weight = [1] * len(size)
        for i in range(len(size) - 1):
            idx = len(size) - i - 2
            self.weight[idx] = self.weight[idx + 1] * self.size[idx + 1]
        self.max_index = self.weight[0] * self.size[0] - 1

    def flatten(self, cord):
        assert len(self.size) == len(cord)
        res = 0
        for i in range(len(self.size)):
            assert self.size[i] > cord[i]
            res += self.weight[i] * cord[i]
        return res

    def fold(self, index):
        assert self.max_index >= index
        res = ()
        for i in range(len(self.size)):
            res = (*res, index // self.weight[i])
            index %= self.weight[i]
        return res


if __name__ == '__main__':
    size_t = np.random.randint(1, 15, 5, dtype=np.int32)
    size_t = (4, 2, 3, 5)
    cord_u = CordTool(size=size_t)
    for i in range(100):
        cord_t = [np.random.randint(0, j) for j in size_t]
        ind = cord_u.flatten(cord_t)
        cordd = cord_u.fold(ind)
        assert cordd == tuple(cord_t)
        print(cordd, cord_t, ind)
    # size_t = np.random.randint(1, 150, (4, 6), dtype=np.int32)
    # print(size_t)
    # save_csv(size_t, 'test')
    # print(read_csv('test'))
