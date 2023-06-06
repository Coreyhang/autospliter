import multiprocessing
from multiprocessing import Pool, freeze_support, cpu_count
import time
import numpy as np


class B:
    def __init__(self):
        pass


def Bar(arg):
    print(arg)


class aaa:
    def __init__(self):
        self.amin()

    def Foo(i):
        bb = B()
        a = np.zeros((1,4))
        for j in range(4):
            print(i, j)
            a[0, j] += j

    def amin(self):
        freeze_support()
        print("cpu num is ", multiprocessing.cpu_count())
        t_start = time.time()
        pool = Pool(5)

        for i in range(10):
            pool.apply_async(func=aaa.Foo, args=(i,))

        pool.close()
        pool.join()
        pool.terminate()
        t_end = time.time()
        t = t_end - t_start
        print('the program time is :%s' % t)


if __name__ == '__main__':
    a = aaa()
