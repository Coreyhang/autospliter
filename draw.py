import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time


def draw(lut: np.ndarray, save=None):
    old_time = time.time()
    assert len(lut.shape) == 2 and lut.shape[0] == lut.shape[1]
    print('**** Start Creating Graph ...')
    g = nx.DiGraph()
    g.add_nodes_from(range(lut.shape[0]))
    for i in range(lut.shape[0]):
        for j in range(lut.shape[0]):
            if lut[i, j] > 0:
                g.add_edge(i, j)
    print('**** Graph Generated Done, Start Drawing ...')
    # pos = nx.random_layout(g)
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, node_size=50, width=0.2)  # , node_color='black', edge_color='gray')
    print('**** Draw time is {:.4f}'.format(time.time() - old_time))
    if save is not None:
        plt.savefig('temp/' + save + '.svg', format='svg')
    else:
        plt.show()


if __name__ == '__main__':
    my_lut = np.random.randint(0, 2, size=(102, 102), dtype=np.int32)
    # print(my_lut)
    draw(my_lut)
