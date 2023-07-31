import sys
import numpy as np
import torch

sys.path.extend(['../'])
from graph import tools

num_node = 2095
self_link = [(i, i) for i in range(num_node)]


class MeshGraph:
    def __init__(self, labeling_mode='spatial', inward=None):
        if inward is None:
            self.inward = []
        else:
            self.inward = list(np.asarray(torch.load(inward)))
        self.num_node = num_node
        self.self_link = self_link
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = MeshGraph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
