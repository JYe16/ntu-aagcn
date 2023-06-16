import sys

sys.path.extend(['../'])
from graph import tools

num_node = 504
self_link = [(i, i) for i in range(num_node)]


class MeshGraph:
    def __init__(self, labeling_mode='spatial', inward_ori_index=None):
        if inward_ori_index is None:
            inward_ori_index = []
        self.num_node = num_node
        self.self_link = self_link
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
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