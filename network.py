'''Pretty dumb module for managing networks'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import netplot

from termcolor import colored
from tqdm import trange, tqdm

class Node:

    def __init__(self, n):
        self.n = int(n)
        self.childs = []
        self.distances = []
        self.position = None

        # add a list of lines connected for display purposes
        self.lines = {}

        # the value of... something, I guess?
        self.value = None

    def connect(self, child, distance):
        # connections are called one time for couple
        self.childs.append(child)
        child.childs.append(self)

        self.distances.append(distance)
        child.distances.append(distance)

    def current_dist_from(self, child):
        return np.sqrt(np.sum((self.position - child.position)**2))

    def get_pulled_by_childs(self, epsilon):

        delta = np.zeros(self.position.shape)
        for child, target_dist in zip(self.childs, self.distances):
            real_dist = self.current_dist_from(child)
            delta += (1. - target_dist/real_dist)*(child.position - self.position)

        delta *= epsilon/len(self.childs)
        self.position += delta

    def __hash__(self):
        return self.n

    def __str__(self):
        desc = f'node {self.n}: \n'
        for child, dist in zip(self.childs, self.distances):
            desc += f'\tchild {child.n} at distance {dist}\n'
        return desc

class Network:

    def __init__(self, nodes):
        self.nodes = {}
        self.N = None
        self.executed_paths = [] # suggestion for the future: implement an iteration method
        self.repr_dim = 2

        # theese attributes must be filled
        # at the end of each constructor
        # M stands for matrix, SM stands for sparse
        self.targetM = None
        self.distanceM = None
        self.linkM = None
        self.targetSM = None
        self.distanceSM = None

        # for display purposes ()
        self.scatplot = None
        self.maxs = None
        self.mins = None

        self.max_expansions = 70

    @classmethod
    def from_sparse(cls, sparse_matrix):
        '''generates network from a sparse matrix'''

        # raw init
        net = cls([])

        net.targetSM = np.array(sparse_matrix)
        net.N = int(np.max(net.targetSM.transpose()[:2])) + 1

        print(f'Network has {net.N} elements')

        net.linkM = np.zeros((net.N, net.N), dtype=np.bool)
        net.targetM = np.zeros((net.N, net.N), dtype=np.float32)

        # gets a sparse matrix like (i,j) dist
        # and create nodes
        for i,j,distance in net.targetSM:
            i,j = int(i), int(j)
            node_in  = net.nodes.get(i, Node(i)) # fetch
            node_out = net.nodes.get(j, Node(j)) # fetch
            node_in.connect(node_out, distance)  # connect
            net.nodes[i] = node_in  # put back
            net.nodes[j] = node_out # put back

            print(f'>> linked {i} to {j}')

            net.linkM[i,j] = True
            net.linkM[j,i] = True

            net.targetM[i,j] = distance
            net.targetM[j,i] = distance

        net.colors = np.array([node.n for node in net.nodes.values()], dtype=np.float32)

        return net

    @classmethod
    def connect(cls, networks, links, distances):
        '''connects two networks.

        Args
        ----
            networks : tuple
                a tuple of networks to connect.

            links : list of 2-tuples
                links (net1_node_a, net2_node_b). -1 stands for densely connected.

                e.g, (5, -1) connects every element of net2 to element 5 of net1.
        '''
        net_1, net_2 = networks
        # first shifts every number of the second network
        for N2node in net_2.nodes.values():
            N2node.n += net_1.N

        for link in links:

            node_1, node_2 = link
            node_2 += net_1.N
            dense = False

            if node_1 == -1:
                for node in net_1.nodes.values():
                    node.connect(net_2.nodes[node_2])
                dense = True

            if node_1 == -1:
                for node in net_2.nodes.values():
                    node.connect(net_1.nodes[node_1])
                dense = True

            if not dense:
                net_1.nodes[node_net_1].connect(net_2.nodes[node_net_2])

    def init_positions(self, dim=2):
        self.repr_dim = dim
        for node in self.nodes.values():
            node.position = np.random.uniform(np.zeros((dim)), np.ones((dim)))

    def expand(self, epsilon):
        '''pushes all nodes away from all nodes,
        so basically maximizes the distance of everything from everything'''
        for node in self.nodes.values():
            delta = np.zeros(self.repr_dim)
            for anode in self.nodes.values():
                if node is not anode:
                    real_dist = node.current_dist_from(anode)
                    delta -= (anode.position - node.position)/real_dist
            delta *= epsilon
            node.position += delta/self.N

    def MDE(self, Nsteps=10):
        '''Minimal distortion embedding'''

        # takes one node at a time and makes it get pulled/pushed by its childs
        # but this does not guarantee that two unconnected nodes

        # first part: expand
        if self.max_expansions > 0:
            self.expand(1.)
            self.max_expansions -= 1
            print(f'remaining expansions: {self.max_expansions}')

        # second part: relax completely
        for iteration in range(Nsteps):
            for node in self.nodes.values():
                node.get_pulled_by_childs(0.1)

        # ending: subtracts position of center of mass
        Xcm = np.zeros(self.repr_dim)
        for node in self.nodes.values():
            Xcm += node.position/self.N

        for node in self.nodes.values():
            node.position -= Xcm

        self.get_distanceM()
        print(f'D = {self.get_distortion()}')

    def to_scatter(self):
        return np.array([node.position for node in self.nodes.values()]).transpose()

    def get_distanceM(self):
        self.distanceM = np.zeros((self.N,self.N))
        for node in self.nodes.values():
            for another_node in self.nodes.values():
                self.distanceM[node.n, another_node.n] = np.sqrt(np.sum( (node.position - another_node.position)**2 ))
        return self.distanceM

    def get_distanceSM(self):
        nlinks = np.sum(self.linkM.astype(np.int))
        self.distanceSM = np.array([])
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.linkM[i,j]:
                    self.distanceSM = np.append(self.distanceSM, [i,j, self.distanceM[i,j]] )
        self.distanceSM = self.distanceSM.reshape((-1,3))
        return self.distanceSM

    def get_distortion(self):
        return np.sum(((self.targetM - self.distanceM)*self.linkM.astype(np.float64))**2)

    def print_distanceM(self, target=False):
        M = self.get_distanceM()
        M = self.targetM if target else M
        title = 'Target matrix' if target else 'Current matrix '
        print(title + (30 -len(title))*'-' + f'(D = {A.get_distortion():.1e})')
        for i in range(self.N):
            for j in range(self.N):
                color = 'green' if self.linkM[i,j] else 'red'
                attrs = ['dark'] if i==j else ['bold']
                print(colored(f'{M[i,j]:.2}', color, attrs=attrs), end='\t')
            print()
        print()

    @classmethod
    def Hexahedron(cls):
        M = [[0,1, 1.],
             [1,2, 1.],
             [2,0, 1.],
             [3,0, 1.],
             [3,1, 1.],
             [3,2, 1.],
             [4,0, 1.],
             [4,1, 1.],
             [4,2, 1.]]

        return cls.from_sparse(M)

    @classmethod
    def Line(cls, n):
        M = []
        for i in range(n):
            M.append([i, i+1, 1.])
        return cls.from_sparse(M)

    @classmethod
    def Triangle(cls):
        M = [[0,1, 1.], [1,2, 1.], [2,0, 1.]]
        return cls.from_sparse(M)

    @classmethod
    def Random(cls, number_of_nodes, connection_probability, max_dist):
        M = np.array([])
        for i in range(number_of_nodes):
            for j in range(i+1, number_of_nodes):
                if np.random.uniform(0,1) < connection_probability:
                    M = np.append(M, [i,j, np.random.uniform(0,max_dist)])
        M = M.reshape((-1,3))
        net = cls.from_sparse(M)
        return net


    def __str__(self):
        '''Describes the tree'''
        desc = 'Network ---------\n'
        for node in self.nodes.values():
            desc += str(node)
        return desc

if __name__ == '__main__':
    np.random.seed(121)
    A = Network.Random(50,.50,1.)
    A.init_positions(dim=3)
    A.print_distanceM(target=True)
    # A.MDE(Nsteps=100)
    A.print_distanceM(target=False)
    # print(A.get_distanceSM())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # netplot.plot(A,ax)
    animation = netplot.animate_MDE(A,fig,ax,frames=120, interval=75, blit=False)
    # animation.save('random.mp4',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'), dpi=200)
    netplot.plot_links(A)
    plt.show()
