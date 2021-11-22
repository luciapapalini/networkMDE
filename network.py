'''Pretty dumb module for managing networks'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import netplot
import utils

from termcolor import colored
from tqdm import trange, tqdm

class Node:

    def __init__(self, n):
        self.n = int(n)
        self.childs = {}
        self.distances = {}
        self.position = None
        self._value = None  # the value of... something, I guess?


        # add a list of lines connected for display purposes
        self.lines = {}
        self.synapsis = {}

    @property
    def value(self):
        if self._value is not None:
            return self._value
        raise RuntimeWarning('node value not defined yet')

    @value.setter
    def value(self, value):
        self._value = value # never used the word 'value' so much times in my life

    def connect(self, child, distance):
        # connections are called one time for couple
        self.childs[child] = distance
        child.childs[self] = distance

    def current_dist_from(self, child):
        return np.sqrt(np.sum((self.position - child.position)**2))

    def get_pulled_by_childs(self, epsilon):
        '''Gradient-based move for MDE

        For the single node the loss function (difference from target distance) is

        L = sum_childs[ ( distance_from_child - target_distance_from_child)**2 ]

        so  - grad (L) = 2 * sum_c[ (x - x_c)/|x - x_c| * ( |x - x_c| - target_dist) ]

        '''
        delta = np.zeros(self.position.shape)
        for child, target_dist in self.childs.items():
            real_dist = self.current_dist_from(child)
            delta += (1. - target_dist/real_dist)*(child.position - self.position)

        delta *= epsilon/len(self.childs)
        self.position += delta

    def __hash__(self):
        return self.n

    def __str__(self):
        desc = f'node {self.n}: \n'
        for child, dist in self.childs.items():
            desc += f'\tchild {child.n} at distance {dist}\n'
        return desc

class Link:
    '''almost useless class, may remove it later

    Used only to make 1 -> 2 equal to 2 -> 1 like in 'set()' properties.

    '''

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def __hash__(self):
        # I need a symmetric hardly-repeatable function
        # so that h(1,2) == h(2,1) ( 1->2 is the same as 2->1)
        # but the dict looses efficiency for collisions
        # so h(.,. ) = .+. is not intelligent (6 = 3+3, 2+4, 1+5,...)
        return self.a**10 + self.b**10

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
        self._distanceM = None
        self.linkM = None
        self.targetSM = None
        self.distanceSM = None

        # for display purposes ()
        self.scatplot = None

        #blow-the-glove max iterations
        self.max_expansions = 70

    @classmethod
    def from_sparse(cls, sparse_matrix):
        '''generates network from a sparse matrix'''

        # raw init
        net = cls([])

        net.targetSM = np.array(sparse_matrix)
        net.N = int(np.max(net.targetSM.transpose()[:2])) + 1


        net.linkM = np.zeros((net.N, net.N), dtype=np.bool)
        net.targetM = np.zeros((net.N, net.N), dtype=np.float32)

        # gets a sparse matrix like (i,j) dist
        # and create nodes
        links = {}
        for i,j,distance in net.targetSM:

            i,j = int(i), int(j)

            node_in  = net.nodes.get(i, Node(i)) # fetch from dict or create
            node_out = net.nodes.get(j, Node(j)) # fetch from dict or create
            node_in.connect(node_out, distance)  # connect
            net.nodes[i] = node_in  # put back
            net.nodes[j] = node_out # put back

            print(f'>> linked {i} to {j}', end='\r')

            net.linkM[i,j] = True
            net.linkM[j,i] = True

            net.targetM[i,j] = distance
            net.targetM[j,i] = distance

            links[hash(Link(i,j))] = distance # pretty useless

        print(f'Network has {len(net.nodes)} elements and {len(links)} links')
        return net

    @classmethod
    def from_adiacence(cls, matrix):
        '''generates network from a sparse matrix'''
        # here checks if the matrix is a good one
        # that is to say square, symmetric and M_ii = 0
        # float comparison: dangerous?
        matrix = np.array(matrix)

        if (matrix != matrix.transpose()).any():
            raise ValueError('Matrix is not symmetric')

        if (matrix.diagonal() != np.zeros(len(matrix))).any():
            raise ValueError('Matrix has non-null diagonal')

        sparseM = utils.matrix_to_sparse(matrix)
        net = Network.from_sparse(sparseM)
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
        raise NotImplementedError('I have to finish this')
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
        for node in tqdm(self.nodes.values(), desc='position init', leave=False):
            node.position = np.random.uniform(np.zeros((dim)), np.ones((dim)))

    @property
    def colors(self):
        return [node.value for node in self.nodes.values()]

    @property
    def distanceM(self):
        self._distanceM = np.zeros((self.N,self.N))
        for node in self.nodes.values():
            for another_node in self.nodes.values():
                self._distanceM[node.n, another_node.n] = np.sqrt(np.sum( (node.position - another_node.position)**2 ))
        return self._distanceM

    @distanceM.setter
    def distanceM(self, matrix):
        self._distanceM = matrix

    @property
    def distanceSM(self):
        nlinks = np.sum(self.linkM.astype(np.int))
        self._distanceSM = np.array([])
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.linkM[i,j]:
                    self._distanceSM = np.append(self._distanceSM, [i,j, self._distanceM[i,j]] )
        self._distanceSM = self._distanceSM.reshape((-1,3))
        return self._distanceSM

    @distanceSM.setter
    def distanceSM(self, smatrix):
        self._distanceSM = smatrix

    @property
    def distortion(self):
        return np.sum(((self.targetM - self.distanceM)*self.linkM.astype(np.float64))**2)

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

    def MDE(self, Nsteps=10, verbose=True):
        '''Minimal distortion embedding

        Minimizes the discrepancy between the target distances and the
        actual distances of the points by letting each node be pulled by its childs.

        Also tries to spread apart loosely connected regions of the network, e.g.
        the graph given by the sparse matrix:

                        [[0,1, 1.], [0,2, 1.], [0,3, 1.]]

        can be represented in two equivalent ways (with the same (null) distortion):

                 repr 1                        repr 2

                   1
                   |
                   0                          0 -- 1=2=3
                  / \
                 2   3

        since the distances are respected. The second representation is by the way less clear
        than the first.

        To solve this the algorithm uses a blow-the-glove strategy: for a fixed
        number of iterations each node repel each other, then it is relaxed by
        child-pulling to the minimum distortion.
        '''
        with tqdm(range(Nsteps), desc=f'MDE') as pbar:
            for iteration in pbar:
                if self.max_expansions > 0:
                    self.expand(1.)
                    self.max_expansions -= 1

                for node in self.nodes.values():
                    node.get_pulled_by_childs(0.1)
                pbar.set_description(f'MDE -- distortion {self.distortion :.2f}')

            # if verbose: print(f'>> MDE {int(iteration/Nsteps*100) : 2d}% --- distortion: {self.distortion :.2f}', end='\r')

        # ending: subtracts position of center of mass
        Xcm = np.zeros(self.repr_dim)
        for node in self.nodes.values():
            Xcm += node.position/self.N

        for node in self.nodes.values():
            node.position -= Xcm

    def to_scatter(self):
        return np.array([node.position for node in self.nodes.values()]).transpose()

    def print_distanceM(self, target=False):
        M = self.targetM if target else self.distanceM
        title = 'Target matrix' if target else 'Current matrix '
        print(title + (30 -len(title))*'-' + f'(D = {A.distortion:.1e})')
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
    def Random(cls, number_of_nodes, connection_probability, max_dist=1.):
        '''Random network constructor

        Since it is unclear to me what a random network is,
        I made all wrong on purpose.

        A random matrix is generated, uniform elements.

        Then the symmetrization operation spoils the statistical properties
        since the elements of (M + transpose(M))/2 are distributed
        as a triangle.

        Then links are generated for the element that are above the threshold
        given by connection_probability.

        I know It doesn't really make any sense, but in this way
        'connection_probability' parametrizes the number of connections
        from 0 -> N(N-1/2) in a smooth way.
        '''
        M = np.random.uniform(0,1, size=number_of_nodes**2).reshape((-1, number_of_nodes))
        M = 0.5*(M + M.transpose())
        np.fill_diagonal(M, 1.)
        links = (M < connection_probability).astype(float)
        M = M*links*max_dist

        return Network.from_adiacence(M)


    def __str__(self):
        '''Describes the tree'''
        desc = 'Network ---------\n'
        for node in self.nodes.values():
            desc += str(node)
        return desc

if __name__ == '__main__':
    A = Network.Random(10,.8)
    A.init_positions(dim=3)
    # A.print_distanceM(target=True)
    # A.MDE(Nsteps=100)
    A.print_distanceM(target=False)
    # print(A.get_distanceSM())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    netplot.plotNet(A,ax)
    # animation = netplot.animate_MDE(A,fig,ax,frames=120, interval=75, blit=False)
    # animation.save('random.mp4',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}: D = {A.distortion:.2f} (remaining expansions: {A.max_expansions})', end='\r'), dpi=200)
    # netplot.plot_links(A)
    plt.show()
