"""Pretty dumb module for managing networks and display them in the less
distorted way possible (MDE).

author: djanloo
date: 20 nov 21
"""
import numpy as np

from termcolor import colored
from tqdm import tqdm

import cnets
from . import utils
from . import classiter as ci

class Node:

    def __init__(self, n):

        self.n = int(n)
        self.synapses = ci.cset()
        self._position = None
        self._value = None  # the value of... something, I guess?

    @property
    def value(self):
        if self._value is not None:
            return self._value
        raise RuntimeWarning("Node value not defined yet")

    @value.setter
    def value(self, value):
        self._value = value
    
    @property
    def position(self):
        if self._position is not None:
            return self._position
        raise RuntimeWarning("Node position not defined yet")

    @position.setter
    def position(self, position):
        self._position = position

    def connect(self, child, distance):

        # connections are called one time for couple
        link = uniLink(self, child)
        self.synapses.add(link)
        child.synapses.add(link)
        return link

    def __hash__(self):
        return self.n

    def __str__(self):
        desc = f"node {self.n}: \n"
        for child, dist in self.childs.items():
            desc += f"\tchild {child.n} at distance {dist}\n"
        return desc


class uniLink:
    """Link class to handle unidirected links

    Implements an equivalence relationships between tuples:

        L(1,2) ~ L(2,1)

    this is done by a symmetric hash function and a __eq__ override.
    """

    def __init__(self, node1, node2):

        # The linked nodes
        self.node1 = node1
        self.node2 = node2

        # The value of activation of the link and its length
        self.activation = None
        self.length = None

        # Graphical object to plot
        self.line = None
    
    def __eq__(self, other):
        """Makes links that has the same vertices equal. """
        identical = (self.node1 == other.node1 and self.node2 == other.node2)
        flipped = (self.node1 == other.node2 and self.node2 == other.node1)
        return identical or flipped
    
    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        """Hash fucntion for dicts and sets.
        
        The function must be symmetric w.r.t. nodes, a possible choice is:

            h(uniLink) = c ( h(node1)  + h(node2) )
        
        where c is a constant.

        Since h(node) = node.n (integer) the number of collision is:

            N_coll = n1 + n2 + 1

        To avoid this the constant c is chosen like:
            
            c = 1/(1 + h1**2 )* 1/(1 + h2**2)
        """
        h1 , h2 = hash(self.node1), hash(self.node2)
        return  hash((h1 + h2) / (h1**2 + 1) / (h2**2 + 1))
    
    def __str__(self):
        return f'uL({self.node1.n},{self.node2.n})'


class uniNetwork:
    """Compositional class for describing networks.
    """

    def __init__(self, nodes):

        self.nodes = {}  # Must be corrected: standard constructor does not work
        self.links = {}
        self.N = None

        self._distanceM = None
        self.linkM = None
        self._targetM = None
        self._targetSM = None

        # for display purposes
        self.repr_dim = 2
        self.scatplot = None

    def add_couple(self, nodes , distance):

        self.linkM[nodes[0].n, nodes[1].n] = True
        self.linkM[nodes[1].n, nodes[0].n] = True
        self.links.add(nodes[0].connect(nodes[1], distance) )
        self._targetM[nodes[0].n, nodes[1].n] = distance
        self._targetM[nodes[1].n, nodes[0].n] = distance

    @classmethod
    def from_sparse(cls, sparse_matrix):
        """generates network from a sparse matrix"""

        # raw init
        net = cls([])

        net._targetSM = np.array(sparse_matrix)
        net.N = int(np.max(net._targetSM.transpose()[:2])) + 1

        net.linkM = np.zeros((net.N, net.N), dtype=np.bool)
        net._targetM = np.zeros((net.N, net.N), dtype=np.float32)

        # gets a sparse matrix like (i,j) dist
        # and create nodes
        for i, j, distance in net.targetSM:

            i, j = int(i), int(j)

            node_in = net.nodes.get(i, Node(i))  # fetch from dict or create
            node_out = net.nodes.get(j, Node(j))  # fetch from dict or create
            net.add_couple((node_in, node_out), distance) # connect and add link to set

            print(f">> linked {i} to {j}", end="\r")

        print(f"Network has {len(net.nodes)} elements and {len(net.links)} links")
        return net

    @classmethod
    def from_adiacence(cls, matrix):
        """generates network from an adiacence matrix"""
        # here checks if the matrix is a good one
        # that is to say square, symmetric and M_ii = 0
        # float comparison: dangerous?
        matrix = np.array(matrix)

        if (matrix != matrix.transpose()).any():
            raise ValueError("Matrix is not symmetric")

        if (matrix.diagonal() != np.zeros(len(matrix))).any():
            raise ValueError("Matrix has non-null diagonal")

        sparseM = utils.matrix_to_sparse(matrix)
        net = Network.from_sparse(sparseM)
        return net

    @classmethod
    def connect(cls, networks, links, distances):
        """connects two networks.

        Args
        ----
            networks : tuple
                a tuple of networks to connect.

            links : list of 2-tuples
                links (net1_node_a, net2_node_b). -1 stands for densely connected.

                e.g, (5, -1) connects every element of net2 to element 5 of net1.
        """
        raise NotImplementedError("I have to finish this")
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

    @property
    def values(self):
        return [node.value for node in self.nodes.values()]

    @property
    def activations(self):
        activations = np.array([])
        for node in self.nodes.values():
            for child in node.childs:
                activations = np.append(activations, int(node.synapsis[child]))
        return activations

    @property
    def distanceM(self):
        self._distanceM = np.zeros((self.N, self.N))
        for node in self.nodes.values():
            for another_node in self.nodes.values():
                self._distanceM[node.n, another_node.n] = np.sqrt(
                    np.sum((node.position - another_node.position) ** 2)
                )
        return self._distanceM

    @property
    def targetM(self):
        return self._targetM

    @property
    def distanceSM(self):
        nlinks = np.sum(self.linkM.astype(np.int))
        self._distanceSM = np.array([])
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.linkM[i, j]:
                    self._distanceSM = np.append(
                        self._distanceSM, [i, j, self._distanceM[i, j]]
                    )
        self._distanceSM = self._distanceSM.reshape((-1, 3))
        return self._distanceSM

    @distanceSM.setter
    def distanceSM(self, smatrix):
        self._distanceSM = smatrix

    @property
    def targetSM(self):
        list = []
        for i, j, d in self._targetSM:
            list.append([int(i), int(j), d])
        return list

    @property
    def distortion(self):
        return np.sum(
            ((self._targetM - self.distanceM) * self.linkM.astype(np.float64)) ** 2
        )

    def edges_as_couples(self):
        """returns list of

        [x1,x2], [y1,y2], [z1,z2]

        """
        # for the future: this part is not intelligent and
        # ultra redundant, find a better structure using links
        edges = []
        for node in self.nodes.values():
            for child in node.childs:
                edges.append(np.vstack((node.position, child.position)).transpose())
        return np.array(edges)

    def cMDE(self, Nsteps=1000):
        cnets.MDE(0.1, Nsteps)
        positions = cnets.get_positions()
        for node, position in zip(self.nodes.values(), positions):
            node.position = np.array(position)

    def to_scatter(self):
        return np.array([node.position for node in self.nodes.values()]).transpose()

    def print_distanceM(self, target=False):
        M = self._targetM if target else self.distanceM
        title = "Target matrix" if target else "Current matrix "
        print(title + (30 - len(title)) * "-" + f"(D = {self.distortion:.1e})")
        for i in range(self.N):
            for j in range(self.N):
                color = "green" if self.linkM[i, j] else "red"
                attrs = ["dark"] if i == j else ["bold"]
                print(colored(f"{M[i,j]:.2}", color, attrs=attrs), end="\t")
            print()
        print()

    @classmethod
    def Random(cls, number_of_nodes, connection_probability, max_dist=1.0):
        """Random network constructor

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
        """
        print("Initializing..", end="\r")
        M = np.random.uniform(0, 1, size=number_of_nodes ** 2).reshape(
            (-1, number_of_nodes)
        )
        M = 0.5 * (M + M.transpose())
        np.fill_diagonal(M, 1.0)
        links = (M < connection_probability).astype(float)
        M = M * links * max_dist

        return Network.from_adiacence(M)

    def __str__(self):
        """Describes the tree"""
        desc = "Network ---------\n"
        for node in self.nodes.values():
            desc += str(node)
        return desc
