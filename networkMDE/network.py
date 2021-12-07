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
        self._value = 0  # the value of... something, I guess?

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
        link.length = distance
        self.synapses += link
        child.synapses += link
        return link

    def __hash__(self):
        return self.n

    def __str__(self):
        return f"N({self.n})"


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
        self.activation = 0
        self.length = None

        # Related graphical objects
        self.line = None

    def get_child(self, node):
        if node == self.node1:
            return self.node2
        elif node == self.node2:
            return self.node1
        else:
            raise ValueError(f"This synapsis has no extremum {node}")

    def __eq__(self, other):
        """Makes links that has the same vertices equal."""
        identical = self.node1 == other.node1 and self.node2 == other.node2
        flipped = self.node1 == other.node2 and self.node2 == other.node1
        return identical or flipped

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Hash function for dicts and sets.

        The function must be symmetric w.r.t. nodes, a possible choice is:

            h(uniLink) = c ( h(node1)  + h(node2) )

        where c is a constant.

        Since h(node) = node.n (integer) the number of collision is:

            N_coll = n1 + n2 + 1

        To avoid this the constant c is chosen like:

            c = 1/(1 + h1**2 )* 1/(1 + h2**2)
        """
        h1, h2 = hash(self.node1), hash(self.node2)
        return hash((h1 + h2) / (h1 ** 2 + 1) / (h2 ** 2 + 1))

    def __str__(self):
        return f"uL({self.node1.n}<->{self.node2.n}:{self.length:1.2f})"


class uniNetwork:
    """Compositional class for describing networks."""

    def __init__(self, nodes):

        # Nodes are contained in a dictionary because are labelled
        # but the insertion is not contiguous
        self.nodes = ci.cdict(nodes)  # TODO: standard constructor does not work

        # Links are contained in a set beacuse it really doesn't make sense
        # to define an ordering of links
        self.links = ci.cset()

        # Number of nodes
        self.N = None

        # Descriptive matrices
        self._distanceM = None
        self.linkM = None
        self._targetM = None
        self._targetSM = None

        # Related graphical objects
        self.repr_dim = 2
        self.scatplot = None

        # cnets parameters
        self.is_cnet_initialized = False

        # self.initialize_embedding(dim=2)

    def initialize_embedding(self, dim=2):
        cnets.init_network(self.targetSM, list(self.nodes.value), dim)
        self.repr_dim = dim
        self.is_cnet_initialized = True

    def add_couple(self, node1, node2, distance):

        self.linkM[node1.n, node2.n] = True
        self.linkM[node2.n, node1.n] = True

        self._targetM[node1.n, node2.n] = distance
        self._targetM[node2.n, node1.n] = distance

        self.nodes += {node1.n: node1, node2.n: node2}
        self.links += node1.connect(node2, distance)

    @classmethod
    def from_sparse(cls, sparse_matrix):
        """generates network from a sparse matrix"""

        # raw init
        net = cls({})

        net._targetSM = np.array(sparse_matrix)
        net.N = int(np.max(net._targetSM.transpose()[:2])) + 1

        net.linkM = np.zeros((net.N, net.N), dtype=np.bool)
        net._targetM = np.zeros((net.N, net.N), dtype=np.float32)

        # gets a sparse matrix like (i,j) dist
        # and create nodes
        print("Linking..                  ", end="\r")
        for i, j, distance in net.targetSM:

            i, j = int(i), int(j)

            net.add_couple(
                net.nodes.get(i, Node(i)), net.nodes.get(j, Node(j)), distance
            )  # connect and add link to set

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
        net = uniNetwork.from_sparse(sparseM)
        return net

    @property
    def distanceM(self):
        self._distanceM = np.zeros((self.N, self.N))
        for node in self.nodes:
            for another_node in self.nodes:
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

    @targetSM.setter
    def targetSM(self, value):
        self._targetSM = value

    @property
    def distortion(self):
        return np.sum(
            ((self._targetM - self.distanceM) * self.linkM.astype(np.float64)) ** 2
        )

    def cMDE(self,step=0.1, Nsteps=1000):
        cnets.MDE(step, Nsteps)
        positions = cnets.get_positions()
        for node, position in zip(self, positions):
            node.position = np.array(position)

    def to_scatter(self):
        return np.array(list(self.nodes.position)).transpose()

    def print_distanceM(self, target=False):
        M = self._targetM if target else self.distanceM
        title = "Target matrix" if target else "Current matrix "
        print(title + 30 * "-")
        for i in range(self.N):
            for j in range(self.N):
                color = "green" if self.linkM[i, j] else "red"
                attrs = ["dark"] if i == j else ["bold"]
                print(colored(f"{M[i,j]:.2}", color, attrs=attrs), end="\t")
            print()
        print()

    def update_target_matrix(self):
        print("Pynet - updating targets")
        for node in self.nodes:
            for link in node.synapses:
                other = link.get_child(node)
                self._targetM[node.n, other.n] = link.length
                self._targetM[other.n, node.n] = link.length
        self.targetSM = utils.matrix_to_sparse(self.targetM)

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
        print("Initializing random network...", end="\r")
        M = np.random.uniform(0, 1, size=number_of_nodes ** 2).reshape(
            (-1, number_of_nodes)
        )
        M = 0.5 * (M + M.transpose())
        np.fill_diagonal(M, 1.0)
        links = (M < connection_probability).astype(float)
        M = M * links * max_dist

        return uniNetwork.from_adiacence(M)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.N - 1:
            raise StopIteration
        self.i += 1
        return self.nodes[self.i - 1]

    def __str__(self):
        """Describes the tree"""
        desc = "Network ---------\n"
        for node in self.nodes.values():
            desc += str(node)
        return desc
