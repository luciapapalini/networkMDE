"""simple model t do something with the network

Each node has a value V_n, I want to study the propagation of a signal over the
network. Possible ways:

- each node transmit to the child only if V_child < V_node, with probability d_cn/(sum_c d_cn)

the fact is that 'child' is a reflective property, so net will reach an equilibrium value like
the mean of the initial values.
"""

import numpy as np
import matplotlib.pyplot as plt
from networkMDE import network as nw
from networkMDE import netplot

from termcolor import colored
from timeit import default_timer as time

import cnets


class propagateNet(nw.uniNetwork):
    def __init__(self):
        np.random.seed(31121999)
        cnets.set_seed(31121997)
        self.net = nw.uniNetwork.Random(48, 1)
        for node in self.net:
            for link in node.synapses:
                link.length = np.float32(1.) # must be done using numpy types
        self.net.update_target_matrix()
        # self.net.nodes[0].value = 11

        self.net.initialize_embedding(dim=2)
        # self.net.cMDE(step=0.2, neg_step=0.0, Nsteps=10000)

        self.updated_times = 0
        self.step = 0.1
        print("Initialization terminated")

    def apple_game(self):
        """Toy model for saturation of a market.
        
        The length of the link says how prone is the seller to sell to the buyer
        (shorter = more prone).

        At the beginning one marketer has all the benefits, then the selling starts.
        """
        if self.updated_times < 100:
            for node in self.net:
                Ztot = np.sum(1./np.array(list(node.synapses.length))) # classiter at work
                p_sum = 0
                for link in node.synapses:
                    child = link.get_child(node)
                    p = 1.0/np.array(link.length)/Ztot
                    if np.random.uniform(0, 1) < p:
                        child.value += 1
                        node.value -= 1
                        link.activation = p
                        # link.length *= 1./link.length + p*(1.-1./link.length)   # synapsis enhancement/ distance reduction
                        
                    else:
                        link.activation = np.float32(0.)
        if self.updated_times == 100:
            for link in self.net.links:
                link.activation = np.float32(0)

        self.net.update_target_matrix()
        cnets.set_target(self.net.targetSM)
        self.updated_times += 1
        print(colored(f"updated times: {self.updated_times}","blue"))


    def update(self):
        # if int(self.updated_times) % 5 == 0:
        #     self.apple_game()
        self.net.cMDE(step=self.step,neg_step=0.0, Nsteps=5)
        self.updated_times += 1


A = propagateNet()
# A.net.print_distanceM()
netplot.plot_lines = False
# netplot.plot_net(A.net)
netplot.scat_kwargs['cmap'] = 'viridis'
# plt.show()
animation = netplot.animate_super_network(A, A.update,
                                            frames=200, interval=60, blit=False)
animation.save('cnetsa.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r'), dpi=80)
# plt.show()
