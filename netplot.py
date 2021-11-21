import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

import network

def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def animate_MDE(net,fig,ax, **kwargs):

    scat = ax.plot(*net.to_scatter(), ls='', marker='o', color='k')[0]

    for node in net.nodes.values():
        for child in node.childs:
            line = np.vstack((node.position, child.position)).transpose()
            node.lines[child] = ax.plot(*line, alpha=0.3)[0]

    # define function that updates the plot
    def _update_plot(i):

        net.MDE()

        x,y,z = net.to_scatter()

        scat.set_data(x,y)
        scat.set_3d_properties(z)

        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(y), np.max(y)))
        ax.set_zlim((np.min(z), np.max(z)))

        for node in net.nodes.values():

            for child in node.childs:
                x,y,z = np.vstack((node.position, child.position)).transpose()
                node.lines[child].set_data(x,y)
                node.lines[child].set_3d_properties(z)

                color = hsv_to_rgb((0.5, 1., node.n/net.N ))
                node.lines[child].set_color(color)

    net.animation = animation.FuncAnimation(fig, _update_plot, **kwargs)
    return net.animation

def plot(net, ax, label=''):
    if net.repr_dim == 3:
        net.scatplot = ax.scatter(net.to_scatter()[0],net.to_scatter()[1],zs = net.to_scatter()[2],
                    label=label, c=net.colors, cmap='plasma', s=50)

        for node in net.nodes.values():
            for child in node.childs:
                line = np.vstack((child.position, node.position)).transpose()
                for sub_line in ax.plot(*line, color='k', alpha=0.4):
                    node.lines[child] = sub_line

    if net.repr_dim == 2:
        net.scatplot = ax.scatter(*net.to_scatter(),
                    label=label, c=net.colors, cmap='plasma', s=50)

        for node in net.nodes.values():
            for child in node.childs:
                line = np.vstack((child.position, node.position)).transpose()
                for sub_line in ax.plot(*line, color='k', alpha=0.4):
                    node.lines[child] = sub_line

def plot_links(net):
    fig,ax = plt.subplots()
    ax.imshow(net.linkM.astype(float), cmap='gray')
