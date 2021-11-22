import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

import network

def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def animate_MDE(net,fig,ax, **kwargs):

    scat = ax.scatter([],[],[], marker='o', color='k')

    for node in net.nodes.values():
        for child in node.childs:
            line = np.vstack((node.position, child.position)).transpose()
            node.lines[child] = ax.plot([],[],[], alpha=0.3, color='k')[0]

    # define function that updates the plot
    def _update_plot(i):

        net.MDE()

        x,y,z = net.to_scatter()
        position_on_plane = tuple(np.vstack((x,y)).transpose())

        # note: set_offsets takes ([x1, y1], [x2, y2], ...) and
        # not ([x1, x2, ..], [y1, y2, ...]) as a good boy would do
        scat.set_offsets(position_on_plane)
        scat.set_3d_properties(z, 'z')

        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(y), np.max(y)))
        ax.set_zlim((np.min(z), np.max(z)))

        for node in net.nodes.values():

            for child in node.childs:
                x,y,z = np.vstack((node.position, child.position)).transpose()
                node.lines[child].set_data(x,y)
                node.lines[child].set_3d_properties(z)

                # old: color to lines
                # color = hsv_to_rgb((0.5, 1., node.n/net.N ))
                # node.lines[child].set_color(color)

    net.animation = animation.FuncAnimation(fig, _update_plot, **kwargs)
    return net.animation

def plotNet(net, ax, label=''):
    if net.repr_dim == 3:
        net.scatplot = ax.scatter(net.to_scatter()[0],net.to_scatter()[1],zs = net.to_scatter()[2],
                    label=label, c=net.colors, cmap='plasma', s=50)

        for node in net.nodes.values():
            for child in node.childs:
                line = np.vstack((child.position, node.position)).transpose()
                for sub_line in ax.plot(*line, color='k', alpha=0.3):
                    node.lines[child] = sub_line

    if net.repr_dim == 2:
        net.scatplot = ax.scatter(*net.to_scatter(),
                    label=label, c=net.colors, cmap='plasma', s=50)

        for node in net.nodes.values():
            for child in node.childs:
                line = np.vstack((child.position, node.position)).transpose()
                for sub_line in ax.plot(*line, color='k', alpha=0.4):
                    node.lines[child] = sub_line

def animate_values(super_net, fig, ax, **kwargs):

    scat = ax.scatter(*super_net.net.to_scatter(), marker='o', s=50, cmap='plasma')

    for node in super_net.net.nodes.values():
        for child in node.childs:
            line = np.vstack((node.position, child.position)).transpose()
            node.lines[child] = ax.plot(*line, alpha=0.3, color='k')[0]

    # define function that updates the plot
    def _update_plot(i):

        super_net.update()

        colors = super_net.net.colors
        scat.set_array(np.array(colors))
        vmin, vmax = min(colors), max(colors)
        scat.set_clim(vmin, vmax)

        for node in super_net.net.nodes.values():
            for child in node.childs:
                active = int(node.synapsis[child])
                color = hsv_to_rgb((0., 1., active ))
                alpha = 1. if active else 0.2
                node.lines[child].set_color(color)
                node.lines[child].set_alpha(alpha)

    super_net.net.animation = animation.FuncAnimation(fig, _update_plot, **kwargs)
    return super_net.net.animation

def plot_links(net):
    fig,ax = plt.subplots()
    ax.imshow(net.linkM.astype(float), cmap='gray')
