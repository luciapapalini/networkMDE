'''Graphic module for networks.
Most things are not elegant/efficient, in y defense I have to say
that matplotlib has the most strange way to manage Path3Dcollections and Line3D.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

import network

def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))

def get_graphics(net):

    #creates figure and axis
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(projection='3d') if net.repr_dim==3 else fig.add_subplot()

    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])

    ax.axis('off')

    empty = [[],[], []] if ax.name == '3d' else [[],[]]
    kwargs = {'cmap':'plasma', 's':50}
    if ax.name != '3d':
        kwargs['zorder']=10
    # plots points
    net.scatplot = ax.scatter(*empty, **kwargs)

    # plots each line (twice, probably can be solved using Link class)
    artists = (net.scatplot,)

    for node in net.nodes.values():
        for child in node.childs:
            line_data = np.vstack((child.position, node.position)).transpose()
            # line, = ax.plot(*line_data, color='k', alpha=0.3)
            line, = ax.plot(*empty, color='k', alpha=0.3)
            node.lines[child] = line
            artists += (line,)

    return fig, ax, artists

def update_scatter(ax, scat, position, colors, normalize_colors=True):
    '''Updates the scatter plot position and colors

    Args
    ----
        ax : axis

        scat : scatter plot

        position : np.ndarray

        color : np.ndarray


    Note
    ----
        position must be given in the following format:

        position = [[x1, y1, z1], [x2, y2, z2]]
    '''
    if ax.name == '3d':
        x,y,z = position
    else:
        x,y = position
    # NOTE ABOUT Path3Dcollection (matplotib v3.5.0)
    # set_offsets takes ([x1, y1], [x2, y2], ...) and
    # not ([x1, x2, ..], [y1, y2, ...]) as a good boy would do
    # also, z coordinates must be set AFTER the x-y coordinates
    position_on_plane = tuple(np.vstack((x,y)).transpose())
    scat.set_offsets(position_on_plane)

    min_, max_ = np.min(x), np.max(x)
    min_, max_ = min_ - 0.1*(max_ - min_), max_ + 0.1*(max_ - min_),
    ax.set_xlim((min_, max_))

    min_, max_ = np.min(y), np.max(y)
    min_, max_ = min_ - 0.1*(max_ - min_), max_ + 0.1*(max_ - min_),
    ax.set_ylim((min_, max_))

    if ax.name == '3d':
        scat.set_3d_properties(z, 'z')
        ax.set_zlim((np.min(z), np.max(z)))

    #colors
    scat.set_array(np.array(colors))
    if normalize_colors:
        vmin, vmax = min(colors), max(colors)
        scat.set_clim(vmin, vmax)

def update_lines(ax, lines, positions, colors, alphas):
    '''updates the lines of the network

    Args
    ----
        ax : matplotlib.Axis

        lines : tuple of matplotlib lines

        position : np.ndarray

    Note
    ----
        position must be given in the following format:

        position = [ [[x1,x2] , [y1,y2] , [z1,z2]] ]
    '''

    for position, line, color, alpha in zip(positions, lines, colors, alphas):
        if ax.name == '3d':
            x,y,z = position
        else:
            x,y = position
        line.set_data(x,y)
        line.set_color(color)
        line.set_alpha(alpha)

        if ax.name == '3d':
            line.set_3d_properties(z)


def animate_super_network(super_net, super_net_function, **anim_kwargs):

    fig, ax, (scat, *lines) = get_graphics(super_net.net)

    def _update_graphics(i):

        super_net_function()
        positions = super_net.net.to_scatter()
        values = super_net.net.values
        activations = super_net.net.activations

        point_colors = values
        line_colors = np.array([hsv_to_rgb((0., 1., a )) for a in activations])
        line_alpha  = [.2+ .8*a for a in activations]

        line_data = super_net.net.edges_as_couples()

        update_scatter(ax, scat, positions, point_colors)
        update_lines(ax, tuple(lines), line_data,
                        line_colors, line_alpha)

        return (scat,) + tuple(lines)

    super_net.net.animation = animation.FuncAnimation(fig, _update_graphics, **anim_kwargs)
    return super_net.net.animation


def plot_links(net):
    fig,ax = plt.subplots()
    ax.imshow(net.linkM.astype(float), cmap='gray')
