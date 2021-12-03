"""Graphic module for networks.
Most things are not elegant/efficient, in my defense I have to say
that matplotlib has the most strange way to manage Path3Dcollections and Line3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

# import network

# Default setttings for plots
fig_kwargs = {"figsize": (4, 4)}
scat_kwargs = {"cmap": "plasma", "s": 50, "alpha": 1.0}
line_kwargs = {"color": "k", "alpha": 0.3, "lw": 0.1}

def get_graphics(net):
    """Generates the figure, axis and empty scatterplot/lines for the net."""

    # creates figure and axis
    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(projection="3d") if net.repr_dim == 3 else fig.add_subplot()

    # length scale is useless (?)
    ax.axis("off")

    empty = [[], [], []] if ax.name == "3d" else [[], []]

    # In 2d plots points over lines if not differently specified
    if ax.name != "3d":
        line_kwargs.setdefault("zorder", 0)
        scat_kwargs.setdefault("zorder", 1)

    # plots points
    net.scatplot = ax.scatter(*empty, **scat_kwargs)
    artists = (net.scatplot,)

    for link in net.links:
        # line_data = np.vstack((link.node1.position, link.node2.position)).transpose()
        # line, = ax.plot(*line_data, color='k', alpha=0.3)
        (line,) = ax.plot(*empty, **line_kwargs)
        link.line = line
        artists += (line,)
    
    return fig, ax


def update_scatter(ax, net, colors, normalize_colors=True):
    """Updates the scatter plot position and colors
    """

    position = net.to_scatter()
    scat = net.scatplot

    if ax.name == "3d":
        x_coord, y_coord, z_coord = position
    else:
        x_coord, y_coord = position

    # NOTE ABOUT Path3Dcollection (matplotib v3.5.0)
    # set_offsets takes ([x1, y1], [x2, y2], ...) and
    # not ([x1, x2, ..], [y1, y2, ...]) as a good boy would do
    # also, z coordinates must be set AFTER the x-y coordinates

    position_on_plane = tuple(np.vstack((x_coord, y_coord)).transpose())
    scat.set_offsets(position_on_plane)

    min_, max_ = np.min(x_coord), np.max(x_coord)
    min_, max_ = min_ - 0.1 * (max_ - min_), max_ + 0.1 * (max_ - min_)
    ax.set_xlim((min_, max_))

    min_, max_ = np.min(y_coord), np.max(y_coord)
    min_, max_ = min_ - 0.1 * (max_ - min_), max_ + 0.1 * (max_ - min_)
    ax.set_ylim((min_, max_))

    if ax.name == "3d":
        scat.set_3d_properties(z_coord, "z")
        ax.set_zlim((np.min(z_coord), np.max(z_coord)))

    # colors
    scat.set_array(np.array(colors))
    if normalize_colors:
        vmin, vmax = min(colors), max(colors)
        scat.set_clim(vmin, vmax)


def update_lines(ax, net, colors, alphas):
    """updates the lines of the network
    """
    for link, color, alpha in zip(net.links, colors, alphas):

        line_data = np.vstack((link.node1.position, link.node2.position)).transpose()

        if ax.name == "3d":
            x_coord, y_coord, z_coord = line_data
        else:
            x_coord, y_coord = line_data

        link.line.set_data(x_coord, y_coord)
        link.line.set_color(color)
        link.line.set_alpha(alpha)

        if ax.name == "3d":
            link.line.set_3d_properties(z_coord)


def animate_super_network(super_net, super_net_function, **anim_kwargs):
    """Animates the super_network values evolution function

    Note
    ----
        The update function can include an MDE update on position
    """
    raise NotImplementedError('must be made compatible with classiter')
    fig, ax, (scat, *lines) = get_graphics(super_net.net)

    def _update_graphics(_):

        super_net_function()
        positions = super_net.net.to_scatter()
        values = super_net.net.values
        activations = super_net.net.activations

        point_colors = values
        line_colors = np.array([hsv_to_rgb((0.0, 1.0, a)) for a in activations])
        line_alpha = [0.2 + 0.8 * a for a in activations]

        line_data = super_net.net.edges_as_couples()

        update_scatter(ax, scat, positions, point_colors)
        update_lines(ax, tuple(lines), line_data, line_colors, line_alpha)

        return (scat,) + tuple(lines)

    super_net.net.animation = animation.FuncAnimation(
        fig, _update_graphics, **anim_kwargs
    )
    return super_net.net.animation


def plot_net(net):
    """Plots a statical image for the network embedding"""
    
    _, ax = get_graphics(net)

    activations = net.links.activation
    point_colors = net.nodes.value

    line_colors = np.array([hsv_to_rgb((0.0, 1.0, a)) for a in activations])
    line_alpha = [0.2 + 0.8 * a for a in activations]

    update_scatter(ax, net, point_colors)
    update_lines(ax, net, line_colors, line_alpha)


def plot_links(net):

    _, ax = plt.subplots()
    ax.imshow(net.linkM.astype(float), cmap="gray")
