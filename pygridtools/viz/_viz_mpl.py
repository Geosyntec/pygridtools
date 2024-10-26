import numpy
from matplotlib import patches

from pygridtools import validate


def _plot_domain(domain, betacol='beta', ax=None, show_legend=True):
    """ Plot a model grid's domain.

    Parameters
    ----------
    domain : GeoDataFrame
        Defines the boundary of the model area. Needs to have a column of
        Point geometries and a a column of beta (turning point) values.
    betacol : str
        Label of the column in *domain* that contains the beta (i.e., turning
        point) values of domain. This sum of this column must be 4.
    ax : matplotib.Axes, optional
        The Axes on which the domain will be drawn. If omitted, a new
        one will be created.
    show_legend : bool, default = True

    """
    leg = None

    # setup the figure
    fig, ax = validate.mpl_ax(ax)

    # plot the boundary as a line
    line_artist, = ax.plot(domain.geometry.x, domain.geometry.y, 'k-', label='__nolegend__')

    beta_artists = []
    beta_selectors = [domain[betacol] < 0, domain[betacol] == 0, domain[betacol] > 0]
    beta_markers = ['s', 'o', '^']
    beta_labels = ['negative', 'neutral', 'positive']
    beta_artists = [
        ax.plot(domain.geometry.x[idx], domain.geometry.y[idx], marker, ls='none', label=label)[0]
        for idx, marker, label in zip(beta_selectors, beta_markers, beta_labels)
    ]
    if show_legend:
        leg = ax.legend()

    ax.autoscale()
    ax.margins(0.1)
    return fig, {'domain': line_artist, 'beta': beta_artists, 'legend': leg}


def _plot_boundaries(extent_x=None, extent_y=None, extent=None, islands_x=None,
                     islands_y=None, islands_name=None, islands=None, ax=None):
    """
    """

    # setup the figure
    fig, ax = validate.mpl_ax(ax)

    if extent is not None:
        extent_x, extent_y = extent[extent_x], extent[extent_y]

    if extent_x is not None:
        extent, = ax.plot(extent_x, extent_y, 'k-', label='extent')

    if islands is not None:
        islands_x, islands_y = islands[islands_x], islands[islands_y]

        if islands_name is not None:
            islands_name = islands[islands_name]

    polys = []
    if islands_x is not None and islands_y is not None:
        for name in numpy.unique(islands_name):
            _subset = islands_name == name
            _coords = list(zip(islands_x[_subset], islands_y[_subset]))
            _island = patches.Polygon(_coords, facecolor='0.25')
            polys.append(_island)
            ax.add_patch(_island)

    ax.margins(0.1)
    return fig, {'extent': extent, 'islands': polys}


def _plot_points(x, y, ax=None, **plot_opts):
    """

    """
    fig, ax = validate.mpl_ax(ax)
    dots, = ax.plot(x.flatten(), y.flatten(), 'ko', **plot_opts)
    ax.margins(0.1)
    return fig, {'points': dots}


def _plot_cells(x, y, mask=None, colors=None, ax=None, sticky_edges=False,
                **plot_opts):
    fig, ax = validate.mpl_ax(ax)
    ax.use_sticky_edges = sticky_edges

    if mask is None:
        mask = numpy.zeros(x.shape)[1:, 1:]

    if colors is None:
        colors = numpy.ma.masked_array(mask + 1, mask)
        vmin, vmax = 0, 2
    else:
        vmin = plot_opts.pop('vmin', None)
        vmax = plot_opts.pop('vmax', None)

    cell_colors = numpy.ma.masked_array(data=colors, mask=mask)

    ec = plot_opts.pop('edgecolor', None) or plot_opts.pop('ec', '0.125')
    lw = plot_opts.pop('linewidth', None) or plot_opts.pop('lw', 0.75)
    fc = plot_opts.pop('facecolor', None) or plot_opts.pop('fc', '0.875')
    cmap = plot_opts.pop('cmap', 'Greys')

    cells = ax.pcolor(x, y, cell_colors, edgecolors=ec, lw=lw,
                      vmin=vmin, vmax=vmax, cmap=cmap, facecolor=fc,
                      **plot_opts)
    return fig, {'cells': cells}


def _rotate_tick_labels(ax, angle=45):
    for label in ax.get_xticklabels():
        label.set_rotation_mode('anchor')
        label.set_rotation(angle)
        label.set_horizontalalignment('right')
    return ax
