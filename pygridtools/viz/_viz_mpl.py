import numpy as np
import matplotlib.pyplot as plt
import pandas

from pygridtools import misc
from pygridtools import qa


def _plot_domain(domain_x=None, domain_y=None, beta=None, data=None, ax=None):
    """ Plot a model grid's domain.

    Parameters
    ----------
    x, y, beta : str or array-like, optional
        Either column labels or sequences representing the x- and
        y-coordinates and beta values of each point defining the domain
        of the model grid.
    data : pandas.DataFrame, optional
        If ``x``, ``y``, and ``beta`` are strings (column labels),
        ``data`` must be a pandas.DataFrame containing those labels.
    ax : matplotib.Axes, optional
        The Axes on which the domain will be drawn. If omitted, a new
        one will be created.

    """
    # setup the figure
    fig, ax = qa._check_ax(ax)

    # coerce values into a dataframe if necessary
    if data is not None:
        domain_x, domain_y = data[domain_x], data[domain_y]
        if beta is not None:
            beta = data[beta]

    # plot the boundary as a line
    ax.plot(domain_x, domain_y, 'k-', label='__nolegend__')

    if beta is not None:
        # plot negative turns
        beta_neg1 = beta == -1
        ax.plot(domain_x[beta_neg1], domain_y[beta_neg1], 's',
                linestyle='none', label="negative")

        # plot non-turns
        beta_zero = beta == 0
        ax.plot(domain_x[beta_zero], domain_y[beta_zero], 'o',
                linestyle='none', label="neutral")

        # plot positve turns
        beta_pos1 = beta == 1
        ax.plot(domain_x[beta_pos1], domain_y[beta_pos1], '^',
                linestyle='none', label="positive")

        ax.legend()

    ax.margins(0.1, 0.1)
    return fig


def _plot_boundaries(extent_x=None, extent_y=None, extent=None, islands_x=None,
                     islands_y=None, islands_name=None, islands=None, ax=None):
    """
    """

    # setup the figure
    fig, ax = qa._check_ax(ax)

    if extent is not None:
        extent_x, extent_y = extent[extent_x], extent[extent_y]
        
    if extent_x is not None:
        ax.plot(extent_x, extent_y, 'k-', label='extent')

    if islands is not None:
        islands_x, islands_y = islands[islands_x], islands[islands_y]

        if islands_name is not None:
            islands_name = islands[islands_name]

    if islands_x is not None and islands_y is not None:
        for name in np.unique(islands_name):
            subset = islands_name == name
            coords = list(zip(islands_x[subset], islands_y[subset]))
            patch = plt.Polygon(coords, facecolor='0.25')
            ax.add_patch(patch)

    ax.margins(0.1, 0.1)
    return fig


def _plot_points(x, y, ax=None, **plot_opts):
    """

    """

    fig, ax = qa._check_ax(ax)
    ax.plot(x, y, 'ko', **plot_opts)
    ax.margins(0.1, 0.1)
    return fig


def _plot_cells(x, y, mask=None, ax=None, **plot_opts):
    fig, ax = qa._check_ax(ax)

    rows, cols = x.shape
    if mask is None:
        if hasattr(x, 'mask'):
            mask = x.mask
        else:
            mask = np.zeros(x.shape)

    for jj in range(rows - 1):
        for ii in range(cols - 1):
            if mask[jj, ii]:
                coords = None

            else:
                coords = misc.makePolyCoords(
                    x[jj:jj+2, ii:ii+2],
                    y[jj:jj+2, ii:ii+2],
                )

            if coords is not None:
                rect = plt.Polygon(coords, edgecolor='0.125', linewidth=0.75,
                                   zorder=0, facecolor='0.875')
                ax.add_patch(rect)

    ax.margins(0.1, 0.1)

    return fig
