from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas

from . import misc


def _check_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    return fig, ax


def plotReachDF(boundary, xcol='x', ycol='y', betacol='beta', ax=None):
    """ Plots the reach (domain) of a grid from the boundary coordinates
    stored in a pandas.DataFrame.

    Parameters
    ----------
    boundary : pandas.DataFrame
        Dataframe containing x, y, and beta information about the grid
        domain.
    xcol, ycol, betacol : str, optional
        Strings defining the column labels in ``boundary`` for the
        x- and y-coordinates, and the beta (turning point parameter).
    ax : matplotlib Axes, optional
        Axes on which the data will be plotted. If None, a new one will
        be created.

    Returns
    -------
    fig : matplotlib.Figure

    See Also
    --------
    misc.makeGrid (for info re: the beta parameter)

    """

    if not isinstance(boundary, pandas.DataFrame):
        raise ValueError('`boundary` must be a dataframe')

    # setup the figure
    fig, ax = _check_ax(ax)

    # plot the boundary as a line
    ax.plot(boundary[xcol], boundary[ycol], 'k-', label='__nolegend__')

    # plot negative turns
    beta_neg1 = boundary[boundary['beta'] == -1]
    ax.plot(beta_neg1[xcol], beta_neg1[ycol], 's', linestyle='none')

    # plot non-turns
    beta_zero = boundary[boundary['beta'] ==  0]
    ax.plot(beta_zero[xcol], beta_zero[ycol], 'o', linestyle='none')

    # plot positve turns
    beta_pos1 = boundary[boundary['beta'] ==  1]
    ax.plot(beta_pos1[xcol], beta_pos1[ycol], '^', linestyle='none')

    return fig


def plotPygridgen(grid, ax=None):
    """ Plots a pygridgen' object's boundary and cells.

    Parameters
    ----------
    grid : pygridgen.Grid object
        The grid object to be visualized
    ax : matplotlib Axes, optional
        Axes on which the data will be plotted. If None, a new one will
        be created.

    Returns
    -------
    fig : matplotlib.Figure

    """

    # setup the figure
    fig, ax = _check_ax(ax)

    # coerce the boundary info into a dataframe and plot
    boundary = pandas.DataFrame({
        'x': grid.xbry,
        'y': grid.ybry,
        'beta': grid.beta
    }).pipe(plotReachDF, ax=ax)


    if ax.is_first_col():
        ax.set_ylabel('$ny = {}$'.format(grid.ny), size=14)

    if ax.is_last_row():
        ax.set_xlabel('$nx = {}$'.format(grid.nx), size=14)

    fig = plotCells(grid.x, grid.y, ax=ax, engine='mpl')

    return fig


def plotCells(nodes_x, nodes_y, name=None, engine='mpl', mask=None, ax=None):
    """ Plots a grid's cells.

    Parameters
    ----------
    nodes_x, nodes_y : array-like
        Arrays of x- and y- coordindates defining the nodes of a grid.
    name : str, optional
        Name for the grid to be displayed in the figure's title
        (bokeh only).
    engine : str (default = 'mpl')
        The plotting engine used to draw the figure. Use 'mpl' for
        static images. In the __future__, 'bokeh' will be an option for
        interactive charts.
    mask : array-like of bools, optional
        Mask that can be applyed to ``nodes_x`` and ``nodes_y``.
    ax : matplotlib Axes, optional
        Axes on which the data will be plotted. If None, a new one will
        be created.

    Returns
    -------
    fig : matplotlib.Figure

    """

    nodes_x = np.ma.asarray(nodes_x)
    nodes_y = np.ma.asarray(nodes_y)
    if not np.all(nodes_x.shape == nodes_y.shape):
        raise ValueError("nodes_x and nodes_y must be the same shape")

    if not np.all(nodes_x.mask == nodes_y.mask):
        raise ValueError("node arrays must have identical masks")

    # pragma: no cover
    if engine.lower() == 'bokeh':
        msg = "'bokeh' is not an implemented engine (yet)".format(engine)
        raise NotImplementedError(msg)
        p = _plot_cells_bokeh(nodes_x, nodes_y, name=name)
        return p

    elif engine.lower() in ('mpl', 'matplotlib', 'sns', 'seaborn'):
        fig, ax = _plot_cells_mpl(nodes_x, nodes_y, ax=ax, mask=mask)
        return fig

    else:
        raise ValueError("'{}' is not a valid engine".format(engine))


def _plot_cells_bokeh(nodes_x, nodes_y, name='test'): # pragma: no cover
    raise NotImplementedError

    def getCellColor(row, col):
        return 'blue'

    cell_x = []
    cell_y = []
    cell_color = []

    for row in range(nodes_x.shape[0] - 1):
        for col in range(nodes_x.shape[1] - 1):
            top_x = nodes_x[row:row+2, col]
            bot_x = nodes_x[row:row+2, col+1][::-1]
            cell_x.append(np.hstack([top_x, bot_x]).tolist())

            top_y = nodes_y[row:row+2, col]
            bot_y = nodes_y[row:row+2, col+1][::-1]
            cell_y.append(np.hstack([top_y, bot_y]).tolist())
            cell_color.append(getCellColor(row, col))

    plotting.output_file("{}.html".format(name.replace(' ', '_')),
                        title="{} example".format(name))

    TOOLS = "resize,pan,wheel_zoom,box_zoom,reset,hover,save"

    p = plotting.figure(title="Test Grid Viz", tools=TOOLS)
    # if boundary is not None:
    #     p.patches([boundary.x.values], [boundary.y.values],
    #               line_color='black', fill_color='black')
    p.patches(cell_x, cell_y, fill_color="blue", fill_alpha=0.7,
              line_color="white", line_width=0.5)

    hover = p.select(dict(type=HoverTool))
    hover.snap_to_data = False
    hover.tooltips = OrderedDict([
        ("index", "$index/250."),
        ("(x,y)", "($x, $y)"),
        ("fill color", "$color[hex, swatch]:fill_color"),
    ])

    return p


def _plot_cells_mpl(nodes_x, nodes_y, mask=None, ax=None):
    fig, ax = _check_ax(ax)

    rows, cols = nodes_x.shape
    if mask is None:
        if hasattr(nodes_x, 'mask'):
            mask = nodes_x.mask
        else:
            mask = np.zeros(nodes_x.shape)

    for jj in range(rows - 1):
        for ii in range(cols - 1):
            if mask[jj, ii]:
                coords = None

            else:
                coords = misc.makePolyCoords(
                    nodes_x[jj:jj+2, ii:ii+2],
                    nodes_y[jj:jj+2, ii:ii+2],
                )

            if coords is not None:
                rect = plt.Polygon(coords, edgecolor='0.125', linewidth=0.75,
                                   zorder=0, facecolor='0.875')
                ax.add_artist(rect)

    return fig, ax


def plotBoundaries(river=None, islands=None, engine='mpl', ax=None):
    """ Plots a grid's boundary.

    Parameters
    ----------
    river : array-like
        An N x 2 array of x-y coordinates that define the river. Cells
        whose centroids are outside the river will not be drawn.
    islands : list of array-likes
        A list of N x 2 arrays that define islands in the river. Cells
        whose centroids are inside the islands will not be drawn.
    engine : str (default = 'mpl')
        The plotting engine used to draw the figure. Use 'mpl' for
        static images. In the __future__, 'bokeh' will be an option for
        interactive charts.
    ax : matplotlib Axes, optional
        Axes on which the data will be plotted. If None, a new one will
        be created.

    Returns
    -------
    fig : matplotlib.Figure

    """
    fig, ax = _check_ax(ax)
    if engine == 'mpl':
        _plot_boundaries_mpl(river, islands, ax=ax)
        ax.set_aspect('equal')
        return fig

    elif engine == 'bokeh':
        raise NotImplementedError("bokeh engine not ready yet")

    else:
        raise ValueError("only 'mpl' and 'bokeh' plotting engines available")


def _plot_boundaries_mpl(river=None, islands=None, ax=None):
    if river is not None:
        ax.plot(river[:, 0], river[:, 1], '-')

    for island in islands:
        ax.plot(island[:, 0], island[:, 1], '-')
