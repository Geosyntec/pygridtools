from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn

from bokeh import plotting
from bokeh.models import HoverTool

from . import misc


def checkAx(ax):
    '''
    Pass in an Axes or None, get back an Axes and a Figure
    '''
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    return fig, ax


def plotReachDF(boundary, xcol, ycol, flip=False):
    if not isinstance(boundary, pandas.DataFrame):
        raise ValueError('`boundary` must be a dataframe')

    fg = seaborn.FacetGrid(data=boundary, legend_out=False, size=5,
                           hue='beta', hue_kws={'marker': ['s', 'o', '^']})
    fg.axes[0, 0].plot(boundary[xcol], boundary[ycol], 'k-')
    fg.map(plt.plot, xcol, ycol, linestyle='none')
    fg.add_legend()
    return fg


def plotPygridgen(grid, ax=None):
    fig, ax = checkAx(ax)
    ax.plot(grid.xbry, grid.ybry, 'k-', label='boundary',
            zorder=10, alpha=0.5, linewidth=1.5)

    if ax.is_first_col():
        ax.set_ylabel('$ny = {}$'.format(grid.ny), size=14)

    if ax.is_last_row():
        ax.set_xlabel('$nx = {}$'.format(grid.nx), size=14)

    fig, ax = plotCells(grid.x, grid.y, ax=ax, engine='mpl')
    return fig, ax


def plotCells(nodes_x, nodes_y, name='test', engine='bokeh',
              mask=None, ax=None):
    nodes_x = np.ma.asarray(nodes_x)
    nodes_y = np.ma.asarray(nodes_y)
    if not np.all(nodes_x.shape == nodes_y.shape):
        raise ValueError("nodes_x and nodes_y must be the same shape")

    if not np.all(nodes_x.mask == nodes_y.mask):
        raise ValueError("node arrays must have identical masks")

    if engine.lower() == 'bokeh':
        p = _plot_cells_bokeh(nodes_x, nodes_y, name=name)
        return p

    elif engine.lower() in ('mpl', 'matplotlib', 'sns', 'seaborn'):
        fig = _plot_cells_mpl(nodes_x, nodes_y, ax=ax, mask=mask)
        return fig

    else:
        raise NotImplementedError("'{}' is not a valid engine".format(engine))


def _plot_cells_bokeh(nodes_x, nodes_y, name='test'):

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
    fig, ax = checkAx(ax)

    rows, cols = nodes_x.shape
    if mask is None:
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
    fig, ax = checkAx(ax)
    if engine == 'mpl':
        _plot_boundaries_mpl(river, islands, ax=ax)
        ax.set_aspect('equal')
        return fig, ax

    elif engine == 'bokeh':
        raise NotImplementedError("bokeh engine not ready yet")

    else:
        raise ValueError("only 'mpl' and 'bokeh' plotting engines available")


def _plot_boundaries_mpl(river=None, islands=None, ax=None):
    if river is not None:
        ax.plot(river[:, 0], river[:, 1], '-')

    for island in islands:
        ax.plot(island[:, 0], island[:, 1], '-')
