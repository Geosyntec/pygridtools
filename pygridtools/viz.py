from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn

from bokeh.sampledata import us_counties, unemployment
from bokeh import plotting
from bokeh.models import HoverTool

from . import io


def checkAx(ax):
    '''
    Pass in an Axes or None, get back an Axes and a Figure
    '''
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    return fig, ax


def plotReachDF(boundary, xcol, ycol, reachcol, flip=False):
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

    cm = plt.cm.coolwarm
    cm.set_bad('k')

    for ii in range(grid.nx-1):
        for jj in range(grid.ny-1):
            if isinstance(grid.x_vert, np.ndarray) or not np.any(grid.x_vert.mask[jj:jj+2, ii:ii+2]):
                coords = io.makeQuadCoords(
                    xarr=grid.x_vert[jj:jj+2, ii:ii+2],
                    yarr=grid.y_vert[jj:jj+2, ii:ii+2],
                )
                if hasattr(grid, 'elev'):
                    N = plt.Normalize(vmin=grid.elev.min(), vmax=grid.elev.max())
                    facecolor = cm(N(grid.elev[jj,ii]))
                else:
                    facecolor = 'cornflowerblue'

                if coords is not None:
                    rect = plt.Polygon(coords, edgecolor='k', linewidth=0.25, zorder=0,
                                       facecolor=facecolor, alpha=0.5)
                    ax.add_artist(rect)

    return fig, ax


def plotCells(nodes_x, nodes_y, name='test', engine='bokeh',
              ax=None):
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
        fig = _plot_cells_mpl(nodes_x, nodes_y, name=name, ax=ax)
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


def _plot_cells_mpl(nodes_x, nodes_y, name='test', ax=None):
    fig, ax = checkAx(ax)

    if not nodes_x.shape == nodes_y.shape:
        raise ValueError()

    ny = nodes_x.shape[0]
    nx = nodes_y.shape[1]

    if ax.is_first_col():
        ax.set_ylabel('$ny = {}$'.format(ny), size=14)

    if ax.is_last_row():
        ax.set_xlabel('$nx = {}$'.format(nx), size=14)

    for ii in range(nx - 1):
        for jj in range(ny - 1):
            if not np.any(nodes_x.mask[jj:jj+2, ii:ii+2]):
                coords = io.makeQuadCoords(
                    xarr=nodes_x[jj:jj+2, ii:ii+2],
                    yarr=nodes_y[jj:jj+2, ii:ii+2],
                )

                if coords is not None:
                    rect = plt.Polygon(coords, edgecolor='k', linewidth=0.25,
                                       zorder=0, facecolor='0.75', alpha=0.5)
                    ax.add_artist(rect)

    return fig
