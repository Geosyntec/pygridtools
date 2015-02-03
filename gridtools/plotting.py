
import numpy as np
import matplotlib.pyplot as plt

import iotools

def checkAx(ax):
    '''
    Pass in an Axes or None, get back an Axes and a Figure
    '''
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    return fig, ax


def plotReachDF(reach, ax=None, flip=False):
    fig, ax = checkAx(ax)
    ax.plot(reach.x, reach.y, 'k-')
    return fig


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
    N = plt.Normalize(vmin=grid.elev.min(), vmax=grid.elev.max())

    for ii in range(grid.nx-1):
        for jj in range(grid.ny-1):
            if isinstance(grid.x_vert, np.ndarray) or not np.any(grid.x_vert.mask[jj:jj+2, ii:ii+2]):
                coords = iotools.makeQuadCoords(
                    xarr=grid.x_vert[jj:jj+2, ii:ii+2],
                    yarr=grid.y_vert[jj:jj+2, ii:ii+2],
                )
                if hasattr(grid, 'elev'):
                    facecolor = cm(N(grid.elev[jj,ii]))
                else:
                    facecolor = 'cornflowerblue'

                if coords is not None:
                    rect = plt.Polygon(coords, edgecolor='k', linewidth=0.25, zorder=0,
                                       facecolor=facecolor, alpha=0.5)
                    ax.add_artist(rect)

    return fig, ax
