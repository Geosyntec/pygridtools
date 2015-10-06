import numpy as np
import matplotlib.pyplot as plt


def _check_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_aspect('equal')
    return fig, ax


def _validate_polygon(polyverts, min_points=3):
    polyverts_array = np.asarray(polyverts)
    if polyverts_array.ndim != 2:
        raise ValueError('polyverts must be a 2D array, or '
                         'similar sequence')

    if polyverts_array.shape[1] != 2:
        raise ValueError('polyverts must be two columns of points')

    if polyverts_array.shape[0] < min_points:
        raise ValueError('polyverts must contain at least {} points'.format(min_points))

    return polyverts_array


def _validate_xy_array(x, y, as_pairs=True):
    x, y = np.asanyarray(x), np.asanyarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if hasattr(x, 'mask') != hasattr(y, 'mask'):
        raise ValueError("only 1 of x and y have masks. Must be both or neither.")

    if hasattr(x, 'mask') and not np.all(x.mask == y.mask):
        raise ValueError("x and y has different masks.")

    if as_pairs:
        return np.array(list(zip(x.flatten(), y.flatten())))
    else:
        return x, y
