import numpy as np
import matplotlib.pyplot as plt


def mpl_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_aspect('equal')
    return fig, ax


def polygon(polyverts, min_points=3):
    polyverts_array = np.asarray(polyverts)
    if polyverts_array.ndim != 2:
        raise ValueError('polyverts must be a 2D array, or '
                         'similar sequence')

    if polyverts_array.shape[1] != 2:
        raise ValueError('polyverts must be two columns of points')

    if polyverts_array.shape[0] < min_points:
        raise ValueError('polyverts must contain at least {} points'.format(min_points))

    return polyverts_array


def xy_array(x, y, as_pairs=True):
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


def file_mode(mode):
    if mode.lower() not in ['a', 'w']:
        raise ValueError('`mode` must be either "a" (append) or "w" (write)')

    return mode.lower()


def elev_or_mask(x, other, array_name=None, offset=1, failNone=False):
    if array_name is None:
        array_name = 'other'

    if other is None:
        if failNone:
            raise ValueError('`{}` cannot be `None`'.format(array_name))
        else:
            return np.zeros_like(x)
    else:
        if (
                other.shape[0] != x.shape[0] - offset or
                other.shape[1] != x.shape[1] - offset
        ):
            raise ValueError('`{}` not compatible with `x`'.format(array_name))

        else:
            return other


def equivalent_masks(x, y):
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)

    if x.shape != y.shape:
        raise ValueError('x, y are not the same shape')

    if not np.all(x.mask == y.mask):
        raise ValueError('x, y masks are not the same')
    else:
        return x, y
