import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.mlab as mlab
import pandas


def points_inside_polygon(points, polyverts):
    return mpath.Path(polyverts).contains_points(points)


def makePolyCoords(xarr, yarr, zpnt=None, triangles=False):
    """ Makes an array for coordinates suitable for building
    quadrilateral geometries in shapfiles via fiona.

    Parameters
    ----------
    xarr, yarr : numpy arrays
        Arrays (2x2) of x coordinates and y coordinates for each vertex
        of the quadrilateral.
    zpnt : optional float or None (default)
        If provided, this elevation value will be assigned to all four
        vertices.
    triangles : optional bool (default = False)
        If True, triangles will be returned

    Returns
    -------
    coords : numpy array
        An array suitable for feeding into fiona as the geometry of a record.

    """

    def process_input(array):
        flat = np.hstack([array[0,:], array[1,::-1]])
        return flat[~np.isnan(flat)]

    x = process_input(xarr)
    y = process_input(yarr)
    if (not isinstance(xarr, np.ma.MaskedArray) or xarr.mask.sum() == 0
            or (triangles and len(x) == 3)):
        if zpnt is None:
            coords = np.vstack([x ,y]).T
        else:
            z = np.array([zpnt] * x.shape[0])
            coords = np.vstack([x, y, z]).T

    else:
        coords = None

    return coords


def makeRecord(ID, coords, geomtype, props):
    """ Creates a record to be appended to a shapefile via fiona.

    Parameters
    ----------
    ID : int
        The record ID number
    coords : tuple or array-like
        The x-y coordinates of the geometry. For Points, just a tuple.
        An array or list of tuples for LineStrings or Polygons
    geomtype : string
        A valid GDAL/OGR geometry specification (e.g. LineString, Point,
        Polygon)
    props : dict or collections.OrderedDict
        A dict-like object defining the attributes of the record

    Returns
    -------
    record : dict
        A nested dictionary suitable for the fiona package to append to
        a shapefile

    Notes
    -----
    This is ignore the mask of a MaskedArray. That might be bad.

    """

    if not geomtype in ['Point', 'LineString', 'Polygon']:
        raise ValueError('Geometry {} not suppered'.format(geomtype))

    if isinstance(coords, np.ma.MaskedArray):
        coords = coords.data

    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    record = {
        'id': ID,
        'geometry': {
            'coordinates': coords if geomtype == 'Point' else [coords],
            'type': geomtype
        },
        'properties': props
    }
    return record


def interpolateBathymetry(bathy, x_points, y_points, xcol='x', ycol='y', zcol='z'):
    """ Interpolates x-y-z point data onto the grid of a Gridgen object.
    Matplotlib's nearest-neighbor interpolation schema is used to
    estimate the elevation at the grid centers.

    Parameters
    ----------
    bathy : pandas.DataFrame or None
        The bathymetry data stored as x-y-z points in a DataFrame.
    [x|y]_points : numpy arrays
        The x, y locations onto which the bathymetry will be
        interpolated.
    xcol/ycol/zcol : optional strings
        Column names for each of the quantities defining the elevation
        pints. Defaults are "x/y/z".

    Returns
    -------
    gridbathy : pandas.DataFrame
        The bathymetry for just the area covering the grid.

    """

    try:
        import pygridgen
    except ImportError: # pragma: no cover
        raise ImportError("`pygridgen` not installed. Cannot interpolate bathymetry.")

    if bathy is None:
        elev = np.zeros(x_points.shape)

        if isinstance(x_points, np.ma.MaskedArray):
            elev = np.ma.MaskedArray(data=elev, mask=x_points.mask)

        bathy = pandas.DataFrame({
            xcol: x_points.flatten(),
            ycol: y_points.flatten(),
            zcol: elev.flatten()
        })

    else:
        bathy = bathy[[xcol, ycol, zcol]]

    # find where the bathy is inside our grid
    grididx = (
        (bathy[xcol] <= x_points.max()) &
        (bathy[xcol] >= x_points.min()) &
        (bathy[ycol] <= y_points.max()) &
        (bathy[ycol] >= y_points.min())
    )

    gridbathy = bathy[grididx].dropna(how='any')

    # fill in NaNs with something outside of the bounds
    xx = x_points.copy()
    yy = y_points.copy()
    xx[np.isnan(x_points)] = x_points.max() + 5
    yy[np.isnan(y_points)] = y_points.max() + 5

    # use cubic-spline approximation to interpolate the grid
    csa = pygridgen.csa(gridbathy[xcol], gridbathy[ycol], gridbathy[zcol])
    return csa(xx, yy)


def padded_stack(a, b, how='vert', where='+', shift=0, padval=np.nan):
    """ Merge 2-dimensional numpy arrays with different shapes.

    Parameters
    ----------
    a, b : numpy arrays
        The arrays to be merged
    how : optional string (default = 'vert')
        The method through wich the arrays should be stacked. `'Vert'`
        is analogous to `np.vstack`. `'Horiz'` maps to `np.hstack`.
    where : optional string (default = '+')
        The placement of the arrays relative to each other. Keeping in
        mind that the origin of an array's index is in the upper-left
        corner, `'+'` indicates that the second array will be placed
        at higher index relative to the first array. Essentially:
         - if how == 'vert'
            - `'+'` -> `a` is above (higher index) `b`
            - `'-'` -> `a` is below (lower index) `b`
         - if how == 'horiz'
            - `'+'` -> `a` is to the left of `b`
            - `'-'` -> `a` is to the right of `b`
        See the examples for more info.
    shift : int (default = 0)
        The number of indices the second array should be shifted in
        axis other than the one being merged. In other words, vertically
        stacked arrays can be shifted horizontally, and horizontally
        stacked arrays can be shifted vertically.
    padval : optional, same type as array (default = np.nan)
        Value with which the arrays will be padded.

    Returns
    -------
    Stacked : numpy array
        The merged and padded array

    Examples
    --------
    >>> import pygridtools as pgt
    >>> a = np.arange(12).reshape(4, 3) * 1.0
    >>> b = np.arange(8).reshape(2, 4) * -1.0
    >>> pgt.padded_stack(a, b, how='vert', where='+', shift=1)
        array([[  0.,   1.,   2.,  nan,  nan],
               [  3.,   4.,   5.,  nan,  nan],
               [  6.,   7.,   8.,  nan,  nan],
               [  9.,  10.,  11.,  nan,  nan],
               [ nan,  -0.,  -1.,  -2.,  -3.],
               [ nan,  -4.,  -5.,  -6.,  -7.]])

    >>> pgt.padded_stack(a, b, how='h', where='-', shift=-2)
        array([[ nan,  nan,  nan,  nan,   0.,   1.,   2.],
               [ nan,  nan,  nan,  nan,   3.,   4.,   5.],
               [ -0.,  -1.,  -2.,  -3.,   6.,   7.,   8.],
               [ -4.,  -5.,  -6.,  -7.,   9.,  10.,  11.]]


    """

    a = np.asarray(a)
    b = np.asarray(b)

    if where == '-':
        stacked = padded_stack(b, a, shift=-1*shift, where='+', how=how)

    elif where == '+':
        if how.lower() in ('horizontal', 'horiz', 'h'):
            stacked = padded_stack(a.T, b.T, shift=shift, where=where,
                                   how='v').T

        elif how.lower() in ('vertical', 'vert', 'v'):

            a_pad_left = 0
            a_pad_right = 0
            b_pad_left = 0
            b_pad_right = 0

            diff_cols = a.shape[1] - (b.shape[1] + shift)

            if shift > 0:
                b_pad_left = shift
            elif shift < 0:
                a_pad_left = abs(shift)

            if diff_cols > 0:
                b_pad_right = diff_cols
            else:
                a_pad_right = abs(diff_cols)

            v_pads = (0, 0)
            x_pads = (v_pads, (a_pad_left, a_pad_right))
            y_pads = (v_pads, (b_pad_left, b_pad_right))

            mode = 'constant'
            fill = (padval, padval)
            stacked = np.vstack([
                np.pad(a, x_pads, mode=mode, constant_values=fill),
                np.pad(b, y_pads, mode=mode, constant_values=fill)
            ])

        else:
            gen_msg = 'how must be either "horizontal" or "vertical"'
            raise ValueError(gen_msg)

    else:
        raise ValueError('`where` must be either "+" or "-"')

    return stacked


def make_gefdc_cells(node_mask, cell_mask=None, triangles=False):
    """ Take an array defining the nodes as wet (1) or dry (0) create
    the array of cell values needed for GEFDC.

    Input
    -----
    node_mask : numpy bool array (N x M)
        Bool array specifying if a *node* is present in the raw
        (unmasked) grid.
    cell_mask : optional numpy bool array (N-1 x M-1) or None (default)
        Bool array specifying if a cell should be masked (e.g. due to
        being an island or something like that).
    triangles : optional bool (default = False)
        Currently not implemented. Will eventually enable the writting of
        triangular cells when True.

    Returns
    -------
    cell_array : numpy array
        Integer array of the values written to ``outfile``.

    """

    triangle_cells = {
        0: 3,
        1: 2,
        3: 1,
        2: 4,
    }
    land_cell = 0
    water_cell = 5
    bank_cell = 9

    # I can't figure this out
    if triangles:
        warnings.warn('triangles are experimental')

    # define the initial cells with everything labeled as a bank
    ny, nx = cell_mask.shape
    cells = np.zeros((ny+2, nx+2), dtype=int) + bank_cell

    # loop through each *node*
    for jj in range(1, ny+1):
        for ii in range(1, nx+1):
            # pull out the 4 nodes defining the cell (call it a quad)
            quad = node_mask[jj-1:jj+1, ii-1:ii+1]
            n_wet = quad.sum()

            # anything that's masked is a "bank"
            if not cell_mask[jj-1, ii-1]:
                # if all 4 nodes are wet (=1), then the cell is 5
                if n_wet == 4:
                    cells[jj, ii] = water_cell

                # if only 3  are wet, might be a triangle, but...
                # this ignored since we already raised an error
                elif n_wet == 3 and triangles:
                    dry_node = np.argmin(quad.flatten())
                    cells[jj, ii] = triangle_cells[dry_node]

            # otherwise it's just a bank
            else:
                cells[jj, ii] = bank_cell

    padded_cells = np.pad(cells, 1, mode='constant', constant_values=bank_cell)
    for cj in range(cells.shape[0]):
        for ci in range(cells.shape[1]):
            shift = 3
            total = np.sum(padded_cells[cj:cj+shift, ci:ci+shift])
            if total == bank_cell * shift**2:
                cells[cj, ci] = land_cell

    nrows = cells.shape[0]
    ncols = cells.shape[1]

    # nchunks = np.ceil(ncols / maxcols)
    # if ncols > maxcols:
    #     final_cells = np.zeros((nrows*nchunks, maxcols), dtype=int)
    #     for n in np.arange(nchunks):
    #         col_start = n * maxcols
    #         col_stop = (n+1) * maxcols

    #         row_start = n * nrows
    #         row_stop = (n+1) * nrows

    #         cells_to_move = cells[:, col_start:col_stop]
    #         final_cells[row_start:row_stop, 0:cells_to_move.shape[1]] = cells_to_move
    # else:
    #     final_cells = cells.copy()

    final_cells = cells.copy()
    return final_cells
