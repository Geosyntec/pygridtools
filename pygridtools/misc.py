from collections import OrderedDict
from shapely.geometry import Point, Polygon
import importlib

import numpy
import matplotlib.path as mpath
import pandas
import geopandas
from pygridgen import csa

from pygridtools import validate


def make_poly_coords(xarr, yarr, zpnt=None, triangles=False):
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
        flat = numpy.hstack([array[0, :], array[1, ::-1]])
        return flat[~numpy.isnan(flat)]

    x = process_input(xarr)
    y = process_input(yarr)
    if (not isinstance(xarr, numpy.ma.MaskedArray) or xarr.mask.sum() == 0 or
            (triangles and len(x) == 3)):
        if zpnt is None:
            coords = numpy.vstack([x, y]).T
        else:
            z = numpy.array([zpnt] * x.shape[0])
            coords = numpy.vstack([x, y, z]).T

    else:
        coords = None

    return coords


def make_record(ID, coords, geomtype, props):
    """ Creates a record to be appended to a GIS file via *geopandas*.

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
        A nested dictionary suitable for the a *geopandas.GeoDataFrame*,

    Notes
    -----
    This is ignore the mask of a MaskedArray. That might be bad.

    """

    if geomtype not in ['Point', 'LineString', 'Polygon']:
        raise ValueError(f'Geometry {geomtype} not suppered')

    if isinstance(coords, numpy.ma.MaskedArray):
        coords = coords.data

    if isinstance(coords, numpy.ndarray):
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


def interpolate_bathymetry(bathy, x_points, y_points, xcol='x', ycol='y', zcol='z'):
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

    HASPGG = bool(importlib.util.find_spec("pygridgen"))
    if not HASPGG:
        raise RuntimeError("`pygridgen` not installed. Cannot interpolate bathymetry.")

    if bathy is None:
        elev = numpy.zeros(x_points.shape)

        if isinstance(x_points, numpy.ma.MaskedArray):
            elev = numpy.ma.MaskedArray(data=elev, mask=x_points.mask)

        bathy = pandas.DataFrame({
            xcol: x_points.flatten(),
            ycol: y_points.flatten(),
            zcol: elev.flatten()
        })

    else:
        bathy = bathy[[xcol, ycol, zcol]]

    # find where the bathymetry is inside our grid
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
    xx[numpy.isnan(x_points)] = x_points.max() + 5
    yy[numpy.isnan(y_points)] = y_points.max() + 5

    # use cubic-spline approximation to interpolate the grid
    interpolate = csa.CSA(
        gridbathy[xcol].values,
        gridbathy[ycol].values,
        gridbathy[zcol].values
    )
    return interpolate(xx, yy)


def padded_stack(a, b, how='vert', where='+', shift=0, padval=numpy.nan):
    """ Merge 2-dimensional numpy arrays with different shapes.

    Parameters
    ----------
    a, b : numpy arrays
        The arrays to be merged
    how : optional string (default = 'vert')
        The method through wich the arrays should be stacked. `'Vert'`
        is analogous to `numpy.vstack`. `'Horiz'` maps to `numpy.hstack`.
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
    padval : optional, same type as array (default = numpy.nan)
        Value with which the arrays will be padded.

    Returns
    -------
    Stacked : numpy array
        The merged and padded array

    Examples
    --------
    >>> import pygridtools as pgt
    >>> a = numpy.arange(12).reshape(4, 3) * 1.0
    >>> b = numpy.arange(8).reshape(2, 4) * -1.0
    >>> pgt.padded_stack(a, b, how='vert', where='+', shift=2)
    array([[ 0.,  1.,  2.,   -,   -,   -],
           [ 3.,  4.,  5.,   -,   -,   -],
           [ 6.,  7.,  8.,   -,   -,   -],
           [ 9., 10., 11.,   -,   -,   -],
           [  -,   -, -0., -1., -2., -3.],
           [  -,   -, -4., -5., -6., -7.]])

    >>> pgt.padded_stack(a, b, how='h', where='-', shift=-1)
    array([[-0., -1., -2., -3.,   -,   -,   -],
           [-4., -5., -6., -7.,  0.,  1.,  2.],
           [  -,   -,   -,   -,  3.,  4.,  5.],
           [  -,   -,   -,   -,  6.,  7.,  8.],
           [  -,   -,   -,   -,  9., 10., 11.]])

    """

    a = numpy.asarray(a)
    b = numpy.asarray(b)

    if where == '-':
        stacked = padded_stack(b, a, shift=-1 * shift, where='+', how=how)

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
            stacked = numpy.vstack([
                numpy.pad(a, x_pads, mode=mode, constant_values=fill),
                numpy.pad(b, y_pads, mode=mode, constant_values=fill)
            ])

        else:
            gen_msg = 'how must be either "horizontal" or "vertical"'
            raise ValueError(gen_msg)

    else:
        raise ValueError('`where` must be either "+" or "-"')

    return stacked


def padded_sum(padded, window=1):
    return (padded[window:, window:] + padded[:-window, :-window] +
            padded[:-window, window:] + padded[window:, :-window])


def mask_with_polygon(x, y, *polyverts, inside=True):
    """ Mask x-y arrays inside or outside a polygon

    Parameters
    ----------
    x, y : array-like
        NxM arrays of x- and y-coordinates.
    polyverts : sequence of a polygon's vertices
        A sequence of x-y pairs for each vertex of the polygon.
    inside : bool (default is True)
        Toggles masking the inside or outside the polygon

    Returns
    -------
    mask : bool array
        The NxM mask that can be applied to ``x`` and ``y``.

    """
    # validate input
    polyverts = [validate.polygon(pv) for pv in polyverts]
    points = validate.xy_array(x, y, as_pairs=True)

    # compute the mask
    mask = numpy.dstack([
        mpath.Path(pv).contains_points(points).reshape(x.shape)
        for pv in polyverts
    ]).any(axis=-1)

    if not inside:
        mask = ~mask
    return mask


def gdf_of_cells(X, Y, mask, crs, elev=None, triangles=False):
    """ Saves a GIS file of quadrilaterals representing grid cells.

    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    mask : numpy array or None
        Array describing which cells to mask (exclude) from the output.
        Shape should be N-1 by M-1, where N and M are the dimensions of
        `X` and `Y`.
    crs : string
        A geopandas/proj/fiona-compatible string describing the coordinate
        reference system of the x/y values.
    elev : optional array or None (defauly)
        The elevation of the grid cells. Shape should be N-1 by M-1,
        where N and M are the dimensions of `X` and `Y` (like `mask`).
    triangles : optional bool (default = False)
        If True, triangles can be included

    Returns
    -------
    geopandas.GeoDataFrame

    """

    # check X, Y shapes
    Y = validate.elev_or_mask(X, Y, 'Y', offset=0)

    # check elev shape
    elev = validate.elev_or_mask(X, elev, 'elev', offset=0)

    # check the mask shape
    mask = validate.elev_or_mask(X, mask, 'mask', offset=1)

    X = numpy.ma.masked_invalid(X)
    Y = numpy.ma.masked_invalid(Y)
    ny, nx = X.shape

    row = 0
    geodata = []
    for ii in range(nx - 1):
        for jj in range(ny - 1):
            if not (numpy.any(X.mask[jj:jj + 2, ii:ii + 2]) or mask[jj, ii]):
                row += 1
                Z = elev[jj, ii]
                # build the array or coordinates
                coords = make_poly_coords(
                    xarr=X[jj:jj + 2, ii:ii + 2],
                    yarr=Y[jj:jj + 2, ii:ii + 2],
                    zpnt=Z, triangles=triangles
                )

                # build the attributes
                record = OrderedDict(
                    id=row, ii=ii + 2, jj=jj + 2, elev=Z,
                    ii_jj=f'{ii + 2:02d}_{jj + 2:02d}',
                    geometry=Polygon(shell=coords)
                )

                # append to file is coordinates are not masked
                # (masked = beyond the river boundary)
                if coords is not None:
                    geodata.append(record)

    gdf = geopandas.GeoDataFrame(geodata, crs=crs, geometry='geometry')
    return gdf


def gdf_of_points(X, Y, crs, elev=None):
    """ Saves grid-related attributes of a pygridgen.Gridgen object to a
    GIS file with geomtype = 'Point'.

    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    crs : string
        A geopandas/proj/fiona-compatible string describing the coordinate
        reference system of the x/y values.
    elev : optional array or None (defauly)
        The elevation of the grid cells. Array dimensions must be 1 less than
        X and Y.

    Returns
    -------
    geopandas.GeoDataFrame

    """

    # check that X and Y are have the same shape, NaN cells
    X, Y = validate.equivalent_masks(X, Y)

    # check elev shape
    elev = validate.elev_or_mask(X, elev, 'elev', offset=0)

    # start writting or appending to the output
    row = 0
    geodata = []
    for ii in range(X.shape[1]):
        for jj in range(X.shape[0]):
            # check that nothing is masked (outside of the river)
            if not (X.mask[jj, ii]):
                row += 1

                # build the coords
                coords = (X[jj, ii], Y[jj, ii])

                # build the attributes
                record = OrderedDict(
                    id=int(row), ii=int(ii + 2), jj=int(jj + 2),
                    elev=float(elev[jj, ii]),
                    ii_jj=f'{ii + 2:02d}_{jj + 2:02d}',
                    geometry=Point(coords)
                )

                geodata.append(record)

    gdf = geopandas.GeoDataFrame(geodata, crs=crs, geometry='geometry')
    return gdf
