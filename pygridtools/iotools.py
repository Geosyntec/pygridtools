import os
from collections import OrderedDict
from textwrap import dedent
import warnings

import numpy
import pandas
from shapely.geometry import Point, Polygon
import geopandas

from pygridtools import misc
from pygridtools import validate


def _warn_filterfxn(filterfxn):
    msg = '`filterfxn` no longer supported. Use the `query` method of the resulting dataframe.'
    if filterfxn:
        warnings.warn(msg)


def read_boundary(gisfile, betacol='beta', reachcol=None, sortcol=None,
                  upperleftcol=None, filterfxn=None):
    """ Loads boundary points from a GIS File.

    Parameters
    ----------
    gisfile : string
        Path to the GIS file containaing boundary points.
        Expected schema of the file...

        - order: numeric sort order of the points
        - beta: the 'beta' parameter used in grid generation to define
          turning points

    betacol : string (default='beta')
        Column in the attribute table specifying the beta parameter's
        value at each point.
    sortcol : optional string or None (default)
        Column in the attribute table specifying the sort order of the
        points.
    reachcol : optional string or None (default)
        Column in the attribute table specifying the names of the
        reaches of the river/esturary system.
    upperleftcol : optional string or None (default)
        Column in the attribute table toggling if the a point should be
        consider the upper-left corner of the system. Only one row of
        this column should evaluare to True.
    filterfxn : function or lambda expression or None (default)
        Removed. Use the `query` method of the result.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame of the boundary points with the following columns:

        - x (easting)
        - y (northing)
        - beta (turning parameter)
        - order (for sorting)
        - reach
        - upperleft

    """

    def _get_col_val(df, col, default=None):
        if col is not None:
            return df[col]
        else:
            return default

    _warn_filterfxn(filterfxn)
    gdf = (
        geopandas.read_file(gisfile)
                 .assign(x=lambda df: df['geometry'].x)
                 .assign(y=lambda df: df['geometry'].y)
                 .assign(beta=lambda df: _get_col_val(df, betacol, 0))
                 .assign(order=lambda df: _get_col_val(df, sortcol, df.index))
                 .assign(reach=lambda df: _get_col_val(df, reachcol, 'main'))
                 .assign(upperleft=lambda df: _get_col_val(df, upperleftcol, False))
                 .fillna(0)
                 .sort_values(by=['order'])
    )

    return gdf.loc[:, ['x', 'y', 'beta', 'upperleft', 'reach', 'order', 'geometry']]


def read_polygons(gisfile, filterfxn=None, squeeze=True, as_gdf=False):
    """ Load polygons (e.g., water bodies, islands) from a GIS file.

    Parameters
    ----------
    gisfile : string
        Path to the gisfile containaing boundary points.
    filterfxn : function or lambda expression or None (default)
        Removed. Use the `as_gdf` and the `query` method of the resulting
        GeoDataFrame.
    squeeze : optional bool (default = True)
        Set to True to return an array if only 1 record is present.
        Otherwise, a list of arrays will be returned.
    as_gdf : optional bool (default = False)
        Set to True to return a GeoDataFrame instead of arrays.

    Returns
    -------
    boundary : array or list of arrays, or GeoDataFrame

    Notes
    -----
    Multipart geometries are not supported. If a multipart geometry is
    present in a record, only the first part will be loaded.

    Z-coordinates are also not supported. Only x-y coordinates will be
    loaded.

    """
    _warn_filterfxn(filterfxn)
    gdf = geopandas.read_file(gisfile)
    if as_gdf:
        return gdf
    else:
        data = [numpy.array(g.boundary.coords) for g in gdf['geometry']]
        if len(data) == 1 and squeeze:
            data = data[0]
        return data


def read_grid(gisfile, icol='ii', jcol='jj', othercols=None, expand=1,
              as_gdf=False):
    if othercols is None:
        othercols = []

    grid = (
        geopandas.read_file(gisfile)
                 .rename(columns={icol: 'ii', jcol: 'jj'})
                 .set_index(['ii', 'jj'])
                 .sort_index()
    )
    if as_gdf:
        final_cols = othercols + ['geometry']
        return grid[final_cols]
    else:
        final_cols = ['easting', 'northing'] + othercols
        if (grid.geom_type != 'Point').any():
            msg = "can only read points for now when not returning a geodataframe"
            raise NotImplementedError(msg)

        return grid.assign(easting=grid['geometry'].x, northing=grid['geometry'].y)[final_cols]


def write_points(X, Y, crs, outputfile, river=None, reach=0, elev=None):
    """ Saves grid-related attributes of a pygridgen.Gridgen object to a
    GIS file with geomtype = 'Point'.

    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    crs : string
        A geopandas/proj/fiona-compatible string describing the coordinate
        reference system of the x/y values.
    outputfile : string
        Path to the point-geometry GIS file to which the data will be written.
    river : optional string (default = None)
        The river to be listed in the GIS files's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the GIS file's attribute
        table.
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
                    id=int(row), river=river, reach=reach,
                    ii=int(ii + 2), jj=int(jj + 2), elev=float(elev[jj, ii]),
                    ii_jj='{:02d}_{:02d}'.format(ii + 2, jj + 2),
                    geometry=Point(coords)
                )

                geodata.append(record)

    gdf = geopandas.GeoDataFrame(geodata, crs=crs, geometry='geometry')
    gdf.to_file(outputfile)
    return gdf


def write_cells(X, Y, mask, crs, outputfile, river=None, reach=0,
                elev=None, triangles=False):
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
    outputfile : string
        Path to the point GIS file to which the data will be written.
    river : optional string (default = None)
        The river to be listed in the GIS file's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the GIS file's attribute
        table.
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
                coords = misc.make_poly_coords(
                    xarr=X[jj:jj + 2, ii:ii + 2],
                    yarr=Y[jj:jj + 2, ii:ii + 2],
                    zpnt=Z, triangles=triangles
                )

                # build the attributes
                record = OrderedDict(
                    id=row, river=river, reach=reach,
                    ii=ii + 2, jj=jj + 2, elev=Z,
                    ii_jj='{:02d}_{:02d}'.format(ii + 2, jj + 2),
                    geometry=Polygon(shell=coords)
                )

                # append to file is coordinates are not masked
                # (masked = beyond the river boundary)
                if coords is not None:
                    geodata.append(record)

    gdf = geopandas.GeoDataFrame(geodata, crs=crs, geometry='geometry')
    gdf.to_file(outputfile)
    return gdf
