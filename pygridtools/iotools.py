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
