import numpy
import pandas
from shapely import geometry
import geopandas


def mpl_ax(ax, fallback='new'):
    """ Checks if a value if an Axes. If None, a new one is created or
    the 'current' one is found.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        The value to be tested.
    fallback : str, optional
        If ax is None. ``fallback='new'`` will create a new axes and
        figure. The other option is ``fallback='current'``, where the
        "current" axes are return via ``pyplot.gca()``

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    """

    if ax is None:
        from matplotlib import pyplot
        if fallback == 'new':
            fig, ax = pyplot.subplots()
        elif fallback == 'current':
            ax = pyplot.gca()
            fig = ax.figure
        else:
            raise ValueError("fallback must be either 'new' or 'current'")

    else:
        try:
            fig = ax.figure
        except AttributeError:
            msg = "`ax` must be a matplotlib Axes instance or None"
            raise ValueError(msg)

    return fig, ax


def polygon(polyverts, min_points=3):
    if isinstance(polyverts, geometry.Polygon):
        # might have Z coordinate. only pull X, Y
        polyverts = [xyz[:2] for xyz in polyverts.exterior.coords][:-1]

    polyverts_array = numpy.asarray(polyverts)
    if polyverts_array.ndim != 2:
        raise ValueError('polyverts must be a 2D array, or '
                         'similar sequence')

    if polyverts_array.shape[1] != 2:
        raise ValueError('polyverts must be two columns of points')

    if polyverts_array.shape[0] < min_points:
        raise ValueError('polyverts must contain at least 3 points')

    return polyverts_array


def _explode_geom(row):
    # save each geo part in its own row
    gsr = geopandas.GeoSeries([
        poly for poly in geometry.shape(row['geometry']).geoms
    ])
    meta = row['properties']
    return geopandas.GeoDataFrame(meta, geometry=gsr, index=gsr.index)


def _explode_gdf(gdf):
    return (
        pandas.concat(
            [_explode_geom(row) for row in gdf.iterfeatures()],
            ignore_index=True, sort=True
        ).pipe(geopandas.GeoDataFrame).set_crs(gdf.crs)
    )


def simple_polygon_gdf(gdf):
    if not gdf.geom_type.str.endswith('Polygon').all():
        raise ValueError("geometries must all be (Multi)Polygons")
    elif gdf.geom_type.eq('MultiPolygon').any():
        simples = gdf.loc[gdf.geom_type.eq('Polygon'), :]
        multis = gdf.loc[gdf.geom_type.eq('MultiPolygon'), :].pipe(_explode_gdf)
        gdf = pandas.concat([
            simples, multis
        ], ignore_index=True, sort=True).pipe(geopandas.GeoDataFrame, crs=gdf.crs)
    return gdf


def xy_array(x, y, as_pairs=True):
    x, y = numpy.asanyarray(x), numpy.asanyarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if hasattr(x, 'mask') != hasattr(y, 'mask'):
        raise ValueError("only 1 of x and y have masks. Must be both or neither.")

    if hasattr(x, 'mask') and not numpy.all(x.mask == y.mask):
        raise ValueError("x and y has different masks.")

    if as_pairs:
        return numpy.array(list(zip(x.flatten(), y.flatten())))
    else:
        return x, y


def elev_or_mask(x, other, array_name=None, offset=1, failNone=False):
    if array_name is None:
        array_name = 'other'

    if other is None:
        if failNone:
            raise ValueError(f'`{array_name}` cannot be `None`')
        else:
            return numpy.zeros_like(x)
    else:
        if (
                other.shape[0] != x.shape[0] - offset or
                other.shape[1] != x.shape[1] - offset
        ):
            raise ValueError(f'`{array_name}` not compatible with `x`')

        else:
            return other


def equivalent_masks(x, y):
    x = numpy.ma.masked_invalid(x)
    y = numpy.ma.masked_invalid(y)

    if x.shape != y.shape:
        raise ValueError('x, y are not the same shape')

    if not numpy.all(x.mask == y.mask):
        raise ValueError('x, y masks are not the same')
    else:
        return x, y
