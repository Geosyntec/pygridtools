import warnings

import numpy
from matplotlib import pyplot
import geopandas

try:
    import ipywidgets
except ImportError:  # pragma: no cover
    ipywidgets = None

from pygridgen.tests import requires
import pygridgen as pgg

from pygridtools import viz


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


def _change_shape(g, irows, jcols, plotfxn, plotopts=None):
    """ changes the number of rows and cols in a Gridgen (g) and passes the
    grid nodes to a plotting function
    """
    if not plotopts:
        plotopts = {}
    g.ny = irows
    g.nx = jcols
    g.generate_grid()
    return plotfxn(g.x, g.y, **plotopts)


@requires(ipywidgets, 'ipywidgets')
def interactive_grid_shape(grid, max_n=200, plotfxn=None, **kwargs):
    """ Interactive ipywidgets for select the shape of a grid

    Parameters
    ----------
    grid : pygridgen.Gridgen
        The base grid from which the grids of new shapes (resolutions) will be
        generated.
    max_n : int (default = 200)
        The maximum number of possible cells in each dimension.
    plotfxn : callable, optional
        Function that plots the grid to provide user feedback. The call
        signature of this function must accept to positional parameters for the
        x- and y-arrays of node locations, and then accept any remaining keyword
        arguments. If not provided, *pygridtools.viz.plot_cells* is used.
    **kwargs : plotting parametes, optional
        All remaining keyword arguments are passed to *plotfxn*

    Returns
    -------
    newgrid : pygridgen.Gridgen
        The reshaped grid
    widget : ipywidgets.interactive
        Collection of IntSliders for changing the number cells along each axis
        in the grid.

    Examples
    --------
    >>> from pygridgen import grid
    >>> from pygridtools import viz, iotools
    >>> def make_fake_bathy(shape):
    ...     j_cells, i_cells = shape
    ...     y, x = numpy.mgrid[:j_cells, :i_cells]
    ...     z = (y - (j_cells // 2))** 2 - x
    ...     return z
    >>> def plot_grid(x, y, ax=None):
    ...     shape = x[1:, 1:].shape
    ...     bathy = make_fake_bathy(shape)
    ...     if not ax:
    ...         fig, ax = pyplot.subplots(figsize=(8, 8))
    ...     ax.set_aspect('equal')
    ...     return viz.plot_cells(x, y, ax=ax, cmap='Blues', colors=bathy, lw=0.5, ec='0.3')
    >>> d = numpy.array([
    ... (13, 16,  1.00), (18, 13,  1.00), (12,  7,  0.50),
    ... (10, 10, -0.25), ( 5, 10, -0.25), ( 5,  0,  1.00),
    ... ( 0,  0,  1.00), ( 0, 15,  0.50), ( 8, 15, -0.25),
    ... (11, 13, -0.25)])
    >>> g = grid.Gridgen(d[:, 0], d[:, 1], d[:, 2], (75, 75), ul_idx=1, focus=None)
    >>> new_grid, widget = iotools.interactive_grid_shape(g, plotfxn=plot_grid)  # doctest: +SKIP
    """

    if not plotfxn:
        plotfxn = viz.plot_cells

    common_opts = dict(min=2, max=max_n, continuous_update=False)
    return grid, ipywidgets.interactive(
        _change_shape,
        g=ipywidgets.fixed(grid),
        irows=ipywidgets.IntSlider(value=grid.ny, **common_opts),
        jcols=ipywidgets.IntSlider(value=grid.nx, **common_opts),
        plotfxn=ipywidgets.fixed(plotfxn),
        plotopts=ipywidgets.fixed(kwargs)
    )


class _FocusProperties:
    """A dummy class to hold the properties of the grid._FocusPoint() object.
    This class is required so that multiple ipywidgets.interactive widgets
    can interact on the same plot.
    """
    def __init__(self, pos=0.5, axis='x', factor=0.5, extent=0.5):
        """

        Parameters
        ----------
        pos : float
            Relative position within the grid of the focus. This must
            be in the range [0, 1]
        axis : string ('x' or 'y')
            Axis along which the grid will be focused.
        factor : float
            Amount to focus grid. Creates cell sizes that are factor
            smaller (factor > 1) or larger (factor < 1) in the focused
            region.
        extent : float
            Lateral extent of focused region."""

        self.pos = pos
        self.axis = axis
        self.factor = factor
        self.extent = extent

    @property
    def focuspoint(self):
        """Property returns grid._FocusPoint"""
        return pgg.grid._FocusPoint(pos=self.pos, axis=self.axis, factor=self.factor, extent=self.extent)


def _plot_focus_points(focus_points, g, plotfxn, plotopts=None):
    """Plots multiple focus points on a grid.

    Parameters
    ----------
    focus_points : tuple of FocusProperties
        These focus points are applied to grid `g`.
    g : grid.Gridgen
        The grid to plot.
    plotfxn : callable
        Function that plots the grid to provide user feedback. The call
        signature of this function must accept to positional parameters for the
        x- and y-arrays of node locations, and then accept any remaining keyword
        arguments.

    Additional Parameters
    ---------------------
    All remaining keyword arguments are passed to *plotfxn*

    Returns
    -------
    plotfxn(g.x, g.y, **plotopts)
    """
    if not plotopts:
        plotopts = {}

    # extact the grid._FocusPoint from the dummy class
    f = pgg.grid.Focus(*(fp.focuspoint for fp in focus_points))
    g.focus = f
    g.generate_grid()

    return plotfxn(g.x, g.y, **plotopts)


def _change_focus(fpoint, others, axis, pos, factor, extent, g, plotfxn, plotopts=None):
    """
    The function changes the properties of `fpoint` only and
    plots `fpoint` with `others`.

    Parameters
    ----------
    fpoint : FocusProperties
        The focus point modified and applied to grid `g`.
    others : tuple of FocusProperties
        These focus points are applied to grid `g` but not modified.
    axis : string ('x' or 'y')
        Axis along which the grid will be focused by `fpoint`.
    pos : float
        Relative position within the grid of the focus `fpoint`. This must
        be in the range [0, 1]
    factor : float
        Amount to focus grid by `fpoint`. Creates cell sizes that are factor
        smaller (factor > 1) or larger (factor < 1) in the focused
        region.
    extent : float
        Lateral extent of focused region by `fpoint`.
    g : grid.Gridgen
        The grid to plot.
    plotfxn : callable
        Function that plots the grid to provide user feedback. The call
        signature of this function must accept to positional parameters for the
        x- and y-arrays of node locations, and then accept any remaining keyword
        arguments.

    Additional Parameters
    ---------------------
    All remaining keyword arguments are passed to *plotfxn*

    Returns
    -------
    _plot_focus_points(focuspoints, g, plotfxn, plotopts)
    """
    # update fpoint properties
    fpoint.pos = pos
    fpoint.axis = axis
    fpoint.factor = factor
    fpoint.extent = extent
    # concat points to plot
    focuspoints = (fpoint,) + others
    return _plot_focus_points(focuspoints, g, plotfxn, plotopts)


def _plot_points_wrapper(x, y, **kwargs):
    figsize = kwargs.pop('figsize', (9, 9))
    fig, ax = pyplot.subplots(figsize=figsize)
    ax.set_aspect('equal')

    return viz.plot_points(x, y, ax=ax, **kwargs)


@requires(ipywidgets, 'ipywidgets')
def interactive_grid_focus(g, n_points, plotfxn=None, **kwargs):
    """
    Interactive ipywidgets for changing focus points.

    Parameters
    ----------
    grid : pygridgen.Gridgen
        The base grid from which the grids of new focus points will be
        generated.
    n_points : int
        The number of focal points to add to the grid.
    plotfxn : callable, optional
        Function that plots the grid to provide user feedback. The call
        signature of this function must accept to positional parameters for the
        x- and y-arrays of node locations, and then accept any remaining keyword
        arguments. If not provided, *pygridtools.viz.plot_points* is used.

    Additional Parameters
    ---------------------
    All remaining keyword arguments are passed to *plotfxn*

    Returns
    -------
    newgrid : pygridgen.Gridgen
        The reshaped grid
    widget : ipywidgets.interactive
        Collection of Tab / IntSliders for changing the number focus points
        along each axis in the grid.

    Examples
    --------
    >>> from pygridgen import grid
    >>> from pygridtools import viz, iotools
    >>> d = numpy.array([
    ... (13, 16,  1.00), (18, 13,  1.00), (12,  7,  0.50),
    ... (10, 10, -0.25), ( 5, 10, -0.25), ( 5,  0,  1.00),
    ... ( 0,  0,  1.00), ( 0, 15,  0.50), ( 8, 15, -0.25),
    ... (11, 13, -0.25)])
    >>> g = grid.Gridgen(d[:, 0], d[:, 1], d[:, 2], (75, 75), ul_idx=1, focus=None)
    >>> n = 4 # number of focus objects
    >>> new_grid, widget = iotools.interactive_grid_focus(g, n)  # doctest: +SKIP
    """

    if not plotfxn:
        plotfxn = _plot_points_wrapper

    # common linear slider options
    common_opts = dict(min=0.01, max=1, step=0.01, continuous_update=False)
    # common log slider options
    common_log_f_opts = dict(min=-2, max=2, step=0.1, continuous_update=False)

    # we need to create a list of widgets for ipywidgets.Tab()
    widgets = []
    # set up our dummy classes
    focus_points = [_FocusProperties() for f in range(n_points)]
    for n, fp in enumerate(focus_points):

        widget = ipywidgets.interactive(
            _change_focus,
            fpoint=ipywidgets.fixed(focus_points[n]),
            others=ipywidgets.fixed(tuple(focus_points[:n] + focus_points[n + 1:])),
            axis=ipywidgets.ToggleButtons(
                options=['x', 'y'],
                description='Axis:'),
            pos=ipywidgets.FloatSlider(0.5, **common_opts),
            # currently the only log slider
            factor=ipywidgets.FloatLogSlider(1, **common_log_f_opts),
            extent=ipywidgets.FloatSlider(0.5, **common_opts),
            g=ipywidgets.fixed(g),
            plotfxn=ipywidgets.fixed(plotfxn),
            plotopts=ipywidgets.fixed(kwargs)
        )

        widgets += [widget]

    tab_nest = ipywidgets.Tab()
    tab_nest.children = widgets
    for n in range(len(widgets)):
        tab_nest.set_title(n, f'Focus {n + 1}')

    return focus_points, tab_nest
