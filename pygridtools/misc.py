import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pandas
import fiona

import pygridgen

from . import viz
from . import io


def points_inside_poly(points, polyverts):
    return mpath.Path(polyverts).contains_points(points)


def interpolateBathymetry(bathy, grid, xcol='x', ycol='y', zcol='z'):
    '''
    Interpolates x-y-z point data onto the grid of a Gridgen object.
    Matplotlib's nearest-neighbor interpolation schema is used to
    estimate the elevation at the grid centers.

    Parameters
    ----------
    bathy : pandas.DataFrame or None
        The bathymetry data stored as x-y-z points in a DataFrame.
    grid : pygridgen.grid.Gridgen object
        The grid onto which the bathymetry will be interpolated
    xcol/ycol/zcol : optional strings
        Column names for each of the quantities defining the elevation
        pints. Defaults are "easting/northing/elevation".

    Returns
    -------
    gridbathy : pandas.DataFrame
        The bathymetry for just the area covering the grid.

    Notes
    -----
    This operates on the grid object in place.

    '''
    import matplotlib.delaunay as mdelaunay

    if bathy is None:
        elev = np.zeros(grid.x_rho.shape)

        if isinstance(grid.x_rho, np.ma.MaskedArray):
            elev = np.ma.MaskedArray(data=elev, mask=grid.x_rho.mask)

        bathy = pandas.DataFrame({
            'x': grid.x_rho.flatten(),
            'y': grid.y_rho.flatten(),
            'z': elev.flatten()
        })

    else:
        print('  using provided bathymetry data')
        bathy = bathy[['x', 'y', 'z']]

        # find where the bathy is inside our grid
        grididx = (
            (bathy['x'] <= grid.x_rho.max()) &
            (bathy['x'] >= grid.x_rho.min()) &
            (bathy['y'] <= grid.y_rho.max()) &
            (bathy['y'] >= grid.y_rho.min())
        )

        gridbathy = bathy[grididx]

        # triangulate the grid
        print('  triangulating the bathymetry')
        triangles = mdelaunay.Triangulation(gridbathy['x'], gridbathy['y'])

        print('  extrapolating the bathymetry')
        try:
            extrapolate = triangles.nn_extrapolator(gridbathy['z'])
        except:
            extrapolate = triangles.nn_extrapolator(gridbathy['z'][:-1])

        elev = np.ma.masked_invalid(extrapolate(grid.x_rho, grid.y_rho))

    grid.elev = elev
    return bathy


def makeGrid(coords=None, bathydata=None, makegrid=True, grid=None,
             plot=True, xlimits=None, ax=None, figpath=None,
             outdir=None, title=None, verbose=False, **gparams):
    '''
    Generate and (optionally) visualize a grid, and create input files
    for the GEDFC preprocessor (makes grid input files for GEFDC).

    Parameters
    ----------
    coords : optional pandas.DataFrame or None (default)
        Defines the boundary of the model area. Must be provided if
        `makegrid` = True. Required columns:
          - 'x' (easting)
          - 'y' (northing),
          - 'beta' (turning points, must sum to 1)
    bathydata : optional pandas.DataFrame or None (default)
        Point bathymetry/elevation data. Will be interpolated unto the
        grid if provided. If None, a default value of 0 will be used.
        Required columns:
          - 'x' (easting)
          - 'y' (northing),
          - 'z' (elevation)
    makegrid : optional bool (default = True)
        Set to false to not generate a new grid in favor of using the
        provided object.
    plot : optional bool (default = False)
        Draws a plot of the grid
    xlimits : optional bool (default = False)
        Sets the xlimit of `Axes` object
    ax : optional `matplotlib.Axes object or None (default)
        Axes on which the grid will be drawn if `plot` = True. If
        ommitted as `plot` = True, a new Axes will be created.
    **gparams : optional kwargs
        Parameters to be passed to the pygridgen.grid.Gridgen constructor.
        Only used if `makegrid` = True and `coords` is not None.
        `ny` and `nx` are required. Other values are optional.

    Returns
    -------
    grid : pygridgen.grid.Gridgen obejct
    fig : matplotlib.Figure object

    Notes
    -----
    - Generating the grid can take some time. pass in `verbose` = True
       to watch the progress in the console.

    See Also
    --------
    pygridgen.grid.Gridgen

    '''

    # generate the grid.
    if grid is None:
        if makegrid:
            try:
                nx = gparams.pop('nx')
                ny = gparams.pop('ny')
            except KeyError:
                raise ValueError('must provide `nx` and `ny` if '
                                 '`makegrid` = True')
            if verbose:
                print('generating grid')
            grid = pygridgen.Gridgen(coords.x, coords.y, coords.beta,
                                     (ny, nx), **gparams)
        else:
            raise ValueError("must provide `grid` if `makegrid` = False")
    if verbose:
        print('interpolating bathymetry')
    newbathy = interpolateBathymetry(bathydata, grid, xcol='x',
                                     ycol='y', zcol='z')

    if plot:
        if verbose:
            print('plotting data and saving image')
        fig, ax = viz.plotPygridgen(grid, ax=ax)
        ax.set_aspect('equal')
        if xlimits is not None:
            ax.set_xlim(xlimits)

        if figpath is not None:
            fig.savefig(figpath)
    else:
        fig = None

    return grid, fig


def padded_stack(a, b, how='vert', where='+', shift=0):
    '''Merge 2-dimensional numpy arrays with different shapes

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
            - `'+'` -> `a` is above `b`
            - `'-'` -> `a` is below `b`
         - if how == 'horiz'
            - `'+'` -> `a` is to the left of `b`
            - `'-'` -> `a` is to the right of `b`
        See the examples for more info.
    shift : int (default = 0)
        The number of indices the second array should be shifted in
        axis other than the one being merged. In other words, vertically
        stacked arrays can be shifted horizontally, and horizontally
        stacked arrays can be shifted vertically.
    [a|b]_transform : function, lambda, or None (default)
        Individual transformations that will be applied to the arrays
        *prior* to being merged. This can be numeric of even alter the
        shapes (e.g., `np.flipud`, `np.transpose`)

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


    '''
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
            fill = (np.nan, np.nan)
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


def _outputfile(outputdir, filename):
    if outputdir is None:
        outputdir = '.'
    return os.path.join(outputdir, filename)


class _PointSet(object):
    def __init__(self, array):
        self._points = np.asarray(array)

    @property
    def points(self):
        return self._points
    @points.setter
    def points(self, value):
        self._points = np.asarray(value)

    @property
    def shape(self):
        return self.points.shape

    def transform(self, fxn, *args, **kwargs):
        self.points = fxn(self.points, *args, **kwargs)
        return self

    def transpose(self):
        return self.transform(np.transpose)

    def merge(self, other, how='vert', where='+', shift=0):
        return self.transform(padded_stack, other.points, how=how,
                              where=where, shift=shift)


class ModelGrid(object):
    def __init__(self, nodes_x, nodes_y):
        if not np.all(nodes_x.shape == nodes_y.shape):
            raise ValueError('input arrays must have the same shape')

        self._nodes_x = _PointSet(nodes_x)
        self._nodes_y = _PointSet(nodes_y)
        self._template = None
        self._cell_mask = np.zeros(self.cell_shape, dtype=bool)

    @property
    def nodes_x(self):
        '''_PointSet object of x-nodes'''
        return self._nodes_x
    @nodes_x.setter
    def nodes_x(self, value):
        self._nodes_x = value

    @property
    def nodes_y(self):
        return self._nodes_y
    @nodes_y.setter
    def nodes_y(self, value):
        '''_PointSet object of y-nodes'''
        self._nodes_y = value

    @property
    def cells_x(self):
        '''_PointSet object of x-cells'''
        xc = 0.25 * (
            self.xn[1:,1:] + self.xn[1:,:-1] +
            self.xn[:-1,1:] + self.xn[:-1,:-1]
        )
        return xc

    @property
    def cells_y(self):
        yc = 0.25 * (
            self.yn[1:,1:] + self.yn[1:,:-1] +
            self.yn[:-1,1:] + self.yn[:-1,:-1]
        )
        return yc

    @property
    def shape(self):
        return self.nodes_x.shape

    @property
    def cell_shape(self):
        return self.cells_x.shape

    @property
    def xn(self):
        '''shortcut to x-coords of nodes'''
        return self.nodes_x.points

    @property
    def yn(self):
        '''shortcut to y-coords of nodes'''
        return self.nodes_y.points

    @property
    def xc(self):
        '''shortcut to x-coords of nodes'''
        return self.cells_x

    @property
    def yc(self):
        '''shortcut to y-coords of nodes'''
        return self.cells_y

    @property
    def icells(self):
        '''rows of cells'''
        return self.cell_shape[1]

    @property
    def jcells(self):
        '''columns of cells'''
        return self.cell_shape[0]

    @property
    def inodes(self):
        '''rows of nodes'''
        return self.shape[1]

    @property
    def jnodes(self):
        '''columns of nodes'''
        return self.shape[0]

    @property
    def cell_mask(self):
        return self._cell_mask
    @cell_mask.setter
    def cell_mask(self, value):
        self._cell_mask = value

    @property
    def template(self):
        '''template shapefile'''
        return self._template
    @template.setter
    def template(self, value):
        self._template = value

    def transform(self, fxn, *args, **kwargs):
        self.nodes_x = self.nodes_x.transform(fxn, *args, **kwargs)
        self.nodes_y = self.nodes_y.transform(fxn, *args, **kwargs)
        return self

    def transpose(self):
        return self.transform(np.transpose)

    def fliplr(self):
        '''reverses the columns'''
        return self.transform(np.fliplr)

    def flipud(self):
        '''reverses the rows'''
        return self.transform(np.flipud)

    def merge(self, other, how='vert', where='+', shift=0):
        '''Merge with another grid

        Parameters
        ----------
        other : ModelGrid
            The other ModelGrid object.
        '''
        self.nodes_x = self.nodes_x.merge(other.nodes_x, how=how,
                                          where=where, shift=shift)
        self.nodes_y = self.nodes_y.merge(other.nodes_y, how=how,
                                          where=where, shift=shift)
        return self

    def mask_cells_with_polygon(self, polyverts, inside=True, inplace=True):
        polyverts = np.asarray(polyverts)
        if polyverts.ndim != 2:
            raise ValueError('polyverts must be a 2D array, or a '
                             'similar sequence')

        if polyverts.shape[1] != 2:
            raise ValueError('polyverts must be two columns of points')

        if polyverts.shape[0] < 3:
            raise ValueError('polyverts must contain at least 3 points')

        coords = self.as_coord_pairs(which='cells')
        mask = points_inside_poly(coords, polyverts).reshape(self.cell_shape)

        if not inside:
            mask = ~mask

        if inplace:
            self.cell_mask = np.bitwise_or(self.cell_mask, mask)

        else:
            return mask

    def writeGEFDCControlFile(self, outputdir=None, filename='gefdc.inp',
                              bathyrows=0, title='test'):
        outfile = _outputfile(outputdir, filename)

        gefdc = io._write_gefdc_control_file(
            outfile,
            title,
            self.inodes + 2,
            self.jnodes + 2,
            bathyrows
        )
        return gefdc

    def writeGEFDCCellFile(self, outputdir=None, filename='cell.inp',
                           usetriangles=False, maxcols=125):
        outfile = _outputfile(outputdir, filename)

        cells = io._write_cellinp(
            ~np.isnan(self.xn),
            outfile,
            triangle_cells=usetriangles,
            maxcols=maxcols,
            testing=True
        )
        return cells

    def writeGEFDCGridFile(self, outputdir=None, filename='grid.out'):
        outfile = _outputfile(outputdir, filename)
        df = io._write_gridout_file(self.xn, self.yn, outfile)
        return df

    def writeGEFDCGridextFile(self, outputdir, shift=2, filename='gridext.inp'):
        outfile = _outputfile(outputdir, filename)
        df = self.as_dataframe().stack(level='i', dropna=True).reset_index()
        df['i'] += shift
        df['j'] += shift
        io._write_gridext_file(df, outfile)
        return df

    def _plot_nodes(self, boundary=None, engine='mpl', ax=None, **kwargs):
        raise NotImplementedError
        if engine == 'mpl':
            return viz._plot_nodes_mpl(self.xn, self.yn, boundary=boundary,
                                       ax=ax, **kwargs)
        elif engine == 'bokeh':
            return viz._plot_nodes_bokeh(self.xn, self.yn, boundary=boundary,
                                         **kwargs)

    def plotCells(self, boundary=None, engine='mpl', ax=None, **kwargs):
        return viz.plotCells(self.xn, self.yn, engine=engine, **kwargs)

    def as_dataframe(self, usemask=False, which='nodes'):

        x, y = self._get_x_y(which, usemask=usemask)

        def make_cols(top_level):
            columns = pandas.MultiIndex.from_product(
                [[top_level], range(x.shape[1])],
                names=['coord', 'i']
            )
            return columns

        index = pandas.Index(range(x.shape[0]), name='j')
        easting_cols = make_cols('easting')
        northing_cols = make_cols('northing')

        easting = pandas.DataFrame(x, index=index, columns=easting_cols)
        northing = pandas.DataFrame(y, index=index, columns=northing_cols)
        return easting.join(northing)

    def as_coord_pairs(self, usemask=False, which='nodes'):
        x, y = self._get_x_y(which, usemask=usemask)
        return np.array(list(zip(x.flatten(), y.flatten())))

    def to_shapefile(self, outputfile, usemask=True, which='nodes',
                     river=None, reach=0, elev=None, template=None,
                     geom='Point', mode='w'):

        if usemask:
            if which == 'nodes':
                raise NotImplementedError('`usemask` not implemented for nodes')
            mask = self.cell_mask.copy()
        else:
            mask = None

        if template is None:
            template = self.template

        if geom.lower() == 'point':
            x, y = self._get_x_y(which)

            io.savePointShapefile(x, y, template, outputfile,
                                  mode=mode, river=river, reach=reach,
                                  elev=elev)

        elif geom.lower() in ('cell', 'cells', 'grid', 'polygon'):
            io.saveGridShapefile(self.xn, self.yn, mask, template,
                                 outputfile, mode=mode, river=river,
                                 reach=reach, elev=elev)
        else:
            raise ValueError("geom must be either 'Point' or 'Polygon'")

    def _get_x_y(self, which, usemask=False):
        if which == 'nodes' and usemask:
            raise ValueError("can only mask cells, not nodes")

        if which.lower() == 'nodes':
            x, y = self.xn, self.yn
        elif which.lower() == 'cells':
            x, y = self.xc, self.yc
        else:
            raise ValueError('`which` must be either "nodes" or "cells"')

        if usemask:
            x = np.ma.masked_array(x, self.cell_mask)
            y = np.ma.masked_array(y, self.cell_mask)

        return x, y

    @staticmethod
    def from_dataframes(df_x, df_y, icol='i'):
        nodes_x = df_x.unstack(level='i')
        nodes_y = df_y.unstack(level='i')
        return ModelGrid(nodes_x, nodes_y)

    @staticmethod
    def from_shapefile(shapefile, icol='ii', jcol='jj'):
        df = io.readGridShapefile(shapefile, icol=icol, jcol=jcol)
        return ModelGrid.from_dataframes(df['easting'], df['northing'])

    @staticmethod
    def from_Gridgen(gridgen):
        return ModelGrid(gridgen.x, gridgen.y)


