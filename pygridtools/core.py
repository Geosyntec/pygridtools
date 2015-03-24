from __future__ import division

import warnings

import numpy as np
import pandas
import pygridgen

from . import misc
from . import io
from . import viz


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
        return self.transform(misc.padded_stack, other.points, how=how,
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

    def mask_cells_with_polygon(self, polyverts, use_cells=True,
                                inside=True, triangles=False,
                                min_nodes=3, inplace=True):
        polyverts = np.asarray(polyverts)
        if polyverts.ndim != 2:
            raise ValueError('polyverts must be a 2D array, or a '
                             'similar sequence')

        if polyverts.shape[1] != 2:
            raise ValueError('polyverts must be two columns of points')

        if polyverts.shape[0] < 3:
            raise ValueError('polyverts must contain at least 3 points')

        if use_cells:
            cells = self.as_coord_pairs(which='cells')
            cell_mask = misc.points_inside_poly(
                cells, polyverts
            ).reshape(self.cell_shape)
        else:
            nodes = self.as_coord_pairs(which='nodes')
            _node_mask = misc.points_inside_poly(
                nodes, polyverts
            ).reshape(self.shape)

            cell_mask = (
                _node_mask[1:, 1:] + _node_mask[:-1, :-1] +
                _node_mask[:-1, 1:] + _node_mask[1:, :-1]
            ) < min_nodes,
        if not inside:
            cell_mask = ~cell_mask

        if inplace:
            self.cell_mask = np.bitwise_or(self.cell_mask, cell_mask)

        else:
            return cell_mask

    def writeGEFDCControlFile(self, outputdir=None, filename='gefdc.inp',
                              bathyrows=0, title='test'):
        outfile = io._outputfile(outputdir, filename)

        gefdc = io._write_gefdc_control_file(
            outfile,
            title,
            self.inodes + 1,
            self.jnodes + 1,
            bathyrows
        )
        return gefdc

    def writeGEFDCCellFile(self, outputdir=None, filename='cell.inp',
                           triangles=False, maxcols=125):

        cells = misc.make_gefdc_cells(
            ~np.isnan(self.xn), self.cell_mask, triangles=triangles
        )
        outfile = io._outputfile(outputdir, filename)

        io._write_cellinp(cells, outputfile=outfile,
                                  flip=True, maxcols=maxcols)
        return cells

    def writeGEFDCGridFile(self, outputdir=None, filename='grid.out'):
        outfile = io._outputfile(outputdir, filename)
        df = io._write_gridout_file(self.xn, self.yn, outfile)
        return df

    def writeGEFDCGridextFile(self, outputdir, shift=2, filename='gridext.inp'):
        outfile = io._outputfile(outputdir, filename)
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

    def to_shapefile(self, outputfile, usemask=True, which='cells',
                     river=None, reach=0, elev=None, template=None,
                     geom='Polygon', mode='w', triangles=False):


        x, y = self._get_x_y(which, usemask=usemask)

        if geom.lower() == 'point':
            io.savePointShapefile(x, y, template, outputfile,
                                  mode=mode, river=river, reach=reach,
                                  elev=elev)

        elif geom.lower() in ('cell', 'cells', 'grid', 'polygon'):
            if usemask:
                mask = self.cell_mask.copy()
            else:
                mask = None
            io.saveGridShapefile(x, y, mask, template,
                                 outputfile, mode=mode, river=river,
                                 reach=reach, elev=elev,
                                 triangles=triangles)
            if which == 'cells':
                warnings.warn("polygons always constructed from nodes")
        else:
            raise ValueError("geom must be either 'Point' or 'Polygon'")

    def _get_x_y(self, which, usemask=False):
        if which.lower() == 'nodes':
            if usemask:
                raise ValueError("can only mask cells, not nodes")
            else:
                x, y = self.xn, self.yn

        elif which.lower() == 'cells':
            x, y = self.xc, self.yc
            if usemask:
                x = np.ma.masked_array(x, self.cell_mask)
                y = np.ma.masked_array(y, self.cell_mask)

        else:
            raise ValueError('`which` must be either "nodes" or "cells"')

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
    newbathy = misc.interpolateBathymetry(bathydata, grid, xcol='x',
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
