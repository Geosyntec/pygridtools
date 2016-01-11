from __future__ import division

import warnings

import numpy as np
import pandas

from pygridtools import misc
from pygridtools import iotools
from pygridtools import viz


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

        self._domain = None
        self._extent = None
        self._islands = None

    @property
    def nodes_x(self):
        """_PointSet object of x-nodes"""
        return self._nodes_x
    @nodes_x.setter
    def nodes_x(self, value):
        self._nodes_x = value

    @property
    def nodes_y(self):
        return self._nodes_y
    @nodes_y.setter
    def nodes_y(self, value):
        """_PointSet object of y-nodes"""
        self._nodes_y = value

    @property
    def cells_x(self):
        """_PointSet object of x-cells"""
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
        """shortcut to x-coords of nodes"""
        return self.nodes_x.points

    @property
    def yn(self):
        """shortcut to y-coords of nodes"""
        return self.nodes_y.points

    @property
    def xc(self):
        """shortcut to x-coords of cells/centroids"""
        return self.cells_x

    @property
    def yc(self):
        """shortcut to y-coords of cells/centroids"""
        return self.cells_y

    @property
    def icells(self):
        """rows of cells"""
        return self.cell_shape[1]

    @property
    def jcells(self):
        """columns of cells"""
        return self.cell_shape[0]

    @property
    def inodes(self):
        """rows of nodes"""
        return self.shape[1]

    @property
    def jnodes(self):
        """columns of nodes"""
        return self.shape[0]

    @property
    def cell_mask(self):
        return self._cell_mask
    @cell_mask.setter
    def cell_mask(self, value):
        self._cell_mask = value

    @property
    def template(self):
        """template shapefile"""
        return self._template
    @template.setter
    def template(self, value):
        self._template = value

    @property
    def domain(self):
        return self._domain
    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def extent(self):
        return self._extent
    @extent.setter
    def extent(self, value):
        self._extent = value

    @property
    def islands(self):
        return self._islands
    @islands.setter
    def islands(self, value):
        self._islands = value

    def transform(self, fxn, *args, **kwargs):
        self.nodes_x = self.nodes_x.transform(fxn, *args, **kwargs)
        self.nodes_y = self.nodes_y.transform(fxn, *args, **kwargs)
        return self

    def transpose(self):
        return self.transform(np.transpose)

    def fliplr(self):
        """reverses the columns"""
        return self.transform(np.fliplr)

    def flipud(self):
        """reverses the rows"""
        return self.transform(np.flipud)

    def merge(self, other, how='vert', where='+', shift=0):
        """ Merge with another grid using pygridtools.misc.padded_stack.

        Parameters
        ----------
        other : ModelGrid
            The other ModelGrid object.
        how : optional string (default = 'vert')
            The method through wich the arrays should be stacked.
            `'Vert'` is analogous to `np.vstack`. `'Horiz'` maps to
            `np.hstack`.
        where : optional string (default = '+')
            The placement of the arrays relative to each other. Keeping
            in mind that the origin of an array's index is in the
            upper-left corner, `'+'` indicates that the second array
            will be placed at higher index relative to the first array.
            Essentially:
              - if how == 'vert'
                - `'+'` -> `a` is above (higher index) `b`
                - `'-'` -> `a` is below (lower index) `b`
              - if how == 'horiz'
                - `'+'` -> `a` is to the left of `b`
                - `'-'` -> `a` is to the right of `b`
            See the examples and pygridtools.misc.padded_stack for more
            info.
        shift : int (default = 0)
            The number of indices the second array should be shifted in
            axis other than the one being merged. In other words,
            vertically stacked arrays can be shifted horizontally,
            and horizontally stacked arrays can be shifted vertically.

        Returns
        -------
        self (operates in-place)

        Notes
        -----
        The ``cell_mask`` attribute is not automatically updated
        following merge operates. See the Examples section on handling
        this manually.

        Examples
        --------
        >>> domain1 = pandas.DataFrame({
            'x': [2, 5, 5, 2],
            'y': [6, 6, 4, 4],
            'beta': [1, 1, 1, 1]
        })
        >>> domain2 = pandas.DataFrame({
            'x': [6, 11, 11, 5],
            'y': [5, 5, 3, 3],
            'beta': [1, 1, 1, 1]
        })
        >>> grid1 = pgt.makeGrid(domain=domain1, nx=6, ny=5, rawgrid=False)
        >>> grid2 = pgt.makeGrid(domain=domain2, nx=8, ny=7, rawgrid=False)
        >>> grid1.merge(grid2, how='horiz')
        >>> # update the cell mask to include new NA points:
        >>> grid1.cell_mask = np.ma.masked_invalid(grid1.xc).mask

        See Also
        --------
        pygridtools.padded_stack

        """

        self.nodes_x = self.nodes_x.merge(other.nodes_x, how=how,
                                          where=where, shift=shift)
        self.nodes_y = self.nodes_y.merge(other.nodes_y, how=how,
                                          where=where, shift=shift)
        return self

    def mask_cells_with_polygon(self, polyverts, use_centroids=True,
                                min_nodes=3, inside=True,
                                use_existing=True, triangles=False,
                                inplace=True):

        """ Create mask for the cells of the ModelGrid with a polygon.

        Parameters
        ----------
        polyverts : sequence of a polygon's vertices
            A sequence of x-y pairs for each vertex of the polygon.
        use_centroids : bool (default = True)
            When True, the cell centroid will be used to determine
            whether the cell is "inside" the polygon. If False, the
            nodes are used instead.
        min_nodes : int (default = 3)
            Only used when ``use_centroids`` is False. This is the
            minimum number of nodes inside the polygon required to mark
            the cell as "inside". Must be greater than 0, but no more
            than 4.
        inside : bool (default = True)
            Toggles masking of cells either *inside* (True) or *outside*
            (False) the polygon.
        triangles : bool
            Not yet implemented.
        use_existing : bool (default = True)
            When True, the newly computed mask is combined (via a
            bit-wise `or` opteration) with the existing ``cell_mask``
            attribute of the MdoelGrid.
        inplace : bool (default = True):
            If True, the ``cell_mask`` attribute of the ModelGrid is set
            to the returned masked and None is returned. Otherwise, the
            a new mask is returned the ``cell_mask`` attribute of the
            ModelGrid is unchanged.

        Returns
        -------
        cell_mask : np.array of bools
            The final mask to be applied to the cells of the ModelGrid.

        """

        if triangles:
            raise NotImplementedError("triangles are not yet implemented.")

        if use_centroids:
            cell_mask = misc.mask_with_polygon(self.xc, self.yc, polyverts, inside=inside)
        else:
            if min_nodes <= 0 or min_nodes > 4:
                raise ValueError("`min_nodes` must be greater than 0 and no more than 4.")

            _node_mask = misc.mask_with_polygon(self.xn, self.yn, polyverts, inside=inside).astype(int)
            cell_mask = (
                _node_mask[1:, 1:] + _node_mask[:-1, :-1] +
                _node_mask[:-1, 1:] + _node_mask[1:, :-1]
            ) >= min_nodes
            cell_mask = cell_mask.astype(bool)


        if use_existing:
            cell_mask = np.bitwise_or(self.cell_mask, cell_mask)

        if inplace:
            self.cell_mask = cell_mask

        return cell_mask

    def writeGEFDCControlFile(self, outputdir=None, filename='gefdc.inp',
                              bathyrows=0, title='test'):
        outfile = iotools._outputfile(outputdir, filename)

        gefdc = iotools._write_gefdc_control_file(
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
        outfile = iotools._outputfile(outputdir, filename)

        iotools._write_cellinp(cells, outputfile=outfile,
                                  flip=True, maxcols=maxcols)
        return cells

    def writeGEFDCGridFile(self, outputdir=None, filename='grid.out'):
        outfile = iotools._outputfile(outputdir, filename)
        df = iotools._write_gridout_file(self.xn, self.yn, outfile)
        return df

    def writeGEFDCGridextFile(self, outputdir, shift=2, filename='gridext.inp'):
        outfile = iotools._outputfile(outputdir, filename)
        df = self.to_dataframe().stack(level='i', dropna=True).reset_index()
        df['i'] += shift
        df['j'] += shift
        iotools._write_gridext_file(df, outfile)
        return df

    def plotCells(self, engine='mpl', ax=None,
                  usemask=True, cell_kws=None,
                  domain_kws=None, extent_kws=None,
                  showisland=True, island_kws=None):

        if cell_kws is None:
            cell_kws = {}
        fig = viz.plotCells(self.xn, self.yn, engine=engine, ax=ax,
                            mask=self.cell_mask, **cell_kws)

        if domain_kws is not None:
            fig = viz.plotDomain(data=self.domain, engine=engine, ax=ax, **domain_kws)

        if extent_kws:
            fig = viz.plotBoundaries(extent=self.extent, engine=engine, ax=ax, **extent_kws)

        if island_kws:
            fig = viz.plotBoundaries(islands=self.islands, engine=engine, ax=ax, **island_kws)

        return fig

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

    def to_dataframe(self, usemask=False, which='nodes'):

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

    def to_coord_pairs(self, usemask=False, which='nodes'):
        x, y = self._get_x_y(which, usemask=usemask)
        return np.array(list(zip(x.flatten(), y.flatten())))

    def to_shapefile(self, outputfile, usemask=True, which='cells',
                     river=None, reach=0, elev=None, template=None,
                     geom='Polygon', mode='w', triangles=False):


        if template is None:
            template = self.template

        if geom.lower() == 'point':
            x, y = self._get_x_y(which, usemask=usemask)
            iotools.savePointShapefile(x, y, template, outputfile,
                                  mode=mode, river=river, reach=reach,
                                  elev=elev)

        elif geom.lower() in ('cell', 'cells', 'grid', 'polygon'):
            if usemask:
                mask = self.cell_mask.copy()
            else:
                mask = None
            x, y = self._get_x_y('nodes', usemask=False)
            iotools.saveGridShapefile(x, y, mask, template,
                                 outputfile, mode=mode, river=river,
                                 reach=reach, elev=elev,
                                 triangles=triangles)
            if which == 'cells':
                warnings.warn("polygons always constructed from nodes")
        else:
            raise ValueError("geom must be either 'Point' or 'Polygon'")

    @staticmethod
    def from_dataframe(df, xcol='easting', ycol='northing', icol='i'):
        nodes_x = df_x[xcol].unstack(level='i')
        nodes_y = df_y[ycol].unstack(level='i')
        return ModelGrid(nodes_x, nodes_y)

    @staticmethod
    def from_shapefile(shapefile, icol='ii', jcol='jj'):
        df = iotools.readGridShapefile(shapefile, icol=icol, jcol=jcol)
        return ModelGrid.from_dataframes(df['easting'], df['northing'])

    @staticmethod
    def from_Gridgen(gridgen):
        return ModelGrid(gridgen.x, gridgen.y)


def makeGrid(ny, nx, domain, bathydata=None, verbose=False,
             rawgrid=True, **gparams):
    """ Generate and (optionally) visualize a grid, and create input
    files for the GEDFC preprocessor (makes grid input files for GEFDC).

    Parameters
    ----------
    ny, nx : int
        The number of rows and columns that will make up the grid's
        *nodes*. Note the final grid *cells* will be (ny-1) by (nx-1).
    domain : optional pandas.DataFrame or None (default)
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
    rawgrid : bool (default = True)
        When True, returns a pygridgen.Gridgen object. Otherwise, a
        pygridtools.ModelGrid object is returned.
    **gparams : optional kwargs
        Parameters to be passed to the pygridgen.grid.Gridgen constructor.
        Only used if `makegrid` = True and `domain` is not None.
        `ny` and `nx` are absolutely required. Optional values include:

        ul_idx : optional int (default = 0)
            The index of the what should be considered the upper left
            corner of the grid boundary in the `xbry`, `ybry`, and
            `beta` inputs. This is actually more arbitrary than it
            sounds. Put it some place convenient for you, and the
            algorthim will conceptually rotate the boundary to place
            this point in the upper left corner. Keep that in mind when
            specifying the shape of the grid.
        focus : optional pygridgen.Focus instance or None (default)
            A focus object to tighten/loosen the grid in certain
            sections.
        proj : option pyproj projection or None (default)
            A pyproj projection to be used to convert lat/lon
            coordinates to a projected (Cartesian) coordinate system
            (e.g., UTM, state plane).
        nnodes : optional int (default = 14)
            The number of nodes used in grid generation. This affects
            the precision and computation time. A rule of thumb is that
            this should be equal to or slightly larger than
            -log10(precision).
        precision : optional float (default = 1.0e-12)
            The precision with which the grid is generated. The default
            value is good for lat/lon coordinate (i.e., smaller
            magnitudes of boundary coordinates). You can relax this to
            e.g., 1e-3 when working in state plane or UTM grids and
            you'll typically get better performance.
        nppe : optional int (default = 3)
            The number of points per internal edge. Lower values will
            coarsen the image.
        newton : optional bool (default = True)
            Toggles the use of Gauss-Newton solver with Broyden update
            to determine the sigma values of the grid domains. If False
            simple iterations will be used instead.
        thin : optional bool (default = True)
            Toggle to True when the (some portion of) the grid is
            generally narrow in one dimension compared to another.
        checksimplepoly : optional bool (default = True)
            Toggles a check to confirm that the boundary inputs form a
            valid geometry.
        verbose : optional bool (default = True)
            Toggles the printing of console statements to track the
            progress of the grid generation.

    Returns
    -------
    grid : pygridgen.grid.Gridgen obejct

    Notes
    -----
    If your boundary has a lot of points, this really can take quite
    some time. Setting verbose=True will help track the progress of the
    grid generattion.

    See Also
    --------
    pygridgen.Gridgen, pygridgen.csa, pygridtools.ModelGrid

    """

    try:
        import pygridgen
    except ImportError: # pragma: no cover
        raise ImportError("`pygridgen` not installed. Cannot make grid.")

    if verbose:
        print('generating grid')

    grid = pygridgen.Gridgen(domain.x, domain.y, domain.beta, (ny, nx), **gparams)

    if verbose:
        print('interpolating bathymetry')

    newbathy = misc.interpolateBathymetry(bathydata, grid.x_rho, grid.y_rho,
                                          xcol='x', ycol='y', zcol='z')
    if rawgrid:
        return grid
    else:
        return ModelGrid.from_Gridgen(grid)
