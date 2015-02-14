import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
import pandas
import fiona

import pygridgen

from . import viz
from . import io


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
        print('  generating fake bathymetry data')
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
            grid = Grid(coords.x, coords.y, coords.beta, (ny, nx), **gparams)
        else:
            raise ValueError("must provide `grid` if `makegrid` = False")
    if verbose:
        print('interpolating bathymetry')
    newbathy = interpolateBathymetry(bathydata, grid, xcol='x',
                                     ycol='y', zcol='z')

    if verbose:
        print('writing `gefdc` input files')
    if title is None:
        title = 'Grid of the Unnamed River'

    if outdir is None:
        outdir = '.'

    io.writeGEFDCInputFiles(grid, newbathy, outdir, title)

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


class _NodeSet(object):
    def __init__(self, nodes):
        self._nodes = np.asarray(nodes)

    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, value):
        self._nodes = np.asarray(value)

    def transform(self, fxn, *args, **kwargs):
        self.nodes = fxn(self.nodes, *args, **kwargs)
        return self

    def transpose(self):
        return self.transform(np.transpose)

    def merge(self, other, how='vert', where='+', shift=0):
        return self.transform(padded_stack, other.nodes, how=how,
                              where=where, shift=shift)


class ModelGrid(object):
    def __init__(self, nodes_x, nodes_y):
        if not np.all(nodes_x.shape == nodes_y.shape):
            raise ValueError('input arrays must have the same shape')
        self._nodes_x = _NodeSet(nodes_x)
        self._nodes_y = _NodeSet(nodes_y)

    @property
    def nodes_x(self):
        return self._nodes_x
    @nodes_x.setter
    def nodes_x(self, value):
        self._nodes_x = value

    @property
    def nodes_y(self):
        return self._nodes_y
    @nodes_y.setter
    def nodes_y(self, value):
        self._nodes_y = value

    @property
    def x(self):
        return self.nodes_x.nodes

    @property
    def y(self):
        return self.nodes_y.nodes

    def transform(self, fxn, *args, **kwargs):
        self.nodes_x = self.nodes_x.transform(fxn, *args, **kwargs)
        self.nodes_y = self.nodes_y.transform(fxn, *args, **kwargs)
        return self

    def transpose(self):
        return self.transform(np.transpose)

    def merge(self, other, how='vert', where='+', shift=0):
        self.nodes_x = self.nodes_x.merge(other.nodes_x, how=how,
                                          where=where, shift=shift)
        self.nodes_y = self.nodes_y.merge(other.nodes_y, how=how,
                                          where=where, shift=shift)
        return self


def _add_second_col_level(levelval, olddf):
    '''
    Takes a simple index on a dataframe's columns and adds a new level
    with a single value.
    E.g., df.columns = ['res', 'qual'] -> [('Infl' ,'res'), ('Infl', 'qual')]
    '''
    if isinstance(olddf.columns, pandas.MultiIndex):
        raise ValueError('Dataframe already has MultiIndex on columns')

    colarray = [[levelval]*len(olddf.columns), olddf.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)
    newdf = olddf.copy()
    newdf.columns = colindex
    return newdf


def _process_array(arr, transpose, transform=None):
    '''
    Helper function pull out data from masked arrays and to
    transpose + flip upside down (needed when stitching a wide
    grid to a long one).
    '''
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.data

    if transpose:
        arr = arr.T

    if transform is not None:
        arr = transform(arr)

    return arr


def _grid_attr_to_df(xattr, yattr, transpose, transform=None):
    '''
    Helper function to pull convert numpy.ndarray attributes of Gridgen
    objects into pandas.DataFrames with i/j indices.
    '''
    if xattr.shape[0] != yattr.shape[0] or xattr.shape[1] != yattr.shape[1]:
        raise ValueError('shapes are not equivalent')

    xattr = _process_array(xattr, transpose, transform=transform)
    yattr = _process_array(yattr, transpose, transform=transform)

    data = []
    cols = ['j', 'i', 'northing', 'easting']
    for jj in range(xattr.shape[0]):
        for ii in range(xattr.shape[1]):
            data.append({
                'j': jj,
                'i': ii,
                'easting': xattr[jj, ii],
                'northing': yattr[jj, ii]
            })

    df = pandas.DataFrame(data).set_index(['j', 'i']).unstack(level='i')
    return df


class Grid(pygridgen.Gridgen):
    '''
    Basic object that provides access to the attribures of a Gridgen
    object in a pandas.DataFrame format.

    Parameters
    ----------
    grid : pygridgen.grid.Gridgen objects
        The dang grid, jeez.
    transpose : optional bool (default = False)
        Toggles whether the code will transpose and flip upside-down
        a the grid. Useful for when a wide grid will be stitched on to
        the end of a long grid (or vise versa).
    transform : function, lambda expression, or None (default)
        function to flip the arrays to align the indices with previous
        or subsequent grids as needed. Recommend non-None values are
        numpy.fliplr or numpy.flipud (or a lambda to do both)
        -----
        |   |
        | L |
        | O |
        | N |
        | G |
        |   |
        |   ---------------------------
        |          | WIDE (transpose) |
        -------------------------------

    Attributes
    ----------
    u/v - northing/easting of the u and v velocity vectors for each cell
    nodes - northing/easting of cell verices (lower left corner)
    centers - northing/easting of cell centroids
    psi - northing/easting of ????

    Notes
    -----
    `centers` come come the `grid.x_rho` and `grid.y_rho` attributes.

        transform = kwargs.pop('transform', None)
        transpose = kwargs.pop('transpose', None)
        super(Grid, self).__init__(*args, **kwargs)
    '''

    def __init__(self, *args, **kwargs):
        # input props
        self._transform = kwargs.pop('transform', None)
        self._transpose = kwargs.pop('transpose', False)

        # grid
        super(Grid, self).__init__(*args, **kwargs)

        self.merged_grids = []

        # remaining props
        self._u = None
        self._v = None
        self._nodes = None
        self._centers = None
        self._psi = None

    @property
    def transform(self):
        return self._transform
    @transform.setter
    def transform(self, value):
        self._transform = value

    @property
    def transpose(self):
        return self._transpose
    @transpose.setter
    def transpose(self, value):
        self._transpose = value

    @property
    def u(self):
        if self._u is None:
            self._u = _grid_attr_to_df(
                self.x_u, self.y_u, self.transpose, transform=self.transform
            )
        return self._u
    @u.setter
    def u(self, value):
        self._u = value

    @property
    def v(self):
        if self._v is None:
            self._v = _grid_attr_to_df(
                self.x_v, self.y_v, self.transpose, transform=self.transform
            )
        return self._v
    @v.setter
    def v(self, value):
        self._v = value

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = _grid_attr_to_df(
                self.x, self.y, self.transpose, transform=self.transform
            )
        return self._nodes
    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def centers(self):
        if self._centers is None:
            self._centers = _grid_attr_to_df(
                self.x_rho, self.y_rho, self.transpose, transform=self.transform
            )
        return self._centers
    @centers.setter
    def centers(self, value):
        self._centers = value

    @property
    def psi(self):
        if self._psi is None:
            self._psi = _grid_attr_to_df(
                self.x_psi, self.y_psi, self.transpose, transform=self.transform
            )
        return self._psi
    @psi.setter
    def psi(self, value):
        self._psi = value

    # @property
    # def elev(self):
    #     if self._elev is None and self.elev is not None:
    #         z = self.elev
    #         if self.transpose:
    #             z = z.T
    #         if self.transform is not None:
    #             z = self.transform(z)
    #         self._elev = pandas.DataFrame(z)

    #     return self._elev
    # @elev.setter
    # def elev(self, value):
    #     self._elev = value
    def clear_df_attributes(self):
        self._u = None
        self._v = None
        self._nodes = None
        self._centers = None
        self._psi = None
        #self.elev = None

    # def generate_grid(cleardfs=True):
    #     super(Grid, self).generate_grid()
    #     # if cleardfs:
    #     #     self._clear_df_attributes()

    def _merge_attr(self, attr, otherdf, how='j', where='+', offset=0):

        easting = mergePoints(
            getattr(self, attr).xs('easting', level=0, axis=1, drop_level=True),
            getattr(otherdf, attr).xs('easting', level=0, axis=1, drop_level=True),
            how=how, where=where, offset=offset
        )

        northing = mergePoints(
            getattr(self, attr).xs('northing', level=0, axis=1, drop_level=True),
            getattr(otherdf, attr).xs('northing', level=0, axis=1, drop_level=True),
            how=how, where=where, offset=offset
        )

        easting = _add_second_col_level('easting', easting)
        northing = _add_second_col_level('northing', northing)
        return easting.join(northing)

    def mergeGrid(self, othergdf, how='j', where='+', offset=0):
        self.merged_grids.append(othergdf)
        self.u = self._merge_attr('u', othergdf, how=how,
                                  where=where, offset=offset)

        self.v = self._merge_attr('v', othergdf, how=how,
                                  where=where, offset=offset)

        self.nodes = self._merge_attr('nodes', othergdf, how=how,
                                  where=where, offset=offset)

        self.centers = self._merge_attr('centers', othergdf, how=how,
                                  where=where, offset=offset)

        self.psi = self._merge_attr('psi', othergdf, how=how,
                                  where=where, offset=offset)
        # if self.elev is not None:
        #     self.elev = mergePoints(self.elev, othergdf.elev, how=how,
        #                             where=where, offset=offset)

    def writeGridOut(self, outputfile):
        nodes = self.nodes.stack(level='i', dropna=False)
        with open(outputfile, 'w') as out:
            out.write('## {:d} x {:d}\n'.format(self.nx, self.ny))
            nodes.to_csv(out, sep=' ', na_rep='NaN', index=False,
                         header=False, float_format='%.3f')

    def writeToShapefile(self, outfile, geomtype='Point', template=None,
                              river=None, reach=0):
        if template is None:
            template = 'gis/template/schema_template.shp'

        X = self.nodes['easting'].values
        Y = self.nodes['northing'].values
        #elev = self.elev.values
        if geomtype == 'Point':
            io.savePointShapefile(X, Y, template, outfile, 'w',
                                  river=river, reach=reach, elev=None)
        elif geomtype == 'Polygon':
            io.saveGridShapefile(X, Y, template, outfile, 'w',
                                 river=river, reach=reach, elev=None)
        else:
            raise ValueError('`geomtype {} is not supported'.format(geomtype))


def mergePoints(ref_df, concat_df, how='j', where='+', offset=0):

    errormsg_how = "how must be either i or j"
    errormsg_where = "where must be either + or -"
    valid_values_how = ['j', 'i']
    valid_values_where = ['-', '+']
    if how not in valid_values_how:
        raise ValueError(errormsg_how)

    if where not in valid_values_where:
        raise ValueError(errormsg_where)


    if how == 'j':
        offset_df = concat_df.rename(columns=lambda c: c + offset)

        if where == '+':
            merged = pandas.concat([ref_df, offset_df])

        else:
            merged = pandas.concat([offset_df, ref_df])

    else:
        merged = mergePoints(
            ref_df.T, concat_df.T, how='j', where=where, offset=offset
        )
        merged = merged.T

    return merged.reset_index(drop=True).T.reset_index(drop=True).T

