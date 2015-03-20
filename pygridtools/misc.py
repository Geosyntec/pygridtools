import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pandas
import fiona

import pygridgen

from . import viz


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


def make_gefdc_cells(node_mask, cell_mask=None, use_triangles=False):
    '''
    Take an array defining the nodes as wet (1) or dry (0) create the
    array of cell values needed for GEFDC

    Input
    -----
    node_mask : numpy bool array (N x M)
        Bool array specifying if a *node* is present in the raw
        (unmasked) grid.
    cell_mask : optional numpy bool array (N-1 x M-1) or None (default)
        Bool array specifying if a cell should be masked (e.g. due to
        being an island or something like that).
    use_triangles : optional bool (default = False)
        Currently not implemented. Will eventually enable the writting of
        triangular cells when True.

    Returns
    -------
    cell_array : numpy array
        Integer array of the values written to ``outfile``.

    '''

    triangle_cells = {
        2: 3,
        3: 2,
        0: 4,
        1: 1,
    }
    land_cell = 0
    water_cell = 5
    bank_cell = 9

    # I can't figure this out
    if use_triangles:
        warnings.warn('triangle are experimental')

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
                elif n_wet == 3 and use_triangles:
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


def _outputfile(outputdir, filename):
    if outputdir is None:
        outputdir = '.'
    return os.path.join(outputdir, filename)
