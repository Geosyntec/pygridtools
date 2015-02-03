from __future__ import division

import os
import re
import pdb
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas
import fiona

def loadBoundaryFromShapefile(shapefile, betacol='beta', reachcol=None,
                              sortcol=None, upperleftcol=None,
                              filterfxn=None):
    '''
    Loads boundary points from a shapefile.

    Parameters
    ----------
    shapefile : string
        Path to the shapefile containaing boundary points.
        Expected schema of the shapefile:
            order: numeric sort order of the points
            beta: the 'beta' parameter used in grid generation to define
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
        Pulls out relevant points from the boundary shapefile. Defaults
        to `True` to consider everything.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame of the boundary points with the following columns:
          - x (easting)
          - y (northing)
          - beta (turning parameter)
          - order (for sorting)
          - reach
          -upperleft

    '''

    # load and filter the data
    shp = fiona.open(shapefile, 'r')
    if filterfxn is None:
        reach_bry = filter(lambda x: True, shp)
    else:
        reach_bry = filter(filterfxn, shp)
    shp.close()

    def _get_col_val(rec, col, default=None):
        if col is not None:
            val = rec['properties'][col]
        else:
            val = default

        return val


    # stuff the data into a dataframe
    data = []
    for n, record in enumerate(reach_bry):
        data.append({
            'x': record['geometry']['coordinates'][0],
            'y': record['geometry']['coordinates'][1],
            'beta': record['properties'][betacol],
            'order': _get_col_val(record, sortcol, default=n*5),
            'reach': _get_col_val(record, reachcol, default='main'),
            'upperleft': _get_col_val(record, upperleftcol, default=False),
        })
    df = pandas.DataFrame(data).fillna(0)

    # return just the right columns and sort the data
    cols = ['x', 'y', 'beta', 'upperleft', 'reach', 'order']
    return df[cols].sort(columns='order')


def dumpGridFiles(grid, filename):
    '''
    Dump vertices from a pygridgen object to file in the standard grid.out
    format.

    Parameters
    ----------
    grid : pygridgen.Gridgen object
        The grid to be dumped
    filename : string
        path and filename of the output file

    Returns
    -------
    None
    '''

    with open(filename, 'w') as f:
        f.write('## {:d} x {:d}\n'.format(grid.nx, grid.ny))

        df = pandas.DataFrame({'x': grid.x.flatten(), 'y': grid.y.flatten()})
        df.to_csv(f, sep=' ', na_rep='NaN', index=False,
                  header=False, float_format='%.3f')


def makeQuadCoords(xarr, yarr, zpnt=None):
    '''
    Makes an array for coordinates suitable for building quadrilateral
    geometries in shapfiles via fiona.

    Parameters
    ----------
    xarr, yarr : numpy arrays
        Arrays (2x2) of x coordinates and y coordinates for each vertex of
        the quadrilateral.
    zpnt : optional float or None (default)
        If provided, this elevation value will be assied to all four vertices

    Returns
    -------
    coords : numpy array
        An array suitable for feeding into fiona as the geometry of a record.

    '''

    if not isinstance(xarr, np.ma.MaskedArray) or xarr.mask.sum() == 0:
        if zpnt is None:
            coords = np.vstack([
                np.hstack([xarr[0,:], xarr[1,::-1]]),
                np.hstack([yarr[0,:], yarr[1,::-1]])
            ]).T
        else:
            xcoords = np.hstack([xarr[0,:], xarr[1,::-1]])
            ycoords = np.hstack([yarr[0,:], yarr[1,::-1]])
            zcoords = np.array([zpnt] * xcoords.shape[0])
            coords = np.vstack([xcoords, ycoords, zcoords]).T
    else:
        coords = None

    return coords


def makeRecord(ID, coords, geomtype, props):
    '''
    Creates a records for the fiona package to append to a shapefile

    Parameters
    ----------
    ID : int
        The record ID number
    coords : tuple or array-like
        The x-y coordinates of the geometry. For Points, just a tuple. An
        array or list of tuples for LineStrings or Polygons
    geomtype : string
        A valid GDAL/OGR geometry specification (e.g. LineString, Point,
        Polygon)
    props : dict or collections.OrderedDict
        A dict-like object defining the attributes of the record

    Returns
    -------
    record : dict
        A nested dictionary suitable for the fiona package to append to a
        shapefile

    Notes
    -----
    This is ignore the mask of a MaskedArray. That might be bad.

    '''
    if not geomtype in ['Point', 'LineString', 'Polygon']:
        raise ValueError('Geometry {} not suppered'.format(geomtype))

    if isinstance(coords, np.ma.MaskedArray):
        coords = coords.data

    if isinstance(coords, np.ndarray):
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


def savePointShapefile(X, Y, template, outputfile, mode, river=None, reach=0, elev=None):
    '''
    Saves grid-related attributes of a pygridgen.Gridgen object to a
    shapefile with geomtype = 'Point'

    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    template : string
        Path to a template shapfiles with the desired schema.
    outputfile : string
        Path to the point shapefile to which the data will be written.
    mode : string
        The mode with which `outputfile` will be written.
        (i.e., 'a' for append and 'w' for write)
    river : optional string (default = None)
        The river to be listed in the shapefile's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the shapefile's attribute
        table.
    elev : optional array or None (defauly)
        The elevation of the grid cells. Array dimensions must be 1 less than
        X and Y.

    Returns
    -------
    None

    '''
    # pull out just the data attributes if necessary
    if isinstance(X, np.ma.MaskedArray):
        X = X.data

    if isinstance(Y, np.ma.MaskedArray):
        Y = Y.data

    # check that the `mode` is a valid value
    if mode not in ['a', 'w']:
        raise ValueError('`mode` must be either "a" (append) or "w" (write)')

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError('`X` and `Y` must have the same shape')

    if elev is None:
        elev = np.zeros(X.shape)
    elif elev.shape[0] != X.shape[0]  or elev.shape[1] != X.shape[1]:
        raise ValueError('`elev` dimensions must be compatible `X` and `Y`')

    # load the template
    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    src_schema['geometry'] = 'Point'

    # start writting or appending to the output
    with fiona.open(
        outputfile, mode,
        driver=src_driver,
        crs=src_crs,
        schema=src_schema
    ) as out:
        row = 0
        for ii in range(X.shape[1]):
            for jj in range(X.shape[0]):

                # check that nothign is masked (outside of the river)
                if not np.isnan(X[jj, ii]) and not np.isnan(Y[jj, ii]):
                    row += 1

                    # build the coords
                    coords = (X[jj, ii], Y[jj, ii])

                    # build the attributes
                    props = OrderedDict(
                        id=row, river=river, reach=reach,
                        ii=ii+2, jj=jj+2, elev=elev[jj,ii],
                        ii_jj='{:02d}_{:02d}'.format(ii+2, jj+2)
                    )

                    # append to the output file
                    record = makeRecord(row, coords, 'Point', props)
                    out.write(record)


def saveGridShapefile(X, Y, template, outputfile, mode, river=None, reach=0, elev=None):
    '''
    Saves a shapefile of quadrilaterals representing grid cells.


    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    template : string
        Path to a template shapfiles with the desired schema.
    outputfile : string
        Path to the point shapefile to which the data will be written.
    mode : string
        The mode with which `outputfile` will be written.
        (i.e., 'a' for append and 'w' for write)
    river : optional string (default = None)
        The river to be listed in the shapefile's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the shapefile's attribute
        table.
    elev : optional array or None (defauly)
        The elevation of the grid cells. Array dimensions must be 1 less than
        X and Y.

    Returns
    -------
    None

    Notes
    -----
    You need to have attached an `elev` attribute to the grid object that is
    an array of the same shape as grid.x_rho and grid.y_rho (cell centers).
    If no such attribute exists, 0 will be used as the elevation
    '''
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError('`X` and `Y` must have the same shape')


    # make sure the vertices are in masked arrays
    if isinstance(X, np.ndarray):
        x_vert = np.ma.masked_invalid(X)
        y_vert = np.ma.masked_invalid(Y)
    else:
        x_vert = X.copy()
        y_vert = Y.copy()

    ny, nx = x_vert.shape

    if elev is None:
        elev = np.ma.masked_invalid(np.zeros((ny, nx)))
    elif elev.shape[0] != ny - 1 or elev.shape[1] != nx - 1:
        raise ValueError('`elev` dimensions must be compatible `X` and `Y`')

    # check that `mode` is valid
    if mode not in ['a', 'w']:
        raise ValueError('`mode` must be either "a" (append) or "w" (write)')

    # load the template
    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    src_schema['geometry'] = 'Polygon'

    # start writting or appending to the output
    with fiona.open(
        outputfile, mode,
        driver=src_driver,
        crs=src_crs,
        schema=src_schema
    ) as out:
        row = 0
        for ii in range(nx-1):
            for jj in range(ny-1):
                if not np.any(x_vert.mask[jj:jj+2, ii:ii+2]):
                    row += 1

                    # pull out the elevation or assign default
                    if elev is not None:
                        elevation = elev[jj, ii]
                    else:
                        elevation = 0

                    # build the array or coordinates
                    coords = makeQuadCoords(
                        xarr=x_vert[jj:jj+2, ii:ii+2],
                        yarr=y_vert[jj:jj+2, ii:ii+2],
                        zpnt=elevation
                    )

                    # build the attributes
                    props = OrderedDict(
                        id=row, river=river, reach=reach,
                        ii=ii+2, jj=jj+2, elev=elevation,
                        ii_jj='{:02d}_{:02d}'.format(ii+2, jj+2)
                    )

                    # append to file is coordinates are not masked
                    # (masked = beyond the river boundary)
                    if coords is not None:
                        record = makeRecord(row, coords, 'Polygon', props)
                        out.write(record)


def saveXYShapefile(tidydata, template, outputfile, mode='w',
                    xcol='x', ycol='y', icol='i', jcol='j'):


    # load the template
    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    src_schema['geometry'] = 'Point'

    def row2shp_record(row, shpio):
        ii = int(row[icol])
        jj = int(row[jcol])
        coords = (float(row[xcol]), float(row[ycol]))
        props = OrderedDict(
            id=int(row.name), river='none', reach='none',
            ii=ii+2, jj=jj+2, elev=0,
            ii_jj='{:02d}_{:02d}'.format(ii+2, jj+2)
        )
        # makeRecord(ID, coords, geomtype, props):
        shpio.write(makeRecord(row.name, coords, 'Point', props))
        return 0

    # start writting or appending to the output
    with fiona.open(
        outputfile, mode,
        driver=src_driver,
        crs=src_crs,
        schema=src_schema
    ) as out:
        _ = tidydata.apply(lambda row: row2shp_record(row, out), axis=1)


def readPointShapefile(*arg, **kwargs):
    return readGridShapefile(*args, **kwargs)


def readGridShapefile(shapefile, icol='ii', jcol='jj', expand=1, othercols=None):
    if isinstance(expand, int):
        factor = expand
    data = []
    with fiona.open(shapefile) as shp:
        for record in shp:
            geomtype = record['geometry']['type']
            if geomtype == 'Point':
                geom = np.array(record['geometry']['coordinates'])
            elif geomtype == 'Polygon':
                geom = np.array(record['geometry']['coordinates']).flatten()

            dfrow = {
                'i': expand * (record['properties'][icol] - 2),
                'j': expand * (record['properties'][jcol] - 2),
                'easting': geom[0],
                'northing': geom[1],
            }
            for col in othercols:
                dfrow[col] = record['properties'][col]

            data.append(dfrow)

    df = pandas.DataFrame(data).set_index(['j', 'i'])
    # df = df.reindex(index=pandas.MultiIndex.from_product([
    #     df.index.get_level_values('j').unique(),
    #     df.index.get_level_values('i').unique(),
    # ], names=['j', 'i']))
    df.sort(inplace=True)
    return df


def writeGEFDCInputFiles(grid, bathy, outputdir, title):

    bathyfile = os.path.join(outputdir, 'depdat.inp')
    bathy.to_csv(bathyfile, sep=' ', header=False, index=False,
                 float_format='%0.1f')
    _write_gefdc_control_file(outputdir, title, grid.nx+2, grid.ny+2, bathy.shape[0])


def _write_cellinp(bool_node_array, outfilename, triangle_cells=False,
                   maxcols=125):
    '''
    Take an array defining the nodes as wet (1) or dry (0) and writes
    the cell.inp input file.

    Input
    -----
    bool_node_array : numpy array of integers (0's and 1's)
        Not really a boolean array per se. Just an 0 or 1 array defining
        a node as wet or dry.
    outfilename : string
        Path *and* filename to the outputfile. Yes, you have to tell it
        to call the file cell.inp
    triangle_cells : optional bool, default is False
        Toggles the definition if triangular cells. Very greedy.
        Probably easier to keep off and add them yourself.
    '''
    triangle_dict = {
        0: 3,
        1: 2,
        2: 4,
        3: 1,
    }

    # I can't figure this out
    if triangle_cells:
        raise NotImplementedError('you should add triangular cells yourself')

    # basic stuff, array shapes, etc
    ny, nx = bool_node_array.shape
    cells = np.zeros(np.array(bool_node_array.shape)+2, dtype=int)

    # loop through each *node*
    for j in range(0, ny):
        for i in range(0, nx):
            # pull out the 4 nodes defining the cell (call it a quad)
            quad = bool_node_array[j:j+2, i:i+2]
            n_wet = quad.sum()

            # the first row of *cells* is alawys all 0's
            if j == 0 and n_wet > 0:
                cells[j, i+1] = 9

            # if all 4 nodes are wet (=1), then the cell is 5
            if n_wet == 4:
                cells[j+1, i+1] = 5

            # # if only 3  are wet, might be a triangle, but...
            # elif quad.sum() in (3, 2, 1):
            #     # this ignored since we already raised an error
            #     if triangle_cells:
            #         dry_node = np.argmin(quad.flatten())
            #         cells[j+1, i+1] = triangle_dict[dry_node]
            #     # and this always happens instead (make it a bank)
            #     else:
            #         cells[j+1, i+1] = 9

            # # 2 wet cells mean it's a bank
            # elif quad.sum() == 2 or quad.sum() == 1:
            #     cells[j+1, i+1] = 9

    for cj in range(1, cells.shape[0]-1):
        for ci in range(1, cells.shape[1]-1):
            if cells[cj, ci] == 5:
                block = cells[cj-1:cj+2, ci-1:ci+2]
                bj, bi = np.nonzero(block == 0)
                for bjj, bii in zip(bj, bi):
                    cells[cj+bjj-1, ci+bii-1] = 9

    cells = np.flipud(cells)

    nrows = cells.shape[0]
    ncols = cells.shape[1]

    nchunks = np.ceil(ncols / maxcols)
    if ncols > maxcols:
        final_cells = np.zeros((nrows*nchunks, maxcols), dtype=int)
        for n in np.arange(nchunks):
            col_start = n * maxcols
            col_stop = (n+1) * maxcols

            row_start = n * nrows
            row_stop = (n+1) * nrows

            cells_to_move = cells[:, col_start:col_stop]
            final_cells[row_start:row_stop, 0:cells_to_move.shape[1]] = cells_to_move
    else:
        final_cells = cells.copy()

    columns = np.arange(1, 126, dtype=int)
    colstr = [list('{:04d}'.format(c)) for c in columns]
    hundreds = ''.join([c[1] for c in colstr])
    tens = ''.join([c[2] for c in colstr])
    ones = ''.join([c[3] for c in colstr])

    with open(outfilename, 'w') as outfile:
        outfile.write(
            'C -- cell.inp for EFDC model. Generated by Paul/python\n'
        )
        outfile.write('C    {}\n'.format(hundreds[:ncols]))
        outfile.write('C    {}\n'.format(tens[:ncols]))
        outfile.write('C    {}\n'.format(ones[:ncols]))
        for n, row in enumerate(final_cells):
            row_number = nrows * np.ceil(n/nrows) - n + 1
            row_strings = row.astype(str)
            cell_text = ''.join(row_strings.tolist())
            row_text = '{0: 3d}  {1:s}\n'.format(int(row_number), cell_text)

            outfile.write(row_text)

    return cells


def _write_gefdc_control_file(outputdir, title, max_i, max_j, bathyrows):
    gefdcfile = os.path.join(outputdir, 'gefdc.inp')
    gefdc = '''
C1  TITLE
C1  (LIMITED TO 80 CHARACTERS)
    '{0}'
C2  INTEGER INPUT
C2  NTYPE   NBPP    IMIN    IMAX    JMIN    JMAX    IC   JC
    0       0       1       {1}     1       {2}     {1}  {2}
C3  GRAPHICS GRID INFORMATION
C3  ISGG    IGM     JGM     DXCG    DYCG    NWTGG
    0       0       0       0.      0.      1
C4  CARTESIAN AND GRAPHICS GRID COORDINATE DATA
C4  CDLON1  CDLON2  CDLON3  CDLAT1  CDLAT2  CDLAT3
    0.      0.      0.      0.      0.      0.
C5  INTEGER INPUT
C5  ITRXM   ITRHM   ITRKM   ITRGM   NDEPSM  NDEPSMF DEPMIN  DDATADJ
    200     200     200     200     4000    0       0       0
C6  REAL INPUT
C6  RPX     RPK     RPH     RSQXM   RSQKM   RSQKIM  RSQHM   RSQHIM  RSQHJM
    1.8     1.8     1.8     1.E-12  1.E-12  1.E-12  1.E-12  1.E-12  1.E-12
C7  COORDINATE SHIFT PARAMETERS
C7  XSHIFT  YSHIFT  HSCALE  RKJDKI  ANGORO
    0.      0.      1.      1.      5.0
C8  INTERPOLATION SWITCHES
C8  ISIRKI  JSIRKI  ISIHIHJ JSIHIHJ
    1       0       0       0
C9  NTYPE = 7 SPECIFIED INPUT
C9  IB      IE      JB      JE      N7RLX   NXYIT   ITN7M   IJSMD   ISMD    JSMD    RP7     SERRMAX
C10 NTYPE = 7 SPECIFIED INPUT
C10 X       Y       IN ORDER    (IB,JB) (IE,JB) (IE,JE) (IB,JE)
C11 DEPTH INTERPOLATION SWITCHES
C11 ISIDEP  NDEPDAT CDEP    RADM    ISIDPTYP    SURFELEV    ISVEG   NVEGDAT NVEGTYP
    1       {3:d}     2       0.5     1           0.0         0       0       0
C12 LAST BOUNDARY POINT INFORMATION
C12 ILT     JLT     X(ILT,JLT)      Y(ILT,JLT)
    0       0       0.0             0.0
C13 I   J       X(I,J)          Y(I,J)
'''.format(title, max_i, max_j, bathyrows)

    with open(gefdcfile, 'w') as f:
        f.write(gefdc)


def gridextToShapefile(inputfile, outputfile, template, river='na', reach=0):
    '''
    Converts gridext.inp from the rtools to a shapefile with
    geomtype = 'Point'

    Parameters
    ----------
    inputfile : string
        Path and filename of the gridext.inp file
    outputfile : string
        Path and filename of the destination shapefile
    template : string
        Path to a template shapfiles with the desired schema.
    river : optional string (default = None)
        The river to be listed in the shapefile's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the shapefile's attribute
        table.

    Returns
    -------
    None

    '''
    errmsg = 'file {} not found in {}'
    if not os.path.exists(inputfile):
        raise ValueError(errmsg.format(inputfile, os.getcwd()))

    df = pandas.read_csv(
        inputfile,
        sep='\s+',
        header=None,
        names=['i', 'j', 'x', 'y'],
        dtype={'i':int, 'j':int, 'x':float, 'y':float}
    )

    # load the template
    if not os.path.exists(template):
        raise ValueError(errmsg.format(inputfile, os.getcwd()))

    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    src_schema['geometry'] = 'Point'

    def row2record(row, outfile):
        coords = (float(row.x), float(row.y))
        props = OrderedDict(
            id=int(row.name), river=river, reach=reach,
            ii=int(row.i), jj=int(row.j), elev=0,
            ii_jj='{:03d}_{:03d}'.format(int(row.i), int(row.j))
        )
        record = makeRecord(int(row.name), coords, 'Point', props)
        try:
            outfile.write(record)
            return 1
        except:
            return 0

    # start writting or appending to the output
    with fiona.open(
        outputfile, 'w',
        driver=src_driver,
        crs=src_crs,
        schema=src_schema
    ) as out:
        df['in gis layer'] = df.apply(
            lambda row: row2record(row, out), axis=1
        )

    return df[df['in gis layer'] == 0]
