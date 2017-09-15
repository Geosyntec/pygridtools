from __future__ import division

import os
from collections import OrderedDict
from textwrap import dedent

import numpy as np
import pandas
import fiona

from pygridtools import misc
from pygridtools import validate


gefdc = GEFDC_TEMPLATE = dedent("""\
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
""")


def _outputfile(outputdir, filename):
    if outputdir is None:
        outputdir = '.'
    return os.path.join(outputdir, filename)


def loadBoundaryFromShapefile(shapefile, betacol='beta', reachcol=None,
                              sortcol=None, upperleftcol=None,
                              filterfxn=None):
    """ Loads boundary points from a shapefile.

    Parameters
    ----------
    shapefile : string
        Path to the shapefile containaing boundary points.
        Expected schema of the shapefile...

        - order: numeric sort order of the points
        - beta: the 'beta' parameter used in grid generation to define turning points

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
        - upperleft

    """

    # load and filter the data
    with fiona.open(shapefile, 'r') as shp:
        if filterfxn is None:
            reach_bry = list(filter(lambda x: True, shp))
        else:
            reach_bry = list(filter(filterfxn, shp))

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
            'beta': _get_col_val(record, betacol, default=0),
            'order': _get_col_val(record, sortcol, default=n * 5),
            'reach': _get_col_val(record, reachcol, default='main'),
            'upperleft': _get_col_val(record, upperleftcol, default=False),
        })
    df = pandas.DataFrame(data).fillna(0)

    # return just the right columns and sort the data
    cols = ['x', 'y', 'beta', 'upperleft', 'reach', 'order']
    return df[cols].sort_values(by=['order'])


def loadPolygonFromShapefile(shapefile, filterfxn=None, squeeze=True):
    """ Load polygons (e.g., water bodies, islands) from a shapefile.

    Parameters
    ----------
    shapefile : string
        Path to the shapefile containaing boundary points.
    filterfxn : function or lambda expression or None (default)
        Pulls out relevant points from the boundary shapefile. Defaults
        to `True` to consider everything.
    squeeze : optional bool (default = True)
        Set to True to return an array if only 1 record is present.
        Otherwise, a list of arrays will be returned.

    Returns
    -------
    boundary : array or list of arrays
        N x 2 array of the coordinates of the boundary.

    Notes
    -----
    Multipart geometries are not supported. If a multipart geometry is
    present in a record, only the first part will be loaded.

    Z-coordinates are also not supported. Only x-y coordinates will be
    loaded.

    """

    # load and filter the data
    with fiona.open(shapefile, 'r') as shp:
        if filterfxn is None:
            polygons = list(filter(lambda x: True, shp))
        else:
            polygons = list(filter(filterfxn, shp))

    data = []
    for record in polygons:
        data.append(np.array(record['geometry']['coordinates'])[0, :, :2])

    if len(data) == 1 and squeeze:
        data = data[0]

    return data


def dumpGridFiles(grid, filename):
    """
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

    """

    with open(filename, 'w') as f:
        f.write('## {:d} x {:d}\n'.format(grid.nx, grid.ny))

        df = pandas.DataFrame({'x': grid.x.flatten(), 'y': grid.y.flatten()})
        df.to_csv(f, sep=' ', na_rep='NaN', index=False,
                  header=False, float_format='%.3f')


def savePointShapefile(X, Y, template, outputfile, mode='w', river=None,
                       reach=0, elev=None):
    """ Saves grid-related attributes of a pygridgen.Gridgen object to a
    shapefile with geomtype = 'Point'.

    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    template : string
        Path to a template shapfiles with the desired schema.
    outputfile : string
        Path to the point shapefile to which the data will be written.
    mode : optional string (default = 'w')
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

    """

    # check that the `mode` is a valid value
    mode = validate.file_mode(mode)

    # check that X and Y are have the same shape, NaN cells
    X, Y = validate.equivalent_masks(X, Y)

    # check elev shape
    elev = validate.elev_or_mask(X, elev, 'elev', offset=0)

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
                # check that nothing is masked (outside of the river)
                if not (X.mask[jj, ii]):
                    row += 1

                    # build the coords
                    coords = (X[jj, ii], Y[jj, ii])

                    # build the attributes
                    props = OrderedDict(
                        id=int(row), river=river, reach=reach,
                        ii=int(ii + 2), jj=int(jj + 2), elev=float(elev[jj, ii]),
                        ii_jj='{:02d}_{:02d}'.format(ii + 2, jj + 2)
                    )

                    # append to the output file
                    record = misc.make_record(row, coords, 'Point', props)
                    out.write(record)


def saveGridShapefile(X, Y, mask, template, outputfile, mode,
                      river=None, reach=0, elev=None, triangles=False):
    """ Saves a shapefile of quadrilaterals representing grid cells.


    Parameters
    ----------
    X, Y : numpy (masked) arrays, same dimensions
        Attributes of the gridgen object representing the x- and y-coords.
    mask : numpy array or None
        Array describing which cells to mask (exclude) from the output.
        Shape should be N-1 by M-1, where N and M are the dimensions of
        `X` and `Y`.
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
        The elevation of the grid cells. Shape should be N-1 by M-1,
        where N and M are the dimensions of `X` and `Y` (like `mask`).
    triangles : optional bool (default = False)
        If True, triangles can be included

    Returns
    -------
    None

    """

    # check that `mode` is valid
    mode = validate.file_mode(mode)

    # check X, Y shapes
    Y = validate.elev_or_mask(X, Y, 'Y', offset=0)

    # check elev shape
    elev = validate.elev_or_mask(X, elev, 'elev', offset=0)

    # check the mask shape
    mask = validate.elev_or_mask(X, mask, 'mask', offset=1)

    X = np.ma.masked_invalid(X)
    Y = np.ma.masked_invalid(Y)
    ny, nx = X.shape

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
        for ii in range(nx - 1):
            for jj in range(ny - 1):
                if not (np.any(X.mask[jj:jj + 2, ii:ii + 2]) or mask[jj, ii]):
                    row += 1
                    Z = elev[jj, ii]
                    # build the array or coordinates
                    coords = misc.make_poly_coords(
                        xarr=X[jj:jj + 2, ii:ii + 2],
                        yarr=Y[jj:jj + 2, ii:ii + 2],
                        zpnt=Z, triangles=triangles
                    )

                    # build the attributes
                    props = OrderedDict(
                        id=row, river=river, reach=reach,
                        ii=ii + 2, jj=jj + 2, elev=Z,
                        ii_jj='{:02d}_{:02d}'.format(ii + 2, jj + 2)
                    )

                    # append to file is coordinates are not masked
                    # (masked = beyond the river boundary)
                    if coords is not None:
                        record = misc.make_record(row, coords, 'Polygon', props)
                        out.write(record)


def readGridShapefile(shapefile, icol='ii', jcol='jj', othercols=None,
                      expand=1):

    data = []
    if othercols is None:
        othercols = []

    with fiona.open(shapefile) as shp:
        for record in shp:
            geomtype = record['geometry']['type']
            if geomtype == 'Point':
                geom = np.array(record['geometry']['coordinates'])
            elif geomtype == 'Polygon':
                raise NotImplementedError("can only read points for now")

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
    columns = ['easting', 'northing']
    columns.extend(othercols)
    return df.sort_index()[columns]


def _write_cellinp(cell_array, outputfile='cell.inp', mode='w',
                   writeheader=True, rowlabels=True,
                   maxcols=125, flip=True):
    """Writes the cell.inp input file from an array of cell definitions.

    Parameters
    ----------
    cell_array : numpy array
        Integer array of the values written to ``outfile``.
    outputfile : optional string (default = "cell.inp")
        Path *and* filename to the output file. Yes, you have to tell it
        to call the file cell.inp
    maxcols : optional int (default = 125)
        Number of columns at which cell.inp should be wrapped. ``gefdc``
        requires this to be 125.
    flip : optional bool (default = True)
        Numpy arrays have their origin in the upper left corner, so in
        a sense south is up and north is down. This means that arrays
        need to be flipped before writing to "cell.inp". Unless you are
        _absolutely_sure_ that your array has been flipped already,
        leave this parameter as True.

    Returns
    -------
    None

    See also
    --------
    _make_gefdc_cells

    """

    if flip:
        cell_array = np.flipud(cell_array)

    nrows, ncols = cell_array.shape

    if cell_array.shape[1] > maxcols:
        first_array = cell_array[:, :maxcols]
        second_array = cell_array[:, maxcols:]
        # padwidth = _second_array.shape[1] - maxcols
        # second_array = np.pad(_second_array, ((0, 0), (0, padwidth)),
        #                       mode='constant', constant_values=0)

        _write_cellinp(first_array, outputfile=outputfile, mode=mode,
                       writeheader=writeheader, rowlabels=rowlabels,
                       maxcols=maxcols, flip=False)
        _write_cellinp(second_array, outputfile=outputfile, mode='a',
                       writeheader=False, rowlabels=False,
                       maxcols=maxcols, flip=False)

    else:
        columns = np.arange(1, maxcols + 1, dtype=int)
        colstr = [list('{:04d}'.format(c)) for c in columns]
        hundreds = ''.join([c[1] for c in colstr])
        tens = ''.join([c[2] for c in colstr])
        ones = ''.join([c[3] for c in colstr])

        with open(outputfile, mode) as outfile:
            if writeheader:
                title = 'C -- cell.inp for EFDC model by pygridtools\n'
                outfile.write(title)
                outfile.write('C    {}\n'.format(hundreds[:ncols]))
                outfile.write('C    {}\n'.format(tens[:ncols]))
                outfile.write('C    {}\n'.format(ones[:ncols]))

            for n, row in enumerate(cell_array):
                row_number = nrows - n
                row_strings = row.astype(str)
                cell_text = ''.join(row_strings.tolist())
                if rowlabels:
                    row_text = '{0:3d}  {1:s}\n'.format(
                        int(row_number), cell_text
                    )
                else:
                    row_text = '     {0:s}\n'.format(cell_text)

                outfile.write(row_text)


def _write_gefdc_control_file(outfile, title, max_i, max_j, bathyrows):
    gefdc = GEFDC_TEMPLATE.format(title, max_i, max_j, bathyrows)

    with open(outfile, 'w') as f:
        f.write(gefdc)

    return gefdc


def _write_gridout_file(xcoords, ycoords, outfile):
    if xcoords.shape != ycoords.shape:
        raise ValueError('input dimensions must be equivalent')

    ny, nx = xcoords.shape
    df = pandas.DataFrame({
        'x': xcoords.flatten(),
        'y': ycoords.flatten()
    })

    with open(outfile, 'w') as f:
        f.write('## {:d} x {:d}\n'.format(nx, ny))
        df.to_csv(f, sep=' ', na_rep='NaN', index=False,
                  header=False, float_format='%.3f')

    return df


def _write_gridext_file(tidydf, outfile, icol='i', jcol='j',
                        xcol='easting', ycol='northing'):
    # make sure cols are in the right order
    df = tidydf[[icol, jcol, xcol, ycol]]

    with open(outfile, 'w') as f:
        df.to_csv(f, sep=' ', index=False, header=False,
                  float_format=None)


def gridextToShapefile(inputfile, outputfile, template, river='na', reach=0):
    """ Converts gridext.inp from the rtools to a shapefile with
    `geomtype = 'Point'`.

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

    """

    errmsg = 'file {} not found in {}'
    if not os.path.exists(inputfile):
        raise ValueError(errmsg.format(inputfile, os.getcwd()))

    df = pandas.read_csv(
        inputfile,
        sep='\s+',
        header=None,
        names=['i', 'j', 'x', 'y'],
        dtype={'i': int, 'j': int, 'x': float, 'y': float}
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
        record = misc.make_record(int(row.name), coords, 'Point', props)
        try:
            outfile.write(record)
            return 1
        except:  # pragma: no cover
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
