from pathlib import Path
from textwrap import dedent

import numpy
import pandas
from shapely.geometry import Point
import geopandas

from pygridtools import iotools
from pygridtools import misc
from pygridtools import validate


GEFDC_TEMPLATE = dedent("""\
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


def write_cellinp(cell_array, outputfile='cell.inp', mode='w',
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
        cell_array = numpy.flipud(cell_array)

    nrows, ncols = cell_array.shape

    if cell_array.shape[1] > maxcols:
        first_array = cell_array[:, :maxcols]
        second_array = cell_array[:, maxcols:]

        write_cellinp(first_array, outputfile=outputfile, mode=mode,
                      writeheader=writeheader, rowlabels=rowlabels,
                      maxcols=maxcols, flip=False)
        write_cellinp(second_array, outputfile=outputfile, mode='a',
                      writeheader=False, rowlabels=False,
                      maxcols=maxcols, flip=False)

    else:
        columns = numpy.arange(1, maxcols + 1, dtype=int)
        colstr = [list('{:04d}'.format(c)) for c in columns]
        hundreds = ''.join([c[1] for c in colstr])
        tens = ''.join([c[2] for c in colstr])
        ones = ''.join([c[3] for c in colstr])

        with Path(outputfile).open(mode) as outfile:
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


def write_gefdc_control_file(outfile, title, max_i, max_j, bathyrows):
    gefdc = GEFDC_TEMPLATE.format(title, max_i, max_j, bathyrows)

    with Path(outfile).open('w') as f:
        f.write(gefdc)

    return gefdc


def write_gridout_file(xcoords, ycoords, outfile):
    xcoords, ycoords = validate.xy_array(xcoords, ycoords, as_pairs=False)

    ny, nx = xcoords.shape
    df = pandas.DataFrame({
        'x': xcoords.flatten(),
        'y': ycoords.flatten()
    })

    with Path(outfile).open('w') as f:
        f.write('## {:d} x {:d}\n'.format(nx, ny))
        df.to_csv(f, sep=' ', index=False, header=False,
                  na_rep='NaN', float_format='%.3f')

    return df


def write_gridext_file(tidydf, outfile, icol='ii', jcol='jj',
                       xcol='easting', ycol='northing'):
    # make sure cols are in the right order
    df = tidydf[[icol, jcol, xcol, ycol]]

    with Path(outfile).open('w') as f:
        df.to_csv(f, sep=' ', index=False, header=False,
                  float_format=None)

    return df


def convert_gridext_to_shp(inputfile, outputfile, crs=None, river='na', reach=0):
    """ Converts gridext.inp from the rtools to a shapefile with
    `geomtype = 'Point'`.

    Parameters
    ----------
    inputfile : string
        Path and filename of the gridext.inp file
    outputfile : string
        Path and filename of the destination shapefile
    crs : string, optional
        A geopandas/proj/fiona-compatible string describing the coordinate
        reference system of the x/y values.
    river : optional string (default = None)
        The river to be listed in the shapefile's attribute table.
    reach : optional int (default = 0)
        The reach of the river to be listed in the shapefile's attribute
        table.

    Returns
    -------
    geopandas.GeoDataFrame

    """

    errmsg = 'file {} not found'
    if not Path(inputfile).exists:
        raise ValueError(errmsg.format(inputfile))

    gdf = (
        pandas.read_csv(inputfile, sep='\s+', engine='python', header=None,
                        dtype={'ii': int, 'jj': int, 'x': float, 'y': float},
                        names=['ii', 'jj', 'x', 'y'])
              .assign(id=lambda df: df.index)
              .assign(ii_jj=lambda df:
                      df['ii'].astype(str).str.pad(3, fillchar='0') + '_' +
                      df['jj'].astype(str).str.pad(3, fillchar='0'))
              .assign(elev=0.0, river=river, reach=reach)
              .assign(geometry=lambda df: df.apply(lambda r: Point((r['x'], r['y'])), axis=1))
              .drop(['x', 'y'], axis='columns')
              .pipe(geopandas.GeoDataFrame, geometry='geometry', crs=crs)
    )

    gdf.to_file(outputfile)

    return gdf


class GEFDCWriter:
    """
    Convenience class to write the GEFDC files for a ModelGrid

    Parameters
    ----------
    mg : pygridtools.ModelGrid
    directory : str or Path
        Where all of the files will be saved

    """

    def __init__(self, mg, directory):
        self.mg = mg
        self.directory = Path(directory)

    def control_file(self, filename='gefdc.inp', bathyrows=0,
                     title=None):
        """
        Generates the GEFDC control (gefdc.inp) file for the EFDC grid
        preprocessor.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.
        bathyrows : int, optional
            The number of rows in the grid's bathymetry data file.
        title : str, optional
            The title of the grid as portrayed in ``filename``.

        Returns
        -------
        gefdc : str
            The text of the output file.

        """

        if not title:
            title = 'Model Grid from pygridtools'

        outfile = self.directory / filename

        gefdc = write_gefdc_control_file(
            outfile,
            title,
            self.mg.inodes + 1,
            self.mg.jnodes + 1,
            bathyrows
        )
        return gefdc

    def cell_file(self, filename='cell.inp', triangles=False,
                  maxcols=125):
        """
        Generates the cell definition/ASCII-art file for GEFDC.

        .. warning:
           This whole thing is probably pretty buggy.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.
        triangles : bool, optional
            Toggles the inclusion of triangular cells.

            .. warning:
               This is experimental and probably buggy if it has been
               implmented at all.

        maxcols : int, optional
            The maximum number of columns to write to each row. Cells
            beyond this number will be writted in separate section at
            the bottom of the file.

        Returns
        -------
        cells : str
            The text of the output file.

        """

        cells = misc.make_gefdc_cells(
            ~numpy.isnan(self.mg.xn), self.mg.cell_mask, triangles=triangles
        )
        outfile = self.directory / filename

        write_cellinp(cells, outputfile=outfile, flip=True, maxcols=maxcols)
        return cells

    def gridout_file(self, filename='grid.out'):
        """
        Writes to the nodes as coordinate pairs for GEFDC.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe of node coordinate pairs.

        """

        outfile = self.directory / filename
        df = write_gridout_file(self.mg.xn, self.mg.yn, outfile)
        return df

    def gridext_file(self, filename='gridext.inp', shift=2):
        """
        Writes to the nodes and I/J cell index as to a file for GEFDC.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.
        shift : int, optional
            The shift that should be applied to the I/J index. The
            default value to 2 means that the first cell is at (2, 2)
            instead of (0, 0).

        Returns
        -------
        df : pandas.DataFrame
            The dataframe of coordinates and I/J index.

        """
        outfile = self.directory / filename
        df = self.mg.to_dataframe().stack(level='ii', dropna=True).reset_index()
        df['ii'] += shift
        df['jj'] += shift
        write_gridext_file(df, outfile)
        return df
