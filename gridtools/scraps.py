
## iotools
def readRect(n, folder=None):
    '''
    Reads the rect.* files output from the stock gridgen examples
    '''
    if folder is None:
        folder = os.path.join('examples', 'stock')

    rect = pandas.read_table(os.path.join(folder, 'rect.{}'.format(n)),
                             sep='\s', names=['x', 'y'])
    return pandas.concat([rect, rect.loc[:0]])


def readXY(n, folder=None):
    '''
    Reads the xy.* files output from the stock gridgen examples
    '''
    if folder is None:
        folder = os.path.join('examples', 'stock')

    xy = pandas.read_table(os.path.join(folder, 'xy.{}'.format(n)),
                           sep='\s+', names=['x', 'y', 'beta'],
                           comment='#')
    return xy.dropna(subset=['x', 'y'])


def readGrid(n, folder=None):
    '''
    Reads the grid.* files output from the stock gridgen examples
    '''
    if folder is None:
        folder = os.path.join('examples', 'stock')

    grid = pandas.read_table(os.path.join(folder, 'grid.{}'.format(n)),
                             sep='\s',
                             names=['x', 'y'], comment='#')
    return grid.dropna(subset=['x', 'y'])


def saveBankShapefiles(grid, template, outputfile, mode, reach=0):

    plinetype = 'LineString'
    if grid.nx < grid.ny:
        x = grid.x
        y = grid.y
    else:
        x = grid.x.T
        y = grid.y.T

    lbank = np.array([zip(x[:,0], y[:, 0])])[0]
    rbank = np.array([zip(x[:,-1], y[:, -1])])[0]
    lines = [
        {
            'name': 'left',
            'data': lbank,
        }, {
            'name': 'right',
            'data': rbank,
        }, {
            'name': 'center',
            'data': (0.5 * (lbank + rbank)),
        },
    ]

    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    #src_schema['geometry'] = plinetype

    with fiona.open(outputfile, mode, driver=src_driver, crs=src_crs, schema=src_schema) as out:
        for n, line in enumerate(lines, 1):
            record = {
                'id': 0,
                'geometry': {
                    'coordinates': line['data'].tolist(),
                    'type': plinetype
                },
                'properties': {
                    'reach': reach,
                    'Id': n,
                    'side': line['name']
                }
            }
            out.write(record)

    return lines


def saveTransectShapefiles(grid, template, outputfile, mode, reach=0):
    plinetype = 'LineString'
    if grid.nx < grid.ny:
        x = grid.x
        y = grid.y
        niter = grid.ny
    else:
        x = grid.x.T
        y = grid.y.T
        niter = grid.nx

    with fiona.open(template, 'r') as src:
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

    #src_schema['geometry'] = plinetype

    with fiona.open(outputfile, mode, driver=src_driver, crs=src_crs, schema=src_schema) as out:
        for n in range(niter):
            xdata = x[n,:]
            ydata = y[n,:]
            record = {
                'id': n,
                'geometry': {
                    'coordinates': np.vstack([xdata, ydata]).T.tolist(),
                    'type': plinetype
                },
                'properties': {
                    'reach': reach,
                    'Id': n,
                    'side': 'NA'
                }
            }
            out.write(record)

    return None



def writeGridgenInput(n, grid, ul_idx=None):
    '''
    writes the input files for gridgen-c

    '''
    reach = pandas.DataFrame({
        'x': grid.xbry.flatten(),
        'y': grid.ybry.flatten(),
        'beta': grid.beta.flatten()
    })

    reach['beta_str'] = reach['beta'].apply(
        lambda x: '' if x == 0 else str(x)
    )

    #rowname = reach.iloc[ul_idx].name
    #reach.loc[rowname, 'beta_str'] = '1*'

    reach[['x', 'y', 'beta_str']].to_csv(
        'PortageCreek/Reach_{:02d}/xy.in'.format(n),
        index=False,
        header=False,
        sep=' '
    )

    parameters = """\
input xy.in
output grid.out
nx {0:d}
ny {1:d}
nnodes {2:d}
precision 1.0e-12
newton 1
rectangle rect.out
sigmas sigmas.out
""".format(grid.nx, grid.ny, grid.nnodes)

    with open('PortageCreek/Reach_{:02d}/params.in'.format(n), 'w') as paramfile:
        paramfile.write(parameters)


def _do_pygridgen(coords, ny, nx, sanitizebeta=True, **gridparams):
    '''
    deprecated
    '''

    betamap = {
        '1': 1,
        '1*': 1,
        '-1': -1,
    }

    if sanitizebeta:
        coords['beta'] = coords.beta.apply(lambda b: betamap.get(b, 0))

    grid = octant.grid.Gridgen(coords.x, coords.y, coords.beta, (ny, nx),
                               **gridparams)
    return grid


def gridgenExample(n, folder=None, xlim=None, ylim=None):
    '''
    Makes a nice graphical representation of the gridgen examples
    (after running them)
    '''
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))
    plotting.plotXY(n, ax=ax1, folder=folder)
    plotting.plotGrid(n, ax=ax1, folder=folder)
    plotting.plotRect(n, ax=ax2, folder=folder)
    if xlim is not None:
        ax1.set_xlim(xlim)

    if ylim is not None:
        ax1.set_ylim(ylim)
    return fig, (ax1, ax2)


def pygridgenExample(n, ax=None, markerprms={}, sanitizebeta=True, **gridparams):
    '''
    Plots the grid generated with pygridgen/octant from the input files
    of the stock gridgen examples
    '''
    coords = readXY(n)

    ny = int(gridparams.pop('ny'))
    nx = int(gridparams.pop('nx'))
    grid = _do_pygridgen(coords, ny, nx, sanitizebeta=sanitizebeta, **gridparams)
    fig, ax = plotting.plotPygridgen(grid, ax=ax)
    plotXY(n, ax=ax)
    return fig, grid

def focus(x, y, xo=0.0, yo=0.0):
    xf = np.tan((x - xo)*2.0)
    yf = np.tan((y - yo)*2.0)
    xf -= xf.min()
    xf /= xf.max()
    yf -= yf.min()
    yf /= yf.max()
    return xf, yf


def readCellFile(path, j_offset=0, i_offset=0, transpose=False,
                 flip_j=False, flip_i=False):
    '''
    Reads in a cell.inp file and adjusts the positions of the cells.

    Parameters
    ----------
    path : string
        Path and filename of the "cell.inp" file
    j_offset : optional int (default = 0)
        Number of cells to add to the existing j (vertical) indices
    i_offset : optional int (default = 0)
        Number of cells to add to the existing i (horizontal) indices
    transpose : optional bool (default = False)
        Toggles if the whole grid needs to be transposed (i.e. swap i/j
        indices)
    flip_j : optional bool (default = False)
        The j-index should increase when moving from downstream to upstream.
        Set to True if the grid is spit out the other way.
    flip_i : optional bool (default = False)
        The i-index should increase when moving from west to east. Set to
        True if the grid is spit out the other way.

    Returns
    -------
    pandas.DataFrame if of the cells as strings in a single column.

    Notes
    -----
    [j|i]_offsets are applied /before/ data are transposed (assuming
        `transpose` = True)

    '''
    # read the file, skipping the headers
    with open(os.path.join(path, 'cell.inp'), 'r') as cellfile:
        data = []
        index = []
        for line in cellfile:
            if not line.startswith('C'):

                # pull out the row number
                idx = int(re.search('^\s*\d+', line).group(0)) + j_offset

                # pull out the cell values
                cells = re.search('\d+$', line).group(0)

                # keep everything in lists
                data.append(map(int, list(cells))) # <- this might not be necessary
                index.append(idx)

    # throw everthing into a dataframe
    startcol = 1 + i_offset
    cols = range(startcol, len(cells)+startcol)

    # maybe flip if the columns
    if flip_i:
        cols = cols[::-1]

    # maybe flip the rows
    if flip_j:
        index = index[::-1]

    df = pandas.DataFrame(data, index=index, columns=cols)
    if transpose:
        df = df.T.sort(axis=1, ascending=True).sort(axis=0, ascending=False)

    # I think this should be put off until after stitching everything together
    # then we can make crazy dataframe, fillna(0) and dump it all
    #rows = df.apply(lambda row: ''.join([str(rv) for rv in row.values]), axis=1)

    # get the horizonal cell names in the column header
    #rows.name = ''.join([str(col) for col in df.columns])

    return df #pandas.DataFrame(rows)


def mergeCellFiles(outputfile, *cellparams):
    '''
    Recursively merge cell.inp files into one.

    Parameters
    ----------
    outputfile : string
        Path and file name of the final cell.inp file
    *cellparams : dicts
        Dictionaries contains parameters to be passed to `readCellFile` for
        each individual subgrid

    Returns
    -------
    newcells : pandas.DataFrame
        DataFrame of the stitched cell data

    Notes
    -----
    - creates a new cell.inp file at the path/filename spec'd by `outputfile`
    - since this operates recursively on a arbitrary number of grids, the
      in which the grids are fed to the function matter. For example:

      >>> mergeCellFiles('mergedcell.inp', cp1, cp2, cp3, cp4)

      will merge cp1 and cp2 into cp12, which will then be merged with cp3
      into cp123 ... and so on. So basically, think hard about the order
      grids are fed into this function.

    '''

    # read in the last then next-to-last grid files
    path0 = cellparams[0].pop('path')
    rows0 = readCellFile(path0, **cellparams[-1])

    path1 = cellparams[1].pop('path')
    rows1 = readCellFile(path1, **cellparams[-2])

    newcells = pandas.concat([rows1.iloc[:-1], rows0.iloc[2:]])

    if len(cellparams) > 3:
        oldcells = cells[2:]
        oldcells.append(newcells)
        mergeCells(*oldcells)
    else:
        newcells['C'] = newcells.apply(lambda r: '{0: >3d}'.format(r.name), axis=1)
        newcells.set_index('C', inplace=True)
        newcells.columns.names = ['C']

        with open(outputfile, 'w') as fout:
            fout.write('C\nC\n')
            fout.writelines(newcells.to_string().replace('"', ''))

    return newcells


def readGridextFile(path, j_offset=0, i_offset=0, transpose=False,
                    flip_i=False, flip_j=False):
    '''
    Reads in a gridext.inp file and adjusts the index of the cells.

    Parameters
    ----------
    path : string
        Path and filename of the "cell.inp" file
    j_offset : optional int (default = 0)
        Number of cells to add to the existing j (vertical) indices
    i_offset : optional int (default = 0)
        Number of cells to add to the existing i (horizontal) indices
    transpose : optional bool (default = False)
        Toggles if the whole grid needs to be transposed (i.e. swap i/j
        indices)

    Returns
    -------
    pandas.DataFrame if of the i/j indices and x/y coords.

    '''

    if transpose:
        cols  = {0:'j', 1:'i', 2: 'x', 3: 'y'}
    else:
        cols  = {0:'i', 1:'j', 2: 'x', 3: 'y'}

    filename = os.path.join(path, 'gridext.inp')
    grid = pandas.read_table(filename, sep='\s+', header=None).rename(columns=cols)
    grid['i'] += i_offset
    grid['j'] += j_offset

    if flip_i:
        grid['i'] = grid['i'].max() + 1 - grid['i']

    if flip_j:
        grid['j'] = grid['j'].max() + 1 - grid['j']

    return grid.sort(columns=['j'], ascending=False).sort(columns=['i'])


def mergeGridextFiles(outputfile, *gridparams):
    '''
    Recursively merge gridext.inp files into one.

    Parameters
    ----------
    outputfile : string
        Path and file name of the final grid.inp file
    *gridparams : dicts
        Dictionaries contains parameters to be passed to `readGridFile` for
        each individual subgrid

    Returns
    -------
    newgrids : pandas.DataFrame
        DataFrame of the stitched grid data

    Notes
    -----
    - creates a new grid.inp file at the path/filename spec'd by `outputfile`
    - since this operates recursively on a arbitrary number of grids, the
      in which the grids are fed to the function matter. For example:

      >>> mergeGridFiles('mergedgridext.inp', gp1, gp2, gp3, gp4)

      will merge gp1 and gp2 into gp12, which will then be merged with gp3
      into gp123 ... and so on. So basically, think hard about the order
      grids are fed into this function.

    '''

    df0 = readGridextFile(
        gridparams[-1]['path'],
        transpose=cellsparams[-1].get('transpose', False),
        j_offset=cellsparams[-2].get('j_offset', 0),
        i_offset=cellsparams[-1].get('i_offset', 0)
    )
    df1 = readGridextFile(
        gridparams[-2]['path'],
        transpose=cellsparams[-2].get('transpose', False),
        j_offset=cellsparams[-2].get('j_offset', df0['j'].max()),
        i_offset=cellsparams[-2].get('i_offset', 0)

    )

    df = pandas.concat([df1, df0])
    df.to_csv(outputfile, sep=' ', float_format='%.3f',
              header=False, index=False)

    return df
