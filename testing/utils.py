import numpy as np
import matplotlib; matplotlib.use('agg')
import pandas
import fiona
import pygridgen

#from pygridtools.misc import Grid

import nose.tools as nt
import numpy.testing as nptest


def makeSimpleBoundary():
    xbry = np.array([1, 2, 2, 2, 3, 4, 4, 3, 2, 2, 1, 1, 1])
    ybry = np.array([4, 4, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 4])
    beta = np.array([1, 1, 0,-1, 0, 1, 1, 0,-1, 1, 1, 0, 0])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'beta': beta, 'reach': 'reach'})


def makeSimpleGrid():
    '''
    Makes a basic grid for testing purposes
    '''
    boundary = makeSimpleBoundary()
    np.random.seed(0)
    ny = 9
    nx = 7
    ul_idx = 0
    grid = pygridgen.Gridgen(boundary.x, boundary.y, boundary.beta,
                            (ny, nx), ul_idx=ul_idx)

    return grid


def makeSimpleBathy():
    xb = np.arange(0, 5, 0.1)
    yb = np.arange(0, 5, 0.1)

    XX, YY = np.meshgrid(xb, yb)
    ZZ = 100 + 0.1 * (XX - XX.min()) + 0.1 * (YY - YY.min())

    bathy = pandas.DataFrame({
        'x': XX.flatten(),
        'y': YY.flatten(),
        'z': ZZ.flatten()
    })

    return bathy


def compareTextFiles(baselinefile, outputfile):
    with open(outputfile) as output:
        results = output.read()

    with open(baselinefile) as baseline:
        expected = baseline.read()

    nt.assert_equal(results, expected)


def compareShapefiles(baselinefile, outputfile, atol=0.001):
    base_records = []
    result_records = []
    with fiona.open(outputfile, 'r') as result:
        result_records = list(result)

    with fiona.open(baselinefile, 'r') as baseline:
        base_records = list(baseline)

    for rr, br in zip(result_records, base_records):
        nt.assert_dict_equal(rr['properties'], br['properties'])
        nt.assert_equal(rr['geometry']['type'], br['geometry']['type'])
        nptest.assert_allclose(
            rr['geometry']['coordinates'],
            br['geometry']['coordinates'],
            atol=atol
        )


def make_nodes():
    from numpy import nan
    x = np.array([
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
    ])

    y = np.array([
        [0.0, 0.0, 0.0, nan, nan, nan, nan],
        [0.5, 0.5, 0.5, nan, nan, nan, nan],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [2.5, 2.5, 2.5, nan, nan, nan, nan],
        [3.0, 3.0, 3.0, nan, nan, nan, nan],
        [3.5, 3.5, 3.5, nan, nan, nan, nan],
        [4.0, 4.0, 4.0, nan, nan, nan, nan],
    ])

    return np.ma.masked_invalid(x, 0), np.ma.masked_invalid(y, 0)
