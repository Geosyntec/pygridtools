import numpy as np
import pandas
import fiona
import pygridgen

import nose.tools as nt


def makeSimpleBoundary():
    xbry = np.array([1, 2, 2, 2, 3, 4, 4, 3, 2, 2, 1, 1, 1])
    ybry = np.array([4, 4, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 4])
    beta = np.array([1, 1, 0,-1, 0, 1, 1, 0,-1, 1, 1, 0, 0])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'beta': beta})


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
    grid.elev = np.random.normal(size=grid.x_rho.shape)

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


def compareShapefiles(baselinefile, outputfile):
    base_records = []
    result_records = []
    with fiona.open(outputfile, 'r') as result:
        result_records = list(result)

    with fiona.open(baselinefile, 'r') as baseline:
        base_records = list(baseline)

    for rr, br in zip(result_records, base_records):
        nt.assert_dict_equal(rr, br)
