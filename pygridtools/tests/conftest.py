import numpy
from numpy import nan
import pandas

import pytest


class FakeGrid(object):
    def __init__(self):
        self.x, self.y = simple_nodes()
        self.xn, self.yn = simple_nodes()
        boundary = simple_boundary()
        self.xbry = boundary['x']
        self.ybry = boundary['y']
        self.beta = boundary['beta']
        self.ny, self.nx = self.x.shape
        self.x_rho, self.y_rho = simple_cells()
        self.cell_mask = self.x_rho.mask.copy()


@pytest.fixture(scope='module')
def fakegrid():
    return FakeGrid()


@pytest.fixture(scope='module')
def simple_boundary():
    xbry = numpy.array([1, 2, 2,  2, 3, 4, 4, 3,  2, 2, 1, 1, 1])
    ybry = numpy.array([4, 4, 3,  2, 2, 2, 1, 1,  1, 0, 0, 1, 4])
    beta = numpy.array([1, 1, 0, -1, 0, 1, 1, 0, -1, 1, 1, 0, 0])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'beta': beta, 'reach': 'reach'})


@pytest.fixture(scope='module')
def simple_islands():
    xbry = numpy.array([1.2, 1.7, 1.7, 1.2, 1.7, 3.2, 3.2, 1.7])
    ybry = numpy.array([3.7, 3.7, 2.2, 2.2, 1.7, 1.7, 1.2, 1.2])
    island = numpy.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'island': island})


@pytest.fixture(scope='module')
def simple_grid():
    '''
    Makes a basic grid for testing purposes
    '''
    try:
        import pygridgen
        boundary = simple_boundary()
        numpy.random.seed(0)
        ny = 9
        nx = 7
        ul_idx = 0
        grid = pygridgen.Gridgen(boundary.x, boundary.y, boundary.beta,
                                 (ny, nx), ul_idx=ul_idx)
    except ImportError:
        grid = fakegrid()

    return grid


@pytest.fixture(scope='module')
def simple_bathy():
    xb = numpy.arange(0, 5, 0.1)
    yb = numpy.arange(0, 5, 0.1)

    XX, YY = numpy.meshgrid(xb, yb)
    ZZ = 100 + 0.1 * (XX - XX.min()) + 0.1 * (YY - YY.min())

    bathy = pandas.DataFrame({
        'x': XX.flatten(),
        'y': YY.flatten(),
        'z': ZZ.flatten()
    })

    return bathy


@pytest.fixture(scope='module')
def simple_nodes():
    x = numpy.array([
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

    y = numpy.array([
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

    return numpy.ma.masked_invalid(x), numpy.ma.masked_invalid(y)


@pytest.fixture(scope='module')
def simple_cells():
    x = numpy.array([
        [1.25, 1.75,  nan,  nan,  nan,  nan],
        [1.25, 1.75,  nan,  nan,  nan,  nan],
        [1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
        [1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
        [1.25, 1.75,  nan,  nan,  nan,  nan],
        [1.25, 1.75,  nan,  nan,  nan,  nan],
        [1.25, 1.75,  nan,  nan,  nan,  nan],
        [1.25, 1.75,  nan,  nan,  nan,  nan],
    ])

    y = numpy.array([
        [0.25, 0.25,  nan,  nan,  nan,  nan],
        [0.75, 0.75,  nan,  nan,  nan,  nan],
        [1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
        [1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
        [2.25, 2.25,  nan,  nan,  nan,  nan],
        [2.75, 2.75,  nan,  nan,  nan,  nan],
        [3.25, 3.25,  nan,  nan,  nan,  nan],
        [3.75, 3.75,  nan,  nan,  nan,  nan],
    ])

    return numpy.ma.masked_invalid(x), numpy.ma.masked_invalid(y)
