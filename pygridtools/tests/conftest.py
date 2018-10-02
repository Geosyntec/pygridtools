import numpy
from numpy import nan
import pandas

import pytest

from pygridtools import ModelGrid


class FakeGrid(object):
    def __init__(self, boundary):
        self.x, self.y = simple_nodes()
        self.xn, self.yn = simple_nodes()
        self.xbry = boundary['x']
        self.ybry = boundary['y']
        self.beta = boundary['beta']
        self.ny, self.nx = self.x.shape
        self.x_rho, self.y_rho = simple_cells()
        self.cell_mask = self.x_rho.mask.copy()


@pytest.fixture(scope='module')
def simple_boundary():
    xbry = numpy.array([1, 2, 2,  2, 3, 4, 4, 3,  2, 2, 1, 1, 1])
    ybry = numpy.array([4, 4, 3,  2, 2, 2, 1, 1,  1, 0, 0, 1, 4])
    beta = numpy.array([1, 1, 0, -1, 0, 1, 1, 0, -1, 1, 1, 0, 0])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'beta': beta, 'reach': 'reach'})


@pytest.fixture(scope='module')
def fakegrid():
    return FakeGrid(simple_boundary)


@pytest.fixture(scope='module')
def simple_islands():
    xbry = numpy.array([1.2, 1.7, 1.7, 1.2, 1.7, 3.2, 3.2, 1.7])
    ybry = numpy.array([3.7, 3.7, 2.2, 2.2, 1.7, 1.7, 1.2, 1.2])
    island = numpy.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    return pandas.DataFrame({'x': xbry, 'y': ybry, 'island': island})


@pytest.fixture(scope='module')
def simple_grid(simple_boundary):
    '''
    Makes a basic grid for testing purposes
    '''
    try:
        import pygridgen
        numpy.random.seed(0)
        ny = 9
        nx = 7
        ul_idx = 0
        grid = pygridgen.Gridgen(simple_boundary.x, simple_boundary.y,
                                 simple_boundary.beta, (ny, nx), ul_idx=ul_idx)
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


@pytest.fixture(scope='module')
def mg(simple_nodes):
    xn, yn = simple_nodes
    g = ModelGrid(xn, yn)
    g.cell_mask = numpy.array([
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1]
    ], dtype=bool)
    return g


@pytest.fixture(scope='module')
def river():
    return [
        (2.5, 59),
        (2.5, 57),
        (5.5, 56),
        (5.5, 52),
        (13.0, 51),
        (13.0, 53),
        (10.0, 54),
        (10.0, 55),
        (8.5, 55),
        (8.5, 59),
        (2.5, 59),
    ]


@pytest.fixture(scope='module')
def river_grid(river):
    _x = numpy.array([[
        0., 1.9, 3.7, 5.1, 6.4, 7.4, 8.2, 8.8, 9.3, 9.7,
        10., 10.3, 10.7, 11.2, 11.8, 12.6, 13.6, 14.9, 16.3, 18.1
    ] * 20]).reshape((20, 20))

    _y = numpy.array([[
        50., 50.2, 50.4, 50.6, 50.7, 50.9, 51.2, 51.5, 51.9, 52.4,
        52.9, 53.5, 54.2, 54.9, 55.6, 56.3, 57.1, 57.8, 58.5, 59.3
    ] * 20]).reshape((20, 20)).T

    return ModelGrid(_x, _y).mask_nodes(river, min_nodes=2)


@pytest.fixture(scope='module')
def river_bathy(river_grid):
    Z = numpy.abs((river_grid.xc - 10)**2 + 4 * (river_grid.yc - 30))
    return numpy.ma.masked_array(Z, river_grid.cell_mask)


@pytest.fixture(scope='module')
def example_crs():
    return {'init': 'epsg:26916'}
