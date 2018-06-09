import os
import warnings
from pkg_resources import resource_filename
import tempfile

import numpy
from numpy import nan
import pandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import core
from . import utils

BASELINE_IMAGES = 'baseline_files/test_core'
try:
    import pygridgen
    HASPGG = True
except ImportError:
    HASPGG = False


@pytest.fixture
def A():
    return numpy.arange(12).reshape(4, 3).astype(float)


@pytest.fixture
def B():
    return numpy.arange(8).reshape(2, 4).astype(float)


@pytest.fixture
def C():
    return numpy.arange(25).reshape(5, 5).astype(float)


@pytest.mark.parametrize('fxn', [numpy.fliplr, numpy.flipud, numpy.fliplr])
def test_transform(A, fxn):
    result = core.transform(A, fxn)
    expected = fxn(A)
    nptest.assert_array_equal(result, expected)


@pytest.mark.parametrize(('index', 'axis', 'first', 'second'), [
    (3, 0, 'top', 'bottom'),
    (2, 1, 'left', 'right'),
    (5, 0, None, None),
    (5, 1, None, None),
])
def test_split_rows(C, index, axis, first, second):
    expected = {
        'top': numpy.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ]),
        'bottom': numpy.array([
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., 24.],
        ]),
        'left': numpy.array([
            [ 0.,  1.],
            [ 5.,  6.],
            [10., 11.],
            [15., 16.],
            [20., 21.],
        ]),
        'right': numpy.array([
            [ 2.,  3.,  4.],
            [ 7.,  8.,  9.],
            [12., 13., 14.],
            [17., 18., 19.],
            [22., 23., 24.],
        ]),
    }
    if first and second:
        a, b = core.split(C, index, axis)
        nptest.assert_array_equal(a, expected[first])
        nptest.assert_array_equal(b, expected[second])
    else:
        with utils.raises(ValueError):
            left, right = core.split(C, index, axis=axis)


@pytest.mark.parametrize('N', [1, 3, None])
def test__interp_between_vectors(N):
    index = numpy.arange(0, 4)
    vector1 = -1 * index**2 - 1
    vector2 = 2 * index**2 + 2

    expected = {
        1: numpy.array([
            [ -1.0,  -2.0,  -5.0, -10.0],
            [  0.5,   1.0,   2.5,   5.0],
            [  2.0,   4.0,  10.0,  20.0],
        ]),
        3: numpy.array([
            [ -1.00,  -2.00,  -5.00, -10.00],
            [ -0.25,  -0.50,  -1.25,  -2.50],
            [  0.50,   1.00,   2.50,   5.00],
            [  1.25,   2.50,   6.25,  12.50],
            [  2.00,   4.00,  10.00,  20.00],
        ])
    }

    if N:
        result = core._interp_between_vectors(vector1, vector2, n_nodes=N)
        nptest.assert_array_equal(result, expected[N])
    else:
        with utils.raises(ValueError):
            core._interp_between_vectors(vector1, vector2, n_nodes=0)


@pytest.mark.parametrize(('n', 'axis'), [
    (1, 0), (4, 0), (1, 1), (3, 1)
])
def test_insert(C, n, axis):
    expected = {
        (1, 0): numpy.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [ 7.5,  8.5,  9.5, 10.5, 11.5],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]),
        (4, 0): numpy.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [ 6.0,  7.0,  8.0,  9.0, 10.0],
            [ 7.0,  8.0,  9.0, 10.0, 11.0],
            [ 8.0,  9.0, 10.0, 11.0, 12.0],
            [ 9.0, 10.0, 11.0, 12.0, 13.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]),
        (1, 1): numpy.array([
            [ 0.0,  1.0,  1.5,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  6.5,  7.0,  8.0,  9.0],
            [10.0, 11.0, 11.5, 12.0, 13.0, 14.0],
            [15.0, 16.0, 16.5, 17.0, 18.0, 19.0],
            [20.0, 21.0, 21.5, 22.0, 23.0, 24.0],
        ]),
        (3, 1): numpy.array([
            [ 0.00,  1.00,  1.25,  1.50,  1.75,  2.00,  3.00,  4.00],
            [ 5.00,  6.00,  6.25,  6.50,  6.75,  7.00,  8.00,  9.00],
            [10.00, 11.00, 11.25, 11.50, 11.75, 12.00, 13.00, 14.00],
            [15.00, 16.00, 16.25, 16.50, 16.75, 17.00, 18.00, 19.00],
            [20.00, 21.00, 21.25, 21.50, 21.75, 22.00, 23.00, 24.00],
        ])
    }
    result = core.insert(C, 2, axis=axis, n_nodes=n)
    nptest.assert_array_equal(result, expected[(n, axis)])


@pytest.mark.parametrize('how', ['h', 'v'])
@pytest.mark.parametrize('where', ['+', '-'])
@pytest.mark.parametrize('shift', [0, 2, -1])
def test_merge(A, B, how, where, shift):
    expected = {
        ('v', '+', 0): numpy.array([
            [0.,  1.,  2., nan],
            [3.,  4.,  5., nan],
            [6.,  7.,  8., nan],
            [9., 10., 11., nan],
            [0.,  1.,  2.,  3.],
            [4.,  5.,  6.,  7.]
        ]),
        ('v', '-', 0): numpy.array([
            [0.,  1.,  2.,  3.],
            [4.,  5.,  6.,  7.],
            [0.,  1.,  2., nan],
            [3.,  4.,  5., nan],
            [6.,  7.,  8., nan],
            [9., 10., 11., nan]
        ]),
        ('v', '+', 2): numpy.array([
            [ 0.,   1.,   2., nan,   nan, nan],
            [ 3.,   4.,   5., nan,   nan, nan],
            [ 6.,   7.,   8., nan,   nan, nan],
            [ 9.,  10.,  11., nan,   nan, nan],
            [nan,  nan,   0.,   1.,   2.,  3.],
            [nan,  nan,   4.,   5.,   6.,  7.]
        ]),
        ('v', '-', 2): numpy.array([
            [nan, nan,  0.,  1.,  2.,  3.],
            [nan, nan,  4.,  5.,  6.,  7.],
            [ 0.,  1.,  2., nan, nan, nan],
            [ 3.,  4.,  5., nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan]
        ]),
        ('v', '+', -1): numpy.array([
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.],
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.]
        ]),
        ('v', '-', -1): numpy.array([
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.],
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.]
        ]),
        ('h', '+', 0): numpy.array([
            [0.,  1.,  2.,  0.,  1.,  2.,  3.],
            [3.,  4.,  5.,  4.,  5.,  6.,  7.],
            [6.,  7.,  8., nan, nan, nan, nan],
            [9., 10., 11., nan, nan, nan, nan]
        ]),
        ('h', '-', 0): numpy.array([
            [ 0.,  1.,  2.,  3., 0.,  1.,  2.],
            [ 4.,  5.,  6.,  7., 3.,  4.,  5.],
            [nan, nan, nan, nan, 6.,  7.,  8.],
            [nan, nan, nan, nan, 9., 10., 11.]
        ]),
        ('h', '+', 2): numpy.array([
            [0.,  1.,  2., nan, nan, nan, nan],
            [3.,  4.,  5., nan, nan, nan, nan],
            [6.,  7.,  8.,  0.,  1.,  2.,  3.],
            [9., 10., 11.,  4.,  5.,  6.,  7.]
        ]),
        ('h', '-', 2): numpy.array([
            [nan, nan, nan, nan, 0.,  1.,  2.],
            [nan, nan, nan, nan, 3.,  4.,  5.],
            [ 0.,  1.,  2.,  3., 6.,  7.,  8.],
            [ 4.,  5.,  6.,  7., 9., 10., 11.]
        ]),
        ('h', '+', -1): numpy.array([
            [nan, nan, nan,  0.,  1.,  2.,  3.],
            [ 0.,  1.,  2.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5., nan, nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan, nan]
        ]),
        ('h', '-', -1): numpy.array([
            [ 0.,  1.,  2.,  3., nan, nan, nan],
            [ 4.,  5.,  6.,  7.,  0.,  1.,  2.],
            [nan, nan, nan, nan,  3.,  4.,  5.],
            [nan, nan, nan, nan,  6.,  7.,  8.],
            [nan, nan, nan, nan,  9., 10., 11.]
        ]),
    }
    result = core.merge(A, B, how=how, where=where, shift=shift)
    nptest.assert_array_equal(result, expected[(how, where, shift)])


@pytest.fixture
def g1(simple_nodes):
    xn, yn = simple_nodes
    g = core.ModelGrid(xn[:, :3], yn[:, :3])

    mask = g.cell_mask
    mask[:2, :2] = True
    g.cell_mask = mask
    return g


@pytest.fixture
def g2(simple_nodes):
    xn, yn = simple_nodes
    g = core.ModelGrid(xn[2:5, 3:], yn[2:5, 3:])
    return g


@pytest.fixture
def polyverts():
    return [(2.4, 0.9), (3.6, 0.9), (3.6, 2.4), (2.4, 2.4)]


def test_ModelGrid_bad_shapes(simple_cells):
    xc, yc = simple_cells
    with utils.raises(ValueError):
        mg = core.ModelGrid(xc, yc[2:, 2:])


def test_ModelGrid_nodes_and_cells(g1, simple_cells):
    xc, yc = simple_cells
    assert (isinstance(g1.nodes_x, numpy.ndarray))
    assert (isinstance(g1.nodes_y, numpy.ndarray))
    assert (isinstance(g1.cells_x, numpy.ndarray))
    nptest.assert_array_equal(g1.cells_x, xc[:, :2])
    assert (isinstance(g1.cells_y, numpy.ndarray))
    nptest.assert_array_equal(g1.cells_y, yc[:, :2])


def test_ModelGrid_counts_and_shapes(g1):
    expected_rows = 9
    expected_cols = 3

    assert (g1.icells == expected_cols - 1)
    assert (g1.jcells == expected_rows - 1)

    assert (g1.inodes == expected_cols)
    assert (g1.jnodes == expected_rows)

    assert (g1.shape == (expected_rows, expected_cols))
    assert (g1.cell_shape == (expected_rows - 1, expected_cols - 1))


def test_ModelGrid_cell_mask(g1):
    expected_mask = numpy.array([
        [1, 1], [1, 1], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0], [0, 0],
    ])
    nptest.assert_array_equal(g1.cell_mask, expected_mask)


@pytest.mark.parametrize(('usemask', 'which', 'error'), [
    (True, 'nodes', ValueError),
    (False, 'nodes', None),
    (True, 'cells', None),
])
def test_ModelGrid_to_dataframe(g1, usemask, which, error):
    def name_cols(df):
        df.columns.names = ['coord', 'ii']
        df.index.names = ['jj']
        return df

    if error:
        with utils.raises(ValueError):
            g1.to_dataframe(usemask=usemask, which=which)
    else:

        expected = {
            (False, 'nodes'): pandas.DataFrame({
                ('easting', 0): {
                    0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                    5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0
                }, ('easting', 1): {
                    0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5,
                    5: 1.5, 6: 1.5, 7: 1.5, 8: 1.5
                }, ('easting', 2): {
                    0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0,
                    5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0
                }, ('northing', 0): {
                    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                    5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0
                }, ('northing', 1): {
                    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                    5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0
                }, ('northing', 2): {
                    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0,
                    5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
            }).pipe(name_cols),
            (True, 'cells'): pandas.DataFrame({
                ('easting', 0): {
                    0: nan, 1: nan, 2: 1.25, 3: 1.25, 4: 1.25,
                    5: 1.25, 6: 1.25, 7: 1.25
                }, ('easting', 1): {
                    0: nan, 1: nan, 2: 1.75, 3: 1.75, 4: 1.75,
                    5: 1.75, 6: 1.75, 7: 1.75
                }, ('northing', 0): {
                    0: nan, 1: nan, 2: 1.25, 3: 1.75, 4: 2.25,
                    5: 2.75, 6: 3.25, 7: 3.75
                }, ('northing', 1): {
                    0: nan, 1: nan, 2: 1.25, 3: 1.75, 4: 2.25,
                    5: 2.75, 6: 3.25, 7: 3.75
                }
            }).pipe(name_cols),
        }

        result = g1.to_dataframe(usemask=usemask, which=which)
        pdtest.assert_frame_equal(result, expected[(usemask, which)], check_names=False)
        pdtest.assert_index_equal(result.columns, expected[(usemask, which)].columns)


@pytest.mark.parametrize(('usemask', 'which', 'error'), [
    (True, 'nodes', ValueError),
    (False, 'nodes', None),
    (True, 'cells', None),
    (False, 'cells', None),
])
def test_ModelGrid_to_coord_pairs(g1, usemask, which, error):
    if error:
        with utils.raises(error):
            g1.to_coord_pairs(usemask=usemask, which=which)
    else:

        expected = {
            ('nodes', False): numpy.array([
                [1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [1.0, 0.5],
                [1.5, 0.5], [2.0, 0.5], [1.0, 1.0], [1.5, 1.0],
                [2.0, 1.0], [1.0, 1.5], [1.5, 1.5], [2.0, 1.5],
                [1.0, 2.0], [1.5, 2.0], [2.0, 2.0], [1.0, 2.5],
                [1.5, 2.5], [2.0, 2.5], [1.0, 3.0], [1.5, 3.0],
                [2.0, 3.0], [1.0, 3.5], [1.5, 3.5], [2.0, 3.5],
                [1.0, 4.0], [1.5, 4.0], [2.0, 4.0]
            ]),
            ('cells', False): numpy.array([
                [1.25, 0.25], [1.75, 0.25], [1.25, 0.75], [1.75, 0.75],
                [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
                [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
                [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
            ]),
            ('cells', True): numpy.array([
                [nan, nan], [nan, nan], [nan, nan], [nan, nan],
                [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
                [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
                [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
            ])
        }

        result = g1.to_coord_pairs(usemask=usemask, which=which)
        nptest.assert_array_equal(result, expected[which, usemask])


def test_ModelGrid_transform(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.transform(lambda x: x * 10)
    nptest.assert_array_equal(g.xn, xn * 10)
    nptest.assert_array_equal(g.yn, yn * 10)


def test_ModelGrid_transform_x(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.transform_x(lambda x: x * 10)
    nptest.assert_array_equal(g.xn, xn * 10)
    nptest.assert_array_equal(g.yn, yn)


def test_ModelGrid_transform_y(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.transform_y(lambda y: y * 10)
    nptest.assert_array_equal(g.xn, xn)
    nptest.assert_array_equal(g.yn, yn * 10)


def test_ModelGrid_transpose(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.transpose()
    nptest.assert_array_equal(g.xn, xn.T)
    nptest.assert_array_equal(g.yn, yn.T)


def test_ModelGrid_fliplr(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.fliplr()
    nptest.assert_array_equal(g.xn, numpy.fliplr(xn))
    nptest.assert_array_equal(g.yn, numpy.fliplr(yn))


def test_ModelGrid_flipud(mg, simple_nodes):
    xn, yn = simple_nodes
    g = mg.flipud()
    nptest.assert_array_equal(g.xn, numpy.flipud(xn))
    nptest.assert_array_equal(g.yn, numpy.flipud(yn))


def test_ModelGrid_split_ax0(mg, simple_nodes):
    xn, yn = simple_nodes
    mgtop, mgbottom = mg.split(3, axis=0)
    nptest.assert_array_equal(mgtop.nodes_x, xn[:3, :])
    nptest.assert_array_equal(mgtop.nodes_y, yn[:3, :])
    nptest.assert_array_equal(mgbottom.nodes_x, xn[3:, :])
    nptest.assert_array_equal(mgbottom.nodes_y, yn[3:, :])


def test_ModelGrid_merge(g1, g2, simple_nodes):
    g3 = g1.merge(g2, how='horiz', where='+', shift=2)
    g4 = core.ModelGrid(*simple_nodes)

    nptest.assert_array_equal(g3.xn, g4.xn)
    nptest.assert_array_equal(g3.xc, g4.xc)


def test_ModelGrid_insert_3_ax0(mg):
    known_xnodes = numpy.ma.masked_invalid(numpy.array([
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
        [1.0, 1.5, 2.0, nan, nan, nan, nan],
    ]))

    known_ynodes = numpy.ma.masked_invalid(numpy.array([
        [0.000, 0.000, 0.000,   nan,   nan,   nan,   nan],
        [0.500, 0.500, 0.500,   nan,   nan,   nan,   nan],
        [0.625, 0.625, 0.625,   nan,   nan,   nan,   nan],
        [0.750, 0.750, 0.750,   nan,   nan,   nan,   nan],
        [0.875, 0.875, 0.875,   nan,   nan,   nan,   nan],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500],
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [2.500, 2.500, 2.500,   nan,   nan,   nan,   nan],
        [3.000, 3.000, 3.000,   nan,   nan,   nan,   nan],
        [3.500, 3.500, 3.500,   nan,   nan,   nan,   nan],
        [4.000, 4.000, 4.000,   nan,   nan,   nan,   nan],
    ]))

    result = mg.insert(2, axis=0, n_nodes=3)
    nptest.assert_array_equal(result.nodes_x, known_xnodes)
    nptest.assert_array_equal(result.nodes_y, known_ynodes)


def test_ModelGrid_insert_3_ax1(mg):
    known_xnodes = numpy.ma.masked_invalid(numpy.array([
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000, 2.500, 3.000, 3.500, 4.000],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000, 2.500, 3.000, 3.500, 4.000],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000, 2.500, 3.000, 3.500, 4.000],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan],
        [1.000, 1.500, 1.625, 1.750, 1.875, 2.000,   nan,   nan,   nan,   nan]
    ]))

    known_ynodes = numpy.ma.masked_invalid(numpy.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, nan, nan],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, nan, nan, nan, nan],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, nan, nan, nan, nan],
        [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, nan, nan, nan, nan],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan],
    ]))

    result = mg.insert(2, axis=1, n_nodes=3)
    nptest.assert_array_equal(result.nodes_x, known_xnodes)
    nptest.assert_array_equal(result.nodes_y, known_ynodes)


def test_extract(mg, simple_nodes):
    xn, yn = simple_nodes
    result = mg.extract(jstart=2, jend=5, istart=3, iend=6)
    nptest.assert_array_equal(result.nodes_x, xn[2:5, 3:6])
    nptest.assert_array_equal(result.nodes_y, yn[2:5, 3:6])


@pytest.mark.parametrize(('inside', 'use_existing'), [
    (True, False),
    (True, True),
    (False, False)
])
def test_ModelGrid_mask_centroids(mg, polyverts, inside, use_existing):
    expected = {
        (True, False): numpy.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]),
        (True, True): numpy.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1]
        ]),
        (False, False):  numpy.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ])
    }

    result = mg.mask_centroids(polyverts, inside=inside, use_existing=use_existing)

    nptest.assert_array_equal(
        result.cell_mask.astype(int),
        expected[(inside, use_existing)].astype(int)
    )


@pytest.mark.parametrize(('kwargs', 'error'), [
    [dict(min_nodes=0), ValueError],
    [dict(min_nodes=5), ValueError],
    [dict(triangles=True), NotImplementedError],
])
def test_ModelGrid_mask_nodes_errors(mg, polyverts, kwargs, error):
    with utils.raises(error):
        mg.mask_nodes(polyverts, **kwargs)


@pytest.mark.parametrize(['geom', 'expectedfile'], [
    ('point', 'mgshp_nomask_nodes_points.shp'),
    ('polygon', 'mgshp_nomask_cells_polys.shp'),
    ('line', None),
])
def test_ModelGrid_to_shapefile_nodes(g1, geom, expectedfile):
    with tempfile.TemporaryDirectory() as outdir:
        outfile = os.path.join(outdir, 'outfile.shp')
        if expectedfile is None:
            with utils.raises(ValueError):
                g1.to_shapefile(outfile, which='nodes', geom=geom, usemask=False)
        else:
                resultfile = resource_filename('pygridtools.tests.baseline_files', expectedfile)
                g1.to_shapefile(outfile, which='nodes', geom=geom, usemask=False)
                utils.compareShapefiles(outfile, resultfile)


@pytest.mark.parametrize('usemask', [True, False])
@pytest.mark.parametrize('geom', ['point', 'polygon'])
def test_ModelGrid_to_shapefile_cells(g1, geom, usemask):
    expectedfile = {
        (True, 'point'): 'mgshp_mask_cells_points.shp',
        (True, 'polygon'): 'mgshp_mask_cells_polys.shp',
        (False, 'point'): 'mgshp_nomask_cells_points.shp',
        (False, 'polygon'): 'mgshp_nomask_cells_polys.shp',
    }
    with tempfile.TemporaryDirectory() as outdir:
        outfile = os.path.join(outdir, 'outfile.shp')
        expected = resource_filename('pygridtools.tests.baseline_files',
                                     expectedfile[usemask, geom])
        g1.to_shapefile(outfile, which='cells', geom=geom, usemask=usemask)
        utils.compareShapefiles(outfile, expected)


@pytest.mark.parametrize(('which', 'usemask', 'error'), [
    ('nodes', True, ValueError),
    ('junk', False, ValueError),
    ('nodes', False, None),
    ('cells', False, None),
])
def test_ModelGrid__get_x_y_nodes_and_mask(g1, which, usemask, error):
    if error:
        with utils.raises(error):
            g1._get_x_y(which, usemask=usemask)
    else:
        x, y = g1._get_x_y(which, usemask=usemask)
        nptest.assert_array_equal(x, getattr(g1, 'x' + which[0]))
        nptest.assert_array_equal(y, getattr(g1, 'y' + which[0]))


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_ModelGrid_plots_basic(simple_nodes):
    mg = core.ModelGrid(*simple_nodes)
    mg.cell_mask = numpy.ma.masked_invalid(mg.xc).mask
    fig, artists = mg.plotCells()
    return fig


@pytest.mark.parametrize(('otherargs', 'gridtype'), [
    (dict(), None),
    (dict(verbose=True), None),
    (dict(rawgrid=False), core.ModelGrid)
])
@pytest.mark.skipif(not HASPGG, reason='pygridgen unavailabile')
def test_makeGrid(simple_boundary, simple_bathy, otherargs, gridtype):
    if not gridtype:
        gridtype = pygridgen.Gridgen

    gridparams = {'nnodes': 12, 'verbose': False, 'ul_idx': 0}
    gridparams.update(otherargs)
    grid = core.makeGrid(
        9, 7,
        domain=simple_boundary,
        bathydata=simple_bathy.dropna(),
        **gridparams
    )
    assert (isinstance(grid, gridtype))
