import os
import warnings

import numpy as np
from numpy import nan
import pandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest
from matplotlib.testing.decorators import image_comparison

from pygridtools import core
from pygridtools import testing

try:
    import pygridgen
    has_pgg = True
except ImportError:
    has_pgg = False


@pytest.fixture
def A():
    return np.arange(12).reshape(4, 3).astype(float)


@pytest.fixture
def B():
    return np.arange(8).reshape(2, 4).astype(float)


@pytest.fixture
def C():
    return np.arange(25).reshape(5, 5).astype(float)


@pytest.mark.parametrize('fxn', [np.fliplr, np.flipud, np.transpose])
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
        'top': np.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ]),
        'bottom': np.array([
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., 24.],
        ]),
        'left': np.array([
            [ 0.,  1.],
            [ 5.,  6.],
            [10., 11.],
            [15., 16.],
            [20., 21.],
        ]),
        'right': np.array([
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
        with pytest.raises(ValueError):
            left, right = core.split(C, index, axis=axis)


@pytest.mark.parametrize('N', [1, 3, None])
def test__interp_between_vectors(N):
    index = np.arange(0, 4)
    vector1 = -1 * index**2 - 1
    vector2 = 2 * index**2 + 2

    expected = {
        1: np.array([
            [ -1.0,  -2.0,  -5.0, -10.0],
            [  0.5,   1.0,   2.5,   5.0],
            [  2.0,   4.0,  10.0,  20.0],
        ]),
        3: np.array([
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
        with pytest.raises(ValueError):
            core._interp_between_vectors(vector1, vector2, n_nodes=0)


@pytest.mark.parametrize(('n', 'axis'), [
    (1, 0), (4, 0), (1, 1), (3, 1)
])
def test_insert(C, n, axis):
    expected = {
        (1, 0): np.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [ 7.5,  8.5,  9.5, 10.5, 11.5],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]),
        (4, 0): np.array([
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
        (1, 1): np.array([
            [ 0.0,  1.0,  1.5,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  6.5,  7.0,  8.0,  9.0],
            [10.0, 11.0, 11.5, 12.0, 13.0, 14.0],
            [15.0, 16.0, 16.5, 17.0, 18.0, 19.0],
            [20.0, 21.0, 21.5, 22.0, 23.0, 24.0],
        ]),
        (3, 1): np.array([
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
        ('v', '+', 0): np.array([
            [0.,  1.,  2., nan],
            [3.,  4.,  5., nan],
            [6.,  7.,  8., nan],
            [9., 10., 11., nan],
            [0.,  1.,  2.,  3.],
            [4.,  5.,  6.,  7.]
        ]),
        ('v', '-', 0): np.array([
            [0.,  1.,  2.,  3.],
            [4.,  5.,  6.,  7.],
            [0.,  1.,  2., nan],
            [3.,  4.,  5., nan],
            [6.,  7.,  8., nan],
            [9., 10., 11., nan]
        ]),
        ('v', '+', 2): np.array([
            [ 0.,   1.,   2., nan,   nan, nan],
            [ 3.,   4.,   5., nan,   nan, nan],
            [ 6.,   7.,   8., nan,   nan, nan],
            [ 9.,  10.,  11., nan,   nan, nan],
            [nan,  nan,   0.,   1.,   2.,  3.],
            [nan,  nan,   4.,   5.,   6.,  7.]
        ]),
        ('v', '-', 2): np.array([
            [nan, nan,  0.,  1.,  2.,  3.],
            [nan, nan,  4.,  5.,  6.,  7.],
            [ 0.,  1.,  2., nan, nan, nan],
            [ 3.,  4.,  5., nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan]
        ]),
        ('v', '+', -1): np.array([
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.],
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.]
        ]),
        ('v', '-', -1): np.array([
            [ 0., 1., 2.,   3.],
            [ 4., 5., 6.,   7.],
            [nan, 0., 1.,   2.],
            [nan, 3., 4.,   5.],
            [nan, 6., 7.,   8.],
            [nan, 9., 10., 11.]
        ]),
        ('h', '+', 0): np.array([
            [0.,  1.,  2.,  0.,  1.,  2.,  3.],
            [3.,  4.,  5.,  4.,  5.,  6.,  7.],
            [6.,  7.,  8., nan, nan, nan, nan],
            [9., 10., 11., nan, nan, nan, nan]
        ]),
        ('h', '-', 0): np.array([
            [ 0.,  1.,  2.,  3., 0.,  1.,  2.],
            [ 4.,  5.,  6.,  7., 3.,  4.,  5.],
            [nan, nan, nan, nan, 6.,  7.,  8.],
            [nan, nan, nan, nan, 9., 10., 11.]
        ]),
        ('h', '+', 2): np.array([
            [0.,  1.,  2., nan, nan, nan, nan],
            [3.,  4.,  5., nan, nan, nan, nan],
            [6.,  7.,  8.,  0.,  1.,  2.,  3.],
            [9., 10., 11.,  4.,  5.,  6.,  7.]
        ]),
        ('h', '-', 2): np.array([
            [nan, nan, nan, nan, 0.,  1.,  2.],
            [nan, nan, nan, nan, 3.,  4.,  5.],
            [ 0.,  1.,  2.,  3., 6.,  7.,  8.],
            [ 4.,  5.,  6.,  7., 9., 10., 11.]
        ]),
        ('h', '+', -1): np.array([
            [nan, nan, nan,  0.,  1.,  2.,  3.],
            [ 0.,  1.,  2.,  4.,  5.,  6.,  7.],
            [ 3.,  4.,  5., nan, nan, nan, nan],
            [ 6.,  7.,  8., nan, nan, nan, nan],
            [ 9., 10., 11., nan, nan, nan, nan]
        ]),
        ('h', '-', -1): np.array([
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
def nodes():
    return testing.makeSimpleNodes()


@pytest.fixture
def cells():
    return testing.makeSimpleCells()


@pytest.fixture
def mg(nodes):
    xn, yn = nodes
    g = core.ModelGrid(xn, yn)
    return g


@pytest.fixture
def g1(nodes):
    xn, yn = nodes
    g = core.ModelGrid(xn[:, :3], yn[:, :3])
    g.template = 'pygridtools/tests/test_data/schema_template.shp'
    return g


@pytest.fixture
def g2(nodes):
    xn, yn = nodes
    g = core.ModelGrid(xn[2:5, 3:], yn[2:5, 3:])
    g.template = 'pygridtools/tests/test_data/schema_template.shp'
    return g


@pytest.fixture
def polyverts():
    return [(2.4, 0.9), (3.6, 0.9), (3.6, 2.4), (2.4, 2.4)]


def test_ModelGrid_bad_shapes(cells):
    xc, yc = cells
    with pytest.raises(ValueError):
        mg = core.ModelGrid(xc, yc[2:, 2:])


def test_ModelGrid_nodes_and_cells(g1, cells):
    xc, yc = cells
    assert (isinstance(g1.nodes_x, np.ndarray))
    assert (isinstance(g1.nodes_y, np.ndarray))
    assert (isinstance(g1.cells_x, np.ndarray))
    nptest.assert_array_equal(g1.cells_x, xc[:, :2])
    assert (isinstance(g1.cells_y, np.ndarray))
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
    expected_mask = np.array([
        [0, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0], [0, 0],
    ])
    nptest.assert_array_equal(g1.cell_mask, expected_mask)


def test_ModelGrid_template(g1):
    assert g1.template.endswith('schema_template.shp')

    g1.template = 'junk'
    assert (g1.template == 'junk')


class Test_ModelGrid(object):
    def setup(self):
        self.xn, self.yn = testing.makeSimpleNodes()
        self.xc, self.yc = testing.makeSimpleCells()
        self.mg = core.ModelGrid(self.xn, self.yn)
        self.g1 = core.ModelGrid(self.xn[:, :3], self.yn[:, :3])
        self.g2 = core.ModelGrid(self.xn[2:5, 3:], self.yn[2:5, 3:])
        self.polyverts = [
            (2.4, 0.9),
            (3.6, 0.9),
            (3.6, 2.4),
            (2.4, 2.4),
        ]

        self.template = 'pygridtools/tests/test_data/schema_template.shp'
        self.g1.template = self.template
        self.g2.template = self.template

        self.known_rows = 9
        self.expected_cols = 3
        self.known_df = pandas.DataFrame({
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
        })
        self.known_df.columns.names = ['coord', 'i']
        self.known_df.index.names = ['j']

        self.known_masked_cell_df = pandas.DataFrame({
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
        })
        self.known_masked_cell_df.columns.names = ['coord', 'i']
        self.known_masked_cell_df.index.names = ['j']

        self.known_coord_pairs = np.array([
            [1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [1.0, 0.5],
            [1.5, 0.5], [2.0, 0.5], [1.0, 1.0], [1.5, 1.0],
            [2.0, 1.0], [1.0, 1.5], [1.5, 1.5], [2.0, 1.5],
            [1.0, 2.0], [1.5, 2.0], [2.0, 2.0], [1.0, 2.5],
            [1.5, 2.5], [2.0, 2.5], [1.0, 3.0], [1.5, 3.0],
            [2.0, 3.0], [1.0, 3.5], [1.5, 3.5], [2.0, 3.5],
            [1.0, 4.0], [1.5, 4.0], [2.0, 4.0]
        ])

        self.known_mask = np.array([
            [1, 1], [1, 1], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
        ], dtype=bool)

        self.known_node_pairs_masked = np.array([
            [ nan,  nan], [ nan,  nan], [ nan,  nan], [ nan,  nan],
            [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
            [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
            [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
        ])

        self.known_node_pairs = np.array([
            [1.25, 0.25], [1.75, 0.25], [1.25, 0.75], [1.75, 0.75],
            [1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75],
            [1.25, 2.25], [1.75, 2.25], [1.25, 2.75], [1.75, 2.75],
            [1.25, 3.25], [1.75, 3.25], [1.25, 3.75], [1.75, 3.75]
        ])

    def test_to_dataframe_nomask_nodes(self):
        pdtest.assert_frame_equal(
            self.g1.to_dataframe(usemask=False, which='nodes'),
            self.known_df,
            check_names=False
        )

        pdtest.assert_index_equal(
            self.g1.to_dataframe().columns,
            self.known_df.columns,
        )

    def test_to_coord_pairs_nomask_nodes(self):
        nptest.assert_array_equal(
            self.g1.to_coord_pairs(usemask=False, which='nodes'),
            self.known_coord_pairs
        )

    def test_to_dataframe_mask_nodes(self):
        with pytest.raises(ValueError):
            self.g1.to_dataframe(usemask=True, which='nodes')

    def test_to_coord_pairs_mask_nodes(self):
        with pytest.raises(ValueError):
            self.g1.to_coord_pairs(usemask=True, which='nodes')

    def test_to_coord_pairs_nomask_cells(self):
        nptest.assert_array_equal(
            self.g1.to_coord_pairs(usemask=False, which='cells'),
            self.known_node_pairs
        )

    def test_to_coord_pairs_mask_cells(self):
        self.g1.cell_mask = self.known_mask
        nptest.assert_array_equal(
            self.g1.to_coord_pairs(usemask=True, which='cells'),
            self.known_node_pairs_masked
        )

    def test_to_dataframe_mask_cells(self):
        self.g1.cell_mask = self.known_mask
        df = self.g1.to_dataframe(usemask=True, which='cells')
        pdtest.assert_frame_equal(df, self.known_masked_cell_df,
                                  check_names=False)

        pdtest.assert_index_equal(
            df.columns, self.known_masked_cell_df.columns,
        )

    def test_transform(self):
        gx = self.g1.xn.copy() * 10
        g = self.g1.transform(lambda x: x * 10)
        nptest.assert_array_equal(g.xn, gx)

    def test_transpose(self):
        gx = self.g1.xn.copy()
        nptest.assert_array_equal(
            self.g1.transpose().xn,
            gx.transpose()
        )

    def test_fliplr(self):
        gx = np.fliplr(self.g1.xn.copy())
        g = self.g1.fliplr()
        nptest.assert_array_equal(g.xn, gx)

    def test_flipud(self):
        gx = np.flipud(self.g1.xn.copy())
        g = self.g1.flipud()
        nptest.assert_array_equal(g.xn, gx)

    def test_split_ax0(self):
        mgtop, mgbottom = self.mg.split(3, axis=0)
        nptest.assert_array_equal(mgtop.nodes_x, self.xn[:3, :])
        nptest.assert_array_equal(mgtop.nodes_y, self.yn[:3, :])
        nptest.assert_array_equal(mgbottom.nodes_x, self.xn[3:, :])
        nptest.assert_array_equal(mgbottom.nodes_y, self.yn[3:, :])

    def test_split_ax0(self):
        mgtop, mgbottom = self.mg.split(3, axis=1)
        nptest.assert_array_equal(mgtop.nodes_x, self.xn[:, :3])
        nptest.assert_array_equal(mgtop.nodes_y, self.yn[:, :3])
        nptest.assert_array_equal(mgbottom.nodes_x, self.xn[:, 3:])
        nptest.assert_array_equal(mgbottom.nodes_y, self.yn[:, 3:])

    def test_merge(self):
        g3 = self.g1.merge(self.g2, how='horiz', where='+', shift=2)
        g4 = core.ModelGrid(self.xn, self.yn)

        nptest.assert_array_equal(g3.xn, g4.xn)
        nptest.assert_array_equal(g3.xc, g4.xc)

    def test_insert_3_ax0(self):
        known_xnodes = np.ma.masked_invalid(np.array([
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

        known_ynodes = np.ma.masked_invalid(np.array([
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

        result = self.mg.insert(2, axis=0, n_nodes=3)
        nptest.assert_array_equal(result.nodes_x, known_xnodes)
        nptest.assert_array_equal(result.nodes_y, known_ynodes)

    def test_insert_3_ax1(self):
        known_xnodes = np.ma.masked_invalid(np.array([
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

        known_ynodes = np.ma.masked_invalid(np.array([
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

        result = self.mg.insert(2, axis=1, n_nodes=3)
        nptest.assert_array_equal(result.nodes_x, known_xnodes)
        nptest.assert_array_equal(result.nodes_y, known_ynodes)

    def test_extract(self):
        result = self.mg.extract(jstart=2, jend=5, istart=3, iend=6)
        nptest.assert_array_equal(result.nodes_x, self.xn[2:5, 3:6])
        nptest.assert_array_equal(result.nodes_y, self.yn[2:5, 3:6])

    def test_mask_cells_with_polygon_inside_not_inplace(self):
        orig_mask = self.mg.cell_mask.copy()
        masked = self.mg.mask_cells_with_polygon(self.polyverts, use_existing=False)
        known_inside_mask = np.array([
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False,  True,  True, False],
            [False, False, False,  True,  True, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False]
        ], dtype=bool)
        nptest.assert_array_equal(masked.cell_mask, known_inside_mask)

    def test_mask_cells_with_polygon_inside_use_existing(self):
        self.mg.cell_mask = np.ma.masked_invalid(self.mg.xc).mask
        masked = self.mg.mask_cells_with_polygon(self.polyverts, use_existing=True)
        known_inside_mask = np.array([
            [False, False,  True, True, True,  True],
            [False, False,  True, True, True,  True],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False,  True, True, True,  True],
            [False, False,  True, True, True,  True],
            [False, False,  True, True, True,  True],
            [False, False,  True, True, True,  True]
        ], dtype=bool)
        nptest.assert_array_equal(masked.cell_mask, known_inside_mask)

    def test_mask_cells_with_polygon_outside(self):
        masked = self.mg.mask_cells_with_polygon(self.polyverts, inside=False, use_existing=False)
        known_outside_mask = np.array([
            [True, True, True,  True,  True, True],
            [True, True, True,  True,  True, True],
            [True, True, True, False, False, True],
            [True, True, True, False, False, True],
            [True, True, True,  True,  True, True],
            [True, True, True,  True,  True, True],
            [True, True, True,  True,  True, True],
            [True, True, True,  True,  True, True]
        ], dtype=bool)
        nptest.assert_array_equal(masked.cell_mask, known_outside_mask)

    def test_mask_cells_with_polygon_use_nodes(self):
        masked = self.mg.mask_cells_with_polygon(self.polyverts, use_centroids=False,
                                                 use_existing=False)
        known_node_mask = np.array([
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False,  True,  True, False],
            [False, False, False,  True,  True, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False]
        ], dtype=bool)
        nptest.assert_array_equal(masked.cell_mask, known_node_mask)

    def test_mask_cells_with_polygon_nodes_too_few_nodes(self):
        with pytest.raises(ValueError):
            self.mg.mask_cells_with_polygon(
                self.polyverts, use_centroids=False, min_nodes=0
            )

    def test_mask_cells_with_polygon_nodes_too_many_nodes(self):
        with pytest.raises(ValueError):
            self.mg.mask_cells_with_polygon(
                self.polyverts, use_centroids=False, min_nodes=5
            )

    def test_mask_cells_with_polygon_triangles(self):
        with pytest.raises(NotImplementedError):
            self.mg.mask_cells_with_polygon(self.polyverts, triangles=True)

    def test_to_shapefile_bad_geom(self):
        with pytest.raises(ValueError):
            self.g1.to_shapefile('junk', geom='Line')

    def test_to_shapefile_nomask_nodes_points(self):
        outfile = 'pygridtools/tests/result_files/mgshp_nomask_nodes_points.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_nomask_nodes_points.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='nodes',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_cells_points(self):
        outfile = 'pygridtools/tests/result_files/mgshp_nomask_cells_points.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_nomask_cells_points.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='cells',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_nodes_polys(self):
        outfile = 'pygridtools/tests/result_files/mgshp_nomask_nodes_polys.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_nomask_cells_polys.shp'
        self.g1.to_shapefile(outfile, usemask=False, which='nodes',
                             geom='polygon')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_nomask_cells_polys(self):
        outfile = 'pygridtools/tests/result_files/mgshp_nomask_cells_polys.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_nomask_cells_polys.shp'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.g1.to_shapefile(outfile, usemask=False, which='cells',
                                 geom='polygon')
            assert (len(w) == 1)

        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_mask_cells_points(self):
        outfile = 'pygridtools/tests/result_files/mgshp_mask_cells_points.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_mask_cells_points.shp'
        self.g1.cell_mask = self.known_mask

        self.g1.to_shapefile(outfile, usemask=True, which='cells',
                             geom='point')
        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_mask_cells_polys(self):
        outfile = 'pygridtools/tests/result_files/mgshp_mask_cells_polys.shp'
        basefile = 'pygridtools/tests/baseline_files/mgshp_mask_cells_polys.shp'
        self.g1.cell_mask = self.known_mask

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.g1.to_shapefile(outfile, usemask=True, which='cells',
                                 geom='polygon')
            assert (len(w) == 1)

        testing.compareShapefiles(outfile, basefile)

    def test_to_shapefile_mask_nodes(self):
        with pytest.raises(ValueError):
            self.g1.to_shapefile('junk', usemask=True, which='nodes', geom='point')

    def test__get_x_y_nodes_and_mask(self):
        with pytest.raises(ValueError):
            self.g1._get_x_y('nodes', usemask=True)

    def test__get_x_y_bad_value(self):
        with pytest.raises(ValueError):
            self.g1._get_x_y('junk', usemask=True)

    def test__get_x_y_nodes(self):
        x, y = self.g1._get_x_y('nodes', usemask=False)
        nptest.assert_array_equal(x, self.g1.xn)
        nptest.assert_array_equal(y, self.g1.yn)

    def test__get_x_y_cells(self):
        x, y = self.g1._get_x_y('cells', usemask=False)
        nptest.assert_array_equal(x, self.g1.xc)
        nptest.assert_array_equal(y, self.g1.yc)

    def test_writeGEFDCControlFile(self):
        known_filename = 'pygridtools/tests/baseline_files/modelgrid_gefdc.inp'
        result_path = 'pygridtools/tests/result_files'
        result_file = 'modelgrid_gefdc.inp'
        self.mg.writeGEFDCControlFile(
            outputdir=result_path,
            filename=result_file,
            title='Model Grid Test'
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCCellFile(self):
        known_filename = 'pygridtools/tests/baseline_files/modelgrid_cell.inp'
        result_path = 'pygridtools/tests/result_files'
        result_file = 'modelgrid_cell.inp'
        self.mg.writeGEFDCCellFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCGridFile(self):
        known_filename = 'pygridtools/tests/baseline_files/modelgrid_grid.out'
        result_path = 'pygridtools/tests/result_files'
        result_file = 'modelgrid_grid.out'
        self.mg.writeGEFDCGridFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )

    def test_writeGEFDCGridextFiles(self):
        known_filename = 'pygridtools/tests/baseline_files/modelgrid_gridext.inp'
        result_path = 'pygridtools/tests/result_files'
        result_file = 'modelgrid_gridext.inp'
        self.mg.writeGEFDCGridextFile(
            outputdir=result_path,
            filename=result_file,
        )
        testing.compareTextFiles(
            os.path.join(result_path, result_file),
            known_filename
        )


@image_comparison(
    baseline_images=[
        'test_ModelGrid_plots_basic',
    ],
    extensions=['png']
)
def test_ModelGrid_plots():
    xn, yn = testing.makeSimpleNodes()
    mg = core.ModelGrid(xn, yn)
    mg.cell_mask = np.ma.masked_invalid(mg.xc).mask

    fig1 = mg.plotCells()


class Test_makeGrid(object):
    def setup(self):
        self.domain = testing.makeSimpleBoundary()
        self.bathy = testing.makeSimpleBathy()
        self.nx = 9
        self.ny = 7
        self.gridparams = {
            'nnodes': 12,
            'verbose': False,
            'ul_idx': 0
        }

    @nptest.dec.skipif(not has_pgg)
    def test_with_and_bathy(self):
        grid = core.makeGrid(
            self.ny, self.nx,
            domain=self.domain,
            bathydata=self.bathy.dropna(),
            **self.gridparams
        )
        assert (isinstance(grid, pygridgen.Gridgen))

    @nptest.dec.skipif(not has_pgg)
    def test_with_bathy_verbose(self):
        params = self.gridparams.copy()
        params['verbose'] = True
        grid = core.makeGrid(
            self.ny, self.nx,
            domain=self.domain,
            bathydata=self.bathy.dropna(),
            **self.gridparams
        )
        assert (isinstance(grid, pygridgen.Gridgen))

    @nptest.dec.skipif(not has_pgg)
    def test_as_ModelGrid(self):
        grid = core.makeGrid(
            self.ny, self.nx,
            domain=self.domain,
            rawgrid=False,
            **self.gridparams
        )
        assert (isinstance(grid, core.ModelGrid))
