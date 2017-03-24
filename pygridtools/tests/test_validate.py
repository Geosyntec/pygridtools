import numpy as np
import matplotlib.pyplot as plt

import pytest
import numpy.testing as nptest


from pygridtools import validate


class Test_mpl_ax(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()

    def teardown(self):
        self.ax.clear()
        plt.close(self.fig)

    def test_with_ax(self):
        fig, ax = validate.mpl_ax(self.ax)
        assert self.fig == fig
        assert self.ax == ax

    def test_without_ax(self):
        fig, ax = validate.mpl_ax(None)
        assert self.fig != fig
        assert self.ax != ax

        assert (isinstance(fig, plt.Figure))
        assert (isinstance(ax, plt.Axes))

    def test_bad_ax(self):
        with pytest.raises(AttributeError):
            validate.mpl_ax('junk')


class Test_polygon(object):
    def setup(self):
        self.poly_list = [
            (2, 2),
            (5, 2),
            (5, 5),
            (2, 5),
        ]

        self.poly_array = np.array(self.poly_list)

        self.too_short = self.poly_list[:2]
        self.too_wide = [
            (2, 2, 1),
            (5, 2, 1),
            (5, 5, 1),
            (2, 5, 1),
        ]

    def test_list(self):
        poly = validate.polygon(self.poly_list)
        assert (np.all(self.poly_array == poly))

    def test_array(self):
        poly = validate.polygon(self.poly_array)
        assert (np.all(self.poly_array == poly))

    def test_too_wide(self):
        with pytest.raises(ValueError):
            validate.polygon(self.too_wide)

    def test_too_short(self):
        with pytest.raises(ValueError):
            validate.polygon(self.too_short)

    def test_wrong_dims(self):
        with pytest.raises(ValueError):
            validate.polygon([self.poly_array, self.poly_array])


class Test_xy_array(object):
    def setup(self):
        self.y1, self.x1 = np.mgrid[:9, :9]
        self.y2, self.x2 = np.mgrid[:3, :3]
        self.known_pairs = np.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2]
        ])

        self.mask1 = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])

        self.mask2 = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ])

    def test_not_as_pairs(self):
        x, y = validate.xy_array(self.y1, self.x1, as_pairs=False)
        assert (np.all(x == self.y1))
        assert (np.all(y == self.x1))

    def test_as_pairs(self):
        xy = validate.xy_array(self.y2, self.x2)
        assert (np.all(xy == self.known_pairs))

    def test_diff_shapes(self):
        with pytest.raises(ValueError):
            validate.xy_array(self.y1, self.x2)

    def test_diff_masks(self):
        y = np.ma.MaskedArray(data=self.y2, mask=self.mask1)
        x = np.ma.MaskedArray(data=self.x2, mask=self.mask2)
        with pytest.raises(ValueError):
            validate.xy_array(x, y)

    def test_only_one_mask(self):
        y = np.ma.MaskedArray(data=self.y2, mask=self.mask1)
        with pytest.raises(ValueError):
            validate.xy_array(self.x2, y)


class Test_file_mode(object):

    def test_errors(self):
        with pytest.raises(ValueError):
            validate.file_mode('z')

    def test_upper(self):
        assert validate.file_mode('A') == 'a'

    def test_lower(self):
        assert validate.file_mode('w') == 'w'


class Test_elev_or_mask(object):
    def setup(self):
        self.mainshape = (8, 7)
        self.offset = 2
        self.offsetshape = tuple([s - self.offset for s in self.mainshape])
        self.X = np.zeros(self.mainshape)
        self.Y = np.zeros(self.mainshape)
        self.Yoffset = np.zeros(self.offsetshape)

    def test_failNone(self):
        with pytest.raises(ValueError):
            validate.elev_or_mask(self.X, None, failNone=True)

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            validate.elev_or_mask(self.X, self.Yoffset)

    def test_offset(self):
        other = validate.elev_or_mask(self.X, self.Yoffset,
                                      offset=self.offset)
        nptest.assert_array_equal(other, self.Yoffset)

    def test_nooffset(self):
        other = validate.elev_or_mask(self.X, self.Y, offset=0)
        nptest.assert_array_equal(other, self.Y)


class Test_equivalent_masks(object):
    def setup(self):
        from numpy import nan
        self.X = np.array([
            1, 2, 3, nan, nan, 7,
            1, 2, 3, nan, nan, 7,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, 7,
        ])

        self.Y1 = self.X.copy()

        self.Y2 = np.array([
            1, 2, 3, nan, nan, 7,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, nan,
            1, 2, 3, nan, nan, 7,
        ])

    def test_error(self):
        with pytest.raises(ValueError):
            validate.equivalent_masks(self.X, self.Y2)

    def test_baseline(self):
        x, y = validate.equivalent_masks(self.X, self.Y1)
        nptest.assert_array_equal(self.X, x.data)
        nptest.assert_array_equal(self.Y1, y.data)
