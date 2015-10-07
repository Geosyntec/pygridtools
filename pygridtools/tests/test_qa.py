import numpy as np
import matplotlib.pyplot as plt

import nose.tools as nt
from matplotlib.testing.decorators import  cleanup


from pygridtools import qa


class test__check_ax(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()

    @cleanup
    def teardown(self):
        self.ax.clear()
        plt.close(self.fig)

    @cleanup
    def test_with_ax(self):
        fig, ax = qa._check_ax(self.ax)
        nt.assert_equal(self.fig, fig)
        nt.assert_equal(self.ax, ax)

    @cleanup
    def test_without_ax(self):
        fig, ax = qa._check_ax(None)
        nt.assert_not_equals(self.fig, fig)
        nt.assert_not_equals(self.ax, ax)

        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))

    @cleanup
    @nt.raises(AttributeError)
    def test_bad_ax(self):
        qa._check_ax('junk')


class test__validate_polygon(object):
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
        poly = qa._validate_polygon(self.poly_list)
        nt.assert_true(np.all(self.poly_array == poly))

    def test_array(self):
        poly = qa._validate_polygon(self.poly_array)
        nt.assert_true(np.all(self.poly_array == poly))

    @nt.raises(ValueError)
    def test_too_wide(self):
        _ = qa._validate_polygon(self.too_wide)

    @nt.raises(ValueError)
    def test_too_short(self):
        _ = qa._validate_polygon(self.too_short)

    @nt.raises(ValueError)
    def test_wrong_dims(self):
        _ = qa._validate_polygon([self.poly_array, self.poly_array])


class test__validate_xy_array(object):
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
        x, y = qa._validate_xy_array(self.y1, self.x1, as_pairs=False)
        nt.assert_true(np.all(x == self.y1))
        nt.assert_true(np.all(y == self.x1))

    def test_as_pairs(self):
        xy = qa._validate_xy_array(self.y2, self.x2)
        nt.assert_true(np.all(xy == self.known_pairs))

    @nt.raises(ValueError)
    def test_diff_shapes(self):
        _ = qa._validate_xy_array(self.y1, self.x2)

    @nt.raises(ValueError)
    def test_diff_masks(self):
        y = np.ma.MaskedArray(data=self.y2, mask=self.mask1)
        x = np.ma.MaskedArray(data=self.x2, mask=self.mask2)
        _ = qa._validate_xy_array(x, y)

    @nt.raises(ValueError)
    def test_only_one_mask(self):
        y = np.ma.MaskedArray(data=self.y2, mask=self.mask1)
        _ = qa._validate_xy_array(self.x2, y)
