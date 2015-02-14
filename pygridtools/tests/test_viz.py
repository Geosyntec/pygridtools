import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn

from pygridtools import viz
import testing

import nose.tools as nt
import numpy.testing as nptest

class test_checkAx:
    def setup(self):
        self.fig, self.ax = plt.subplots()

    def teardown(self):
        self.ax.clear()
        plt.close(self.fig)

    def test_with_ax(self):
        fig, ax = viz.checkAx(self.ax)
        nt.assert_equal(self.fig, fig)
        nt.assert_equal(self.ax, ax)

    def test_without_ax(self):
        fig, ax = viz.checkAx(None)
        nt.assert_not_equals(self.fig, fig)
        nt.assert_not_equals(self.ax, ax)

        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))

    @nt.raises(AttributeError)
    def test_bad_ax(self):
        viz.checkAx('junk')


class test_plotReachDF():
    def setup(self):
        self.boundary = testing.makeSimpleBoundary()
        plt.close('all')

    def teardown(self):
        plt.close('all')

    def test_smoketest_withoutax(self):
        fig = viz.plotReachDF(self.boundary, 'x', 'y', 'reach')
        nt.assert_true(isinstance(fig, seaborn.FacetGrid))
        figfile = 'pygridtools/tests/result_images/plotreach_withoutax.png'
        fig.savefig(figfile, dpi=150)

    def test_smoketest_withflipped(self):
        fig = viz.plotReachDF(self.boundary, 'x', 'y', 'reach', flip=True)
        figfile = 'pygridtools/tests/result_images/plotreach_flip.png'
        fig.savefig(figfile, dpi=150)

    @nt.raises(ValueError)
    def test_badinput(self):
        viz.plotReachDF(self.boundary.values, 'x', 'y', 'reach', flip=True)


class test_plotPygridgen:
    def setup(self):
        self.grid = testing.makeSimpleGrid()
        plt.close('all')


    @nt.raises(AttributeError)
    def test_bad_grid(self):
        viz.plotPygridgen('junk')

    @nt.raises(AttributeError)
    def test_bad_axes(self):
        viz.plotPygridgen(self.grid, ax='junk')

    def test_plot_smoketest(self):
        fig, ax = viz.plotPygridgen(self.grid)
        fig.savefig("pygridtools/tests/result_images/gridsmoke.png", dpi=150)
