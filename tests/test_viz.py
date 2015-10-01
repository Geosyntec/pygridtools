import numpy as np
import matplotlib.pyplot as plt

from pygridtools import viz
import testing

import nose.tools as nt
import numpy.testing as nptest


class test__check_ax(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()

    def teardown(self):
        self.ax.clear()
        plt.close(self.fig)

    def test_with_ax(self):
        fig, ax = viz._check_ax(self.ax)
        nt.assert_equal(self.fig, fig)
        nt.assert_equal(self.ax, ax)

    def test_without_ax(self):
        fig, ax = viz._check_ax(None)
        nt.assert_not_equals(self.fig, fig)
        nt.assert_not_equals(self.ax, ax)

        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))

    @nt.raises(AttributeError)
    def test_bad_ax(self):
        viz._check_ax('junk')


class BasePlotChecker_Mixin(object):
    def setup(self):
        self.boundary = testing.makeSimpleBoundary()
        self.grid = testing.makeSimpleGrid()
        self.nodes = testing.makeSimpleNodes()

    def teardown(self):
        plt.close('all')


class test_plotReachDF(BasePlotChecker_Mixin):
    def test_smoketest_withoutax(self):
        fig = viz.plotReachDF(self.boundary, 'x', 'y')
        nt.assert_true(isinstance(fig, plt.Figure))
        figfile = 'tests/result_images/plotreach_withoutax.png'
        fig.savefig(figfile, dpi=150)

    @nt.raises(ValueError)
    def test_badinput(self):
        viz.plotReachDF(self.boundary.values, 'x', 'y')


class test_plotPygridgen(BasePlotChecker_Mixin):
    @nt.raises(AttributeError)
    def test_bad_grid(self):
        viz.plotPygridgen('junk')

    @nt.raises(AttributeError)
    def test_bad_axes(self):
        viz.plotPygridgen(self.grid, ax='junk')

    def test_plot_smoketest(self):
        fig = viz.plotPygridgen(self.grid)
        fig.savefig("tests/result_images/gridsmoke.png", dpi=150)


class test_plotCells(BasePlotChecker_Mixin):
    def test_plot_smoke_test(self):
        fig = viz.plotCells(self.nodes[0], self.nodes[1])
        fig.savefig("tests/result_images/cellsmoke.png", dpi=150)

    @nt.raises(ValueError)
    def test_bad_shape(self):
        fig = viz.plotCells(self.nodes[0][1:,:], self.nodes[1])

    @nt.raises(NotImplementedError)
    def test_not_implemented_engine(self):
        fig = viz.plotCells(self.nodes[0], self.nodes[1], engine='bokeh')

    @nt.raises(ValueError)
    def test_bad_engine(self):
        fig = viz.plotCells(self.nodes[0], self.nodes[1], engine='JUNK')


@nt.raises(NotImplementedError)
def test__plot_cells_bokeh():
    x, y = testing.makeSimpleNodes()
    p = viz._plot_cells_bokeh(x, y)


def test__plot_cell_mpl():
    x, y = testing.makeSimpleNodes()
    fig, ax = viz._plot_cells_mpl(x, y)
    nt.assert_true(isinstance(fig, plt.Figure))


@nt.raises(NotImplementedError)
def test_plotCells_bokeh():
    x, y = testing.makeSimpleNodes()
    p = viz.plotCells(x, y, engine='bokeh')


def test_plotCells_mpl():
    x, y = testing.makeSimpleNodes()
    p = viz.plotCells(x, y, engine='mpl')

