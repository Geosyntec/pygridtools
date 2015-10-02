import numpy as np
import matplotlib.pyplot as plt

from pygridtools.viz import _viz_mpl

import nose.tools as nt
import numpy.testing as nptest
from matplotlib.testing.decorators import image_comparison, cleanup
import pygridtools.testing as pgtest


class test__check_ax(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()

    @cleanup
    def teardown(self):
        self.ax.clear()
        plt.close(self.fig)

    @cleanup
    def test_with_ax(self):
        fig, ax = _viz_mpl._check_ax(self.ax)
        nt.assert_equal(self.fig, fig)
        nt.assert_equal(self.ax, ax)

    @cleanup
    def test_without_ax(self):
        fig, ax = _viz_mpl._check_ax(None)
        nt.assert_not_equals(self.fig, fig)
        nt.assert_not_equals(self.ax, ax)

        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))

    @cleanup
    @nt.raises(AttributeError)
    def test_bad_ax(self):
        _viz_mpl._check_ax('junk')


@image_comparison(
    baseline_images=[
        'test_domain_without_beta_df',
        'test_domain_without_beta_array',
        'test_domain_with_beta_df',
        'test_domain_with_beta_array',
    ],
    extensions=['png']
)
def test__plot_domain():
    data = pgtest.makeSimpleBoundary()
    fig1 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', data=data)
    fig2 = _viz_mpl._plot_domain(domain_x=data['x'], domain_y=data['y'], data=None)
    fig3 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', beta='beta', data=data)
    fig4 = _viz_mpl._plot_domain(domain_x=data['x'], domain_y=data['y'], beta=data['beta'], data=None)


@image_comparison(
    baseline_images=[
        'test_boundaries_just_model_df',
        'test_boundaries_just_model_array',
        'test_boundaries_just_island_df',
        'test_boundaries_just_island_array',
        'test_boundaries_both_df',
        'test_boundaries_both_array',
    ],
    extensions=['png']
)
def test__plot_boundaries():
    extent = pgtest.makeSimpleBoundary()
    islands = pgtest.makeSimpleIslands()
    fig1 = _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=extent)
    fig2 = _viz_mpl._plot_boundaries(extent_x=extent['x'], extent_y=extent['y'], extent=None)
    fig3 = _viz_mpl._plot_boundaries(islands_x='x', islands_y='y', islands_name='island', islands=islands)
    fig4 = _viz_mpl._plot_boundaries(islands_x=islands['x'], islands_y=islands['y'],
                                     islands_name=islands['island'], islands=None)

    fig5 = _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=extent,
                                     islands_x='x', islands_y='y', islands_name='island',
                                     islands=islands)

    fig6 = _viz_mpl._plot_boundaries(extent_x=extent['x'], extent_y=extent['y'], extent=None,
                                     islands_x=islands['x'], islands_y=islands['y'],
                                     islands_name=islands['island'], islands=None)

@image_comparison(
    baseline_images=[
        'test_points_array',
    ],
    extensions=['png']
)
def test__plot_points():
    x, y = pgtest.makeSimpleNodes()
    fig1 = _viz_mpl._plot_points(x, y)


@image_comparison(
    baseline_images=[
        'test_cells_array',
    ],
    extensions=['png']
)
def test__plot_cells():
    x, y = pgtest.makeSimpleNodes()
    fig1 = _viz_mpl._plot_cells(x, y)
