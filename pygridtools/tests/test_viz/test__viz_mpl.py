import sys

import numpy.testing as nptest
from matplotlib.testing.decorators import image_comparison
import pygridtools.testing as pgtest

from pygridtools.viz import _viz_mpl


WINDOWS = sys.platform.lower() == 'win32'


@image_comparison(
    baseline_images=[
        'test_domain_with_beta_df',
        'test_domain_with_beta_array',
    ],
    extensions=['png']
)
@nptest.dec.skipif(WINDOWS)
def test__plot_domain_with_beta():
    data = pgtest.makeSimpleBoundary()
    fig3 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', beta='beta', data=data)
    fig4 = _viz_mpl._plot_domain(domain_x=data['x'], domain_y=data['y'],
                                 beta=data['beta'], data=None)


@image_comparison(
    baseline_images=[
        'test_domain_without_beta_df',
        'test_domain_without_beta_array',
    ],
    extensions=['png']
)
def test__plot_domain_without_beta():
    data = pgtest.makeSimpleBoundary()
    fig1 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', data=data)
    fig2 = _viz_mpl._plot_domain(domain_x=data['x'], domain_y=data['y'], data=None)


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
    fig3 = _viz_mpl._plot_boundaries(islands_x='x', islands_y='y', islands_name='island',
                                     islands=islands)
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
