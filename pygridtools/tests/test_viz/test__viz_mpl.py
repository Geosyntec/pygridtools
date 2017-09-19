import sys

import pytest
from matplotlib.testing.decorators import image_comparison

from pygridtools.viz import _viz_mpl


WINDOWS = sys.platform.lower() == 'win32'


@image_comparison(
    baseline_images=[
        'test_domain_with_beta_df',
        'test_domain_with_beta_array',
    ],
    extensions=['png']
)
@pytest.mark.skipif(WINDOWS, 'windows')
def test__plot_domain_with_beta(simple_boundary):
    fig3 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', beta='beta', data=simple_boundary)
    fig4 = _viz_mpl._plot_domain(domain_x=simple_boundary['x'], domain_y=simple_boundary['y'],
                                 beta=simple_boundary['beta'], data=None)


@image_comparison(
    baseline_images=[
        'test_domain_without_beta_df',
        'test_domain_without_beta_array',
    ],
    extensions=['png']
)
def test__plot_domain_without_beta():
    fig1 = _viz_mpl._plot_domain(domain_x='x', domain_y='y', data=simple_boundary)
    fig2 = _viz_mpl._plot_domain(domain_x=simple_boundary['x'], domain_y=simple_boundary['y'], data=None)


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
def test__plot_boundaries(simple_boundary, simple_islands):
    fig1 = _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=simple_boundary)
    fig2 = _viz_mpl._plot_boundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None)
    fig3 = _viz_mpl._plot_boundaries(islands_x='x', islands_y='y', islands_name='island',
                                     islands=simple_islands)
    fig4 = _viz_mpl._plot_boundaries(islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                                     islands_name=simple_islands['island'], islands=None)

    fig5 = _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=simple_boundary,
                                     islands_x='x', islands_y='y', islands_name='island',
                                     islands=simple_islands)

    fig6 = _viz_mpl._plot_boundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None,
                                     islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                                     islands_name=simple_islands['island'], islands=None)


@image_comparison(
    baseline_images=[
        'test_points_array',
    ],
    extensions=['png']
)
def test__plot_points(simple_nodes):
    fig1 = _viz_mpl._plot_points(*simple_nodes)


@image_comparison(
    baseline_images=[
        'test_cells_array',
    ],
    extensions=['png']
)
def test__plot_cells(simple_nodes):
    fig1 = _viz_mpl._plot_cells(*simple_nodes)
