import sys

import pytest

from pygridtools.viz import _viz_mpl


WINDOWS = sys.platform.lower() == 'win32'
BASELINE_IMAGES = '../baseline_files/test_viz/test__viz_mpl'


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_with_beta_df(simple_boundary):
    return _viz_mpl._plot_domain(domain_x='x', domain_y='y', beta='beta', data=simple_boundary)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_with_beta_array(simple_boundary):
    return _viz_mpl._plot_domain(domain_x=simple_boundary['x'], domain_y=simple_boundary['y'],
                                 beta=simple_boundary['beta'], data=None)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_without_beta_df(simple_boundary):
    return _viz_mpl._plot_domain(domain_x='x', domain_y='y', data=simple_boundary)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_without_beta_array(simple_boundary):
    return _viz_mpl._plot_domain(domain_x=simple_boundary['x'], domain_y=simple_boundary['y'], data=None)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_model_df(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=simple_boundary)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_model_array(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_island_df(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(islands_x='x', islands_y='y', islands_name='island',
                                     islands=simple_islands)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_island_array(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                                     islands_name=simple_islands['island'], islands=None)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_both_df(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(extent_x='x', extent_y='y', extent=simple_boundary,
                                     islands_x='x', islands_y='y', islands_name='island',
                                     islands=simple_islands)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_both_array(simple_boundary, simple_islands):
    return _viz_mpl._plot_boundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None,
                                     islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                                     islands_name=simple_islands['island'], islands=None)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_points_array(simple_nodes):
    return _viz_mpl._plot_points(*simple_nodes)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_cells_array(simple_nodes):
    return _viz_mpl._plot_cells(*simple_nodes)
