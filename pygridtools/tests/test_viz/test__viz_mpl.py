import sys

import pytest

from pygridtools.viz import _viz_mpl


WINDOWS = sys.platform.lower() == 'win32'
BASELINE_IMAGES = '../baseline_files/test_viz/test__viz_mpl'


@pytest.fixture
def domain_artists_keys():
    return sorted(['domain', 'beta', 'legend'])


@pytest.fixture
def boundary_artists_keys():
    return sorted(['extent', 'islands'])


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_with_beta_df(simple_boundary, domain_artists_keys):
    fig, artists = _viz_mpl._plot_domain(
        domain_x='x',
        domain_y='y',
        beta='beta',
        data=simple_boundary
    )
    assert sorted(artists.keys()) == domain_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_with_beta_array(simple_boundary, domain_artists_keys):
    fig, artists = _viz_mpl._plot_domain(
        domain_x=simple_boundary['x'],
        domain_y=simple_boundary['y'],
        beta=simple_boundary['beta'],
        data=None
    )
    assert sorted(artists.keys()) == domain_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_without_beta_df(simple_boundary, domain_artists_keys):
    fig, artists = _viz_mpl._plot_domain(
        domain_x='x',
        domain_y='y',
        data=simple_boundary
    )
    assert sorted(artists.keys()) == domain_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_domain_without_beta_array(simple_boundary, domain_artists_keys):
    fig, artists = _viz_mpl._plot_domain(
        domain_x=simple_boundary['x'],
        domain_y=simple_boundary['y'],
        data=None
    )
    assert sorted(artists.keys()) == domain_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_model_df(simple_boundary, simple_islands,
                                  boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        extent_x='x',
        extent_y='y',
        extent=simple_boundary
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_model_array(simple_boundary, simple_islands,
                                     boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        extent_x=simple_boundary['x'],
        extent_y=simple_boundary['y'],
        extent=None
    )
    assert sorted(artists.keys()) == boundary_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_island_df(simple_boundary, simple_islands,
                                   boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        islands_x='x',
        islands_y='y',
        islands_name='island',
        islands=simple_islands
    )
    assert sorted(artists.keys()) == boundary_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_just_island_array(simple_boundary, simple_islands,
                                      boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        islands_x=simple_islands['x'],
        islands_y=simple_islands['y'],
        islands_name=simple_islands['island'],
        islands=None
    )
    assert sorted(artists.keys()) == boundary_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_both_df(simple_boundary, simple_islands,
                            boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        extent_x='x',
        extent_y='y',
        extent=simple_boundary,
        islands_x='x',
        islands_y='y',
        islands_name='island',
        islands=simple_islands
    )
    assert sorted(artists.keys()) == boundary_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_boundaries_both_array(simple_boundary, simple_islands,
                               boundary_artists_keys):
    fig, artists = _viz_mpl._plot_boundaries(
        extent_x=simple_boundary['x'],
        extent_y=simple_boundary['y'],
        extent=None,
        islands_x=simple_islands['x'],
        islands_y=simple_islands['y'],
        islands_name=simple_islands['island'],
        islands=None
    )
    assert sorted(artists.keys()) == boundary_artists_keys
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_points_array(simple_nodes):
    fig, artists = _viz_mpl._plot_points(*simple_nodes)
    assert sorted(artists.keys()) == ['points']
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_cells_array(simple_nodes):
    fig, artists = _viz_mpl._plot_cells(*simple_nodes)
    assert sorted(artists.keys()) == ['cells']
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_IMAGES, tolerance=15)
def test_cells_array_masked(river_grid, river_bathy):
    fig, artists = _viz_mpl._plot_cells(river_grid.xn, river_grid.yn, colors=river_bathy,
                                       edgecolor='0.45', cmap='Blues_r', lw=0.75)
    assert sorted(artists.keys()) == ['cells']
    return fig
