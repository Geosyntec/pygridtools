
from pygridtools import viz

import pytest


@pytest.mark.parametrize('engine', ['mpl'])
def test_plot_domain_smoke(simple_boundary_gdf, engine):
    fig3 = viz.plot_domain(simple_boundary_gdf, betacol='beta')  # noqa: F841


@pytest.mark.parametrize('engine', ['mpl'])
def test_plot_boundaries_smoke(simple_boundary, simple_islands, engine):
    fig5 = viz.plot_boundaries(extent_x='x', extent_y='y', extent=simple_boundary,  # noqa: F841
                              islands_x='x', islands_y='y', islands_name='island',
                              islands=simple_islands)

    fig6 = viz.plot_boundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None,  # noqa: F841
                              islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                              islands_name=simple_islands['island'], islands=None)


def test_plot_points_smoke(simple_nodes):
    fig1 = viz.plot_points(*simple_nodes)  # noqa: F841


def test_plot_cells_smoke(simple_nodes):
    fig1 = viz.plot_cells(*simple_nodes)  # noqa: F841
