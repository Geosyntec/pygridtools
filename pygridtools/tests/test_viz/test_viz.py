import numpy as np
import matplotlib.pyplot as plt

from pygridtools import viz

import pytest
import numpy.testing as nptest
import pygridtools.testing as pgtest


@pytest.mark.parametrize('engine', ['mpl'])
def test_plot_domain_smoke(simple_boundary, engine):
    fig3 = viz.plotDomain(domain_x='x', domain_y='y', beta='beta', data=simple_boundary)
    fig4 = viz.plotDomain(domain_x=simple_boundary['x'], domain_y=simple_boundary['y'],
                          beta=simple_boundary['beta'], data=None)


@pytest.mark.parametrize('engine', ['mpl'])
def test_plotBoundaries_smoke(simple_boundary, simple_islands, engine):
    fig5 = viz.plotBoundaries(extent_x='x', extent_y='y', extent=simple_boundary,
                              islands_x='x', islands_y='y', islands_name='island',
                              islands=simple_islands)

    fig6 = viz.plotBoundaries(extent_x=simple_boundary['x'], extent_y=simple_boundary['y'], extent=None,
                              islands_x=simple_islands['x'], islands_y=simple_islands['y'],
                              islands_name=simple_islands['island'], islands=None)


def test_plotPoints_smoke(simple_nodes):
    fig1 = viz.plotPoints(*simple_nodes)


def test_plotCells_smoke(simple_nodes):
    fig1 = viz.plotCells(*simple_nodes)
