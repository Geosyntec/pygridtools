import numpy as np
import matplotlib.pyplot as plt

from pygridtools import viz

import nose.tools as nt
import numpy.testing as nptest
from matplotlib.testing.decorators import image_comparison, cleanup
import pygridtools.testing as pgtest


class test_mpl_engine(object):
    def setup(self):
        pass

    def teardown(self):
        plt.close('all')

    def test_plotDomain_smoke(self):
        data = pgtest.makeSimpleBoundary()
        fig3 = viz.plotDomain(domain_x='x', domain_y='y', beta='beta', data=data)
        fig4 = viz.plotDomain(domain_x=data['x'], domain_y=data['y'], beta=data['beta'], data=None)


    def test_plotBoundaries_smoke(self):
        extent = pgtest.makeSimpleBoundary()
        islands = pgtest.makeSimpleIslands()
        fig5 = viz.plotBoundaries(extent_x='x', extent_y='y', extent=extent,
                                         islands_x='x', islands_y='y', islands_name='island',
                                         islands=islands)

        fig6 = viz.plotBoundaries(extent_x=extent['x'], extent_y=extent['y'], extent=None,
                                         islands_x=islands['x'], islands_y=islands['y'],
                                         islands_name=islands['island'], islands=None)

    def test_plotPoints_smoke(self):
        x, y = pgtest.makeSimpleNodes()
        fig1 = viz.plotPoints(x, y)

    def test_plotCells_smoke(self):
        x, y = pgtest.makeSimpleNodes()
        fig1 = viz.plotCells(x, y)
