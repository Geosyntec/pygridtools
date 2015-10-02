from pygridtools.viz import _viz_bokeh

import nose.tools as nt
from matplotlib.testing.decorators import image_comparison, cleanup
import pygridtools.testing as pgtest


@nt.raises(NotImplementedError)
def test__plot_domain():
    data = pgtest.makeSimpleBoundary()
    fig1 = _viz_bokeh._plot_domain(x='x', y='y', data=data)
    fig2 = _viz_bokeh._plot_domain(x=data['x'], y=data['y'], data=None)
    fig3 = _viz_bokeh._plot_domain(x='x', y='y', beta='beta', data=data)
    fig4 = _viz_bokeh._plot_domain(x=data['x'], y=data['y'], beta=data['beta'], data=None)



@nt.raises(NotImplementedError)
def test__plot_boundaries():
    model = pgtest.makeSimpleBoundary()
    islands = pgtest.makeSimpleIslands()
    fig1 = _viz_bokeh._plot_boundaries(model_x='x', model_y='y', model=model)
    fig2 = _viz_bokeh._plot_boundaries(model_x=model['x'], model_y=model['y'], model=None)
    fig3 = _viz_bokeh._plot_boundaries(island_x='x', island_y='y', island_name='island', islands=islands)
    fig4 = _viz_bokeh._plot_boundaries(island_x=islands['x'], island_y=islands['y'],
                                       island_name=islands['island'], islands=None)

    fig5 = _viz_bokeh._plot_boundaries(model_x='x', model_y='y', model=model,
                                       island_x='x', island_y='y', island_name='island',
                                       islands=islands)

    fig6 = _viz_bokeh._plot_boundaries(model_x=model['x'], model_y=model['y'], model=None,
                                       island_x=islands['x'], island_y=islands['y'],
                                       island_name=islands['island'], islands=None)

@nt.raises(NotImplementedError)
def test__plot_points():
    x, y = pgtest.makeSimpleNodes()
    fig1 = _viz_bokeh._plot_points(x, y)


@nt.raises(NotImplementedError)
def test__plot_cells():
    x, y = pgtest.makeSimpleNodes()
    fig1 = _viz_bokeh._plot_cells(x, y)
