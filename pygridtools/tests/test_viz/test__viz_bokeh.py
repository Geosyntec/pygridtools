from pygridtools.viz import _viz_bokeh

from pygridgen.tests import raises


def test__plot_domain(simple_boundary):
    with raises(NotImplementedError):
        fig1 = _viz_bokeh._plot_domain(x='x', y='y', data=simple_boundary)  # noqa: F841
        fig2 = _viz_bokeh._plot_domain(x=simple_boundary['x'], y=simple_boundary['y'], data=None)  # noqa: F841
        fig3 = _viz_bokeh._plot_domain(x='x', y='y', beta='beta', data=simple_boundary)  # noqa: F841
        fig4 = _viz_bokeh._plot_domain(x=simple_boundary['x'],  # noqa# F841
                                       y=simple_boundary['y'],
                                       beta=simple_boundary['beta'],
                                       data=None)


def test__plot_boundaries(simple_boundary, simple_islands):
    with raises(NotImplementedError):
        fig1 = _viz_bokeh._plot_boundaries(model_x='x', model_y='y', model=simple_boundary)  # noqa: F841
        fig2 = _viz_bokeh._plot_boundaries(model_x=simple_boundary['x'],  # noqa: F841
                                           model_y=simple_boundary['y'],
                                           model=None)
        fig3 = _viz_bokeh._plot_boundaries(island_x='x', island_y='y', island_name='island',  # noqa: F841
                                           islands=simple_islands)
        fig4 = _viz_bokeh._plot_boundaries(island_x=simple_islands['x'], island_y=simple_islands['y'],  # noqa: F841
                                           island_name=simple_islands['island'], islands=None)

        fig5 = _viz_bokeh._plot_boundaries(model_x='x', model_y='y', model=simple_boundary,  # noqa: F841
                                           island_x='x', island_y='y', island_name='island',
                                           islands=simple_islands)

        fig6 = _viz_bokeh._plot_boundaries(model_x=simple_boundary['x'],  # noqa: F841
                                           model_y=simple_boundary['y'],
                                           model=None,
                                           island_x=simple_islands['x'],
                                           island_y=simple_islands['y'],
                                           island_name=simple_islands['island'],
                                           islands=None)


def test__plot_points(simple_nodes):
    with raises(NotImplementedError):
        fig1 = _viz_bokeh._plot_points(*simple_nodes)  # noqa: F841


def test__plot_cells(simple_nodes):
    with raises(NotImplementedError):
        fig1 = _viz_bokeh._plot_cells(*simple_nodes)  # noqa: F841
