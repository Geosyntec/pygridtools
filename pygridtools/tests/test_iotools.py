import os
import sys
from pkg_resources import resource_filename
import tempfile

import numpy
import pandas
from shapely import geometry
import geopandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

try:
    import ipywidgets
except ImportError:
    ipywidgets = None

from pygridtools import iotools
import pygridgen
from pygridgen.tests.utils import raises, requires
from . import utils


@pytest.mark.parametrize(('filterfxn', 'points_in_boundary'), [
    (None, 19),
])
def test_read_boundary(filterfxn, points_in_boundary):
    gisfile = resource_filename('pygridtools.tests.test_data', 'simple_boundary.shp')
    df = iotools.read_boundary(gisfile, filterfxn=filterfxn)

    df_columns = ['x', 'y', 'beta', 'upperleft', 'reach', 'order', 'geometry']
    assert (isinstance(df, geopandas.GeoDataFrame))
    assert (df.columns.tolist() == df_columns)
    assert (df.shape[0] == points_in_boundary)


@pytest.mark.parametrize('as_gdf', [False, True])
def test_read_polygons(as_gdf):
    gisfile = resource_filename('pygridtools.tests.test_data', 'simple_islands.shp')
    known_islands = [
        numpy.array([
            [10.18915802,  3.71280277], [9.34025375,  7.21914648],
            [9.34025375,  7.21914648], [10.15224913, 11.98039216],
            [14.13840830, 11.83275663], [17.68166090,  7.47750865],
            [15.94694348,  3.63898501], [10.18915802,  3.71280277]
        ]),
        numpy.array([
            [1.10957324,  3.67589389], [-0.95732411,  7.69896194],
            [-0.95732411,  7.69896194], [0.81430219, 10.91003460],
            [5.98154556, 10.98385236], [8.67589389,  7.03460208],
            [6.86735871,  3.71280277], [1.10957324,  3.67589389]
        ]),
    ]
    islands = iotools.read_polygons(gisfile, as_gdf=as_gdf)
    if as_gdf:
        expected = geopandas.GeoDataFrame({
            'id': [2, 1],
            'name': ['loser', 'keeper']
        }, geometry=list(map(geometry.Polygon, known_islands)))
        pdtest.assert_frame_equal(
            islands.drop('geometry', axis='columns'),
            expected.drop('geometry', axis='columns')
        )
        assert islands.geom_almost_equals(expected).all()
    else:
        for res, exp in zip(islands, known_islands):
            nptest.assert_array_almost_equal(res, exp)


def test_read_grid():
    pntfile = resource_filename('pygridtools.tests.baseline_files', 'array_point.shp')
    cellfile = resource_filename('pygridtools.tests.baseline_files', 'array_grid.shp')
    result_df = iotools.read_grid(pntfile, othercols=['elev', 'river'])
    known_df = pandas.DataFrame(
        data={'easting': [1., 2., 3.] * 4, 'northing': sorted([4., 5., 6., 7.] * 3)},
        index=pandas.MultiIndex.from_product([[2, 3, 4, 5], [2, 3, 4]], names=['jj', 'ii'])
    ).assign(elev=0.0).assign(river='test').reset_index().set_index(['ii', 'jj']).sort_index()

    pdtest.assert_frame_equal(result_df, known_df)

    with raises(NotImplementedError):
        result_df = iotools.read_grid(cellfile)


def test__change_shape(simple_grid):
    xn = iotools._change_shape(simple_grid, 18, 12, lambda x, y: x)
    assert xn.shape == (18, 12)


@requires(ipywidgets, 'ipywidgets')
@requires(pygridgen, 'pygridgen')
def test_interactive_grid_shape(simple_grid):
    newgrid, widget = iotools.interactive_grid_shape(simple_grid, max_n=100)
    assert isinstance(newgrid, pygridgen.grid.Gridgen)
    assert isinstance(widget, ipywidgets.interactive)
    assert isinstance(widget.children[0], ipywidgets.IntSlider)
    assert isinstance(widget.children[1], ipywidgets.IntSlider)
    assert widget.children[0].max == widget.children[1].max == 100


@pytest.mark.parametrize(('attrname', 'attrtype'), [
    ('pos', float),
    ('axis', str),
    ('factor', float),
    ('extent', float),
    ('focuspoint', pygridgen.grid._FocusPoint)])
def test_fp_attributes(focus_properties, attrname, attrtype):
    attr = getattr(focus_properties, attrname)  # effectively asserts hasattr
    assert isinstance(attr, attrtype)


@pytest.mark.skip(reason="Plotting wrapper")
def test__plot_focus_point():
    pass


def test__change_focus(simple_grid):
    old_axis = 'x'
    old_pos = 0.5
    old_factor = 1
    old_extent = 0.5

    new_axis = 'y'
    new_pos = 0.9
    new_factor = 2
    new_extent = 0.1

    fp = iotools._FocusProperties(
        pos=old_pos, axis=old_axis,factor=old_factor, extent=old_extent)
    others = (iotools._FocusProperties(
        pos=old_pos, axis=old_axis, factor=old_factor, extent=old_extent),)*3

    xn = iotools._change_focus(fp, others, new_axis, new_pos,
        new_factor, new_extent, simple_grid, lambda x, y: x)

    # test single focus modification
    assert fp.axis == new_axis
    assert fp.pos == new_pos
    assert fp.factor == new_factor
    assert fp.extent == new_extent

    # test others are not modified
    for o in others:
        assert o.axis == old_axis
        assert o.pos == old_pos
        assert o.factor == old_factor
        assert o.extent == old_extent


@requires(ipywidgets, 'ipywidgets')
@requires(pygridgen, 'pygridgen')
@pytest.mark.parametrize('tab', [0, 1])
def test_interactive_grid_focus_tabs(simple_grid, tab):
    focus_points, widget = iotools.interactive_grid_focus(simple_grid, n_points=2)
    assert isinstance(focus_points[tab], iotools._FocusProperties)
    assert isinstance(widget, ipywidgets.widgets.widget_selectioncontainer.Tab)


@requires(ipywidgets, 'ipywidgets')
@requires(pygridgen, 'pygridgen')
@pytest.mark.parametrize('parent', [0, 1])
@pytest.mark.parametrize(('child', 'widget_type'), [
    (0, ipywidgets.widgets.widget_selection.ToggleButtons),
    (1, ipywidgets.widgets.widget_float.FloatSlider),
    (2, ipywidgets.widgets.widget_float.FloatLogSlider),
    (3, ipywidgets.widgets.widget_float.FloatSlider)])
def test_interactive_grid_focus_tabs(simple_grid, parent, child, widget_type):
    focus_points, widget = iotools.interactive_grid_focus(simple_grid, n_points=2)
    assert isinstance(widget.children[parent], ipywidgets.widgets.interaction.interactive)
    assert isinstance(widget.children[parent].children[child], widget_type)
    # todo: need to figure ranges after some irl testing to then test min and max:
        # assert widget.children[1].children[0].min == 0.01
        # assert widget.children[1].children[0].max == 1