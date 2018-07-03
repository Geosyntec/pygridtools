import os
from pkg_resources import resource_filename
import tempfile

import numpy
import pandas
from shapely import geometry
import geopandas

import pytest
import numpy.testing as nptest
import pandas.util.testing as pdtest

from pygridtools import gefdc
from . import utils


@pytest.mark.parametrize(('maxcols', 'knownfile'), [
    (150, resource_filename('pygridtools.tests.baseline_files', 'cell_basic.inp')),
    (5, resource_filename('pygridtools.tests.baseline_files', 'cell_chunked.inp'))
])
def test_write_cellinp(maxcols, knownfile):
    cells = numpy.array([
        [9, 9, 9, 9, 9, 9, 0, 0, 0],
        [9, 3, 5, 5, 2, 9, 9, 9, 9],
        [9, 5, 5, 5, 5, 5, 5, 5, 9],
        [9, 5, 5, 5, 5, 5, 5, 5, 9],
        [9, 4, 5, 5, 1, 9, 9, 9, 9],
        [9, 9, 9, 9, 9, 9, 0, 0, 0],
    ])
    with tempfile.TemporaryDirectory() as outputdir:
        outfile = os.path.join(outputdir, 'cell.inp')
        gefdc.write_cellinp(cells, outfile, maxcols=maxcols)
        utils.assert_textfiles_equal(knownfile, outfile)


def test_convert_gridext_to_gis(example_crs):
    gridextfile = resource_filename('pygridtools.tests.test_data', 'gridext.inp')
    baselinefile = resource_filename('pygridtools.tests.baseline_files', 'gridext.shp')
    river = 'test'
    reach = 1

    with tempfile.TemporaryDirectory() as outputdir:
        outputfile = os.path.join(outputdir, 'gridext.shp')
        gefdc.convert_gridext_to_gis(gridextfile, outputfile, example_crs, river=river)
        utils.assert_gis_files_equal(baselinefile, outputfile)


def test_write_gefdc_control_file():
    with tempfile.TemporaryDirectory() as outputdir:
        result_filename = os.path.join(outputdir, 'maingefdc.inp')
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'maingefdc.inp')
        gefdc.write_gefdc_control_file(result_filename, 'Test Input File', 100, 25, 0)
        utils.assert_textfiles_equal(known_filename, result_filename)


def test_write_gridext_file():
    with tempfile.TemporaryDirectory() as outputdir:
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'testgridext.inp')
        result_filename = os.path.join(outputdir, 'testgridext.inp')
        df = pandas.DataFrame(numpy.array([
            [1.25, 3, 4, 3.75],
            [1.75, 4, 4, 3.25],
            [1.25, 4, 5, 3.75],
            [1.75, 5, 5, 3.25],
        ]), columns=['x', 'ii', 'jj', 'y'])
        gefdc.write_gridext_file(df, result_filename, icol='ii', jcol='jj',
                                 xcol='x', ycol='y')
        utils.assert_textfiles_equal(known_filename, result_filename)


@pytest.mark.xfail
def test_write_gridout_file(simple_nodes):
    known_filename = resource_filename('pygridtools.tests.baseline_files', 'testgrid.out')
    x, y = simple_nodes

    with tempfile.TemporaryDirectory() as outdir:
        result_filename = os.path.join(outdir, 'testgrid.out')

        df = gefdc.write_gridout_file(x, y, result_filename)
        utils.assert_textfiles_equal(known_filename, result_filename)


def test_writer_control_file(mg):
    with tempfile.TemporaryDirectory() as result_path:
        writer = gefdc.GEFDCWriter(mg, result_path)
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'modelgrid_gefdc.inp')
        result_file = 'modelgrid_gefdc.inp'
        writer.control_file(
            filename=result_file,
            title='Model Grid Test'
        )
        utils.assert_textfiles_equal(
            known_filename,
            os.path.join(result_path, result_file)
        )


def test_writer_cell_file(mg):
    with tempfile.TemporaryDirectory() as result_path:
        writer = gefdc.GEFDCWriter(mg, result_path)
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'modelgrid_cell.inp')
        result_file = 'modelgrid_cell.inp'
        writer.cell_file(
            filename=result_file,
        )
        utils.assert_textfiles_equal(
            known_filename,
            os.path.join(result_path, result_file)
        )


@pytest.mark.xfail
def test_writer_gridout_file(mg):
    with tempfile.TemporaryDirectory() as result_path:
        writer = gefdc.GEFDCWriter(mg, result_path)
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'modelgrid_grid.out')
        result_file = 'modelgrid_grid.out'
        writer.gridout_file(filename=result_file)
        utils.assert_textfiles_equal(
            known_filename,
            os.path.join(result_path, result_file),
        )


def test_writer_gridext_file(mg):
    with tempfile.TemporaryDirectory() as result_path:
        writer = gefdc.GEFDCWriter(mg, result_path)
        known_filename = resource_filename('pygridtools.tests.baseline_files', 'modelgrid_gridext.inp')

        result_file = 'modelgrid_gridext.inp'
        writer.gridext_file(filename=result_file)
        utils.assert_textfiles_equal(
            known_filename,
            os.path.join(result_path, result_file),
        )
