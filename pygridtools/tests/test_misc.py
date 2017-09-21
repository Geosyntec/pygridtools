import numpy as np
from numpy import nan

import pytest
import numpy.testing as nptest

from pygridtools import misc


np.set_printoptions(linewidth=150, nanstr='-')

try:
    import pygridgen
    HASPGG = True
except ImportError:
    HASPGG = False


@pytest.mark.parametrize(('masked', 'z', 'triangles'), [
    (None, None, False),
    (False, None, False),
    (None, 5, False),
    (None, None, True),
])
def test_make_poly_coords_base(masked, z, triangles):
    xarr = np.array([[1, 2], [1, 2]], dtype=float)
    yarr = np.array([[3, 3], [4, 4]], dtype=float)
    if masked is False:
        xarr = np.ma.masked_array(xarr, mask=False)
        yarr = np.ma.masked_array(yarr, mask=False)

    if z:
        expected = np.array([[1, 3, z], [2, 3, z], [2, 4, z], [1, 4, z]], dtype=float)
    elif triangles:
        expected = np.array([[1, 3], [2, 4], [1, 4]], dtype=float)
        xarr[0, -1] = np.nan
        yarr[0, -1] = np.nan
    else:
        expected = np.array([[1, 3], [2, 3], [2, 4], [1, 4]], dtype=float)

    coords = misc.make_poly_coords(xarr, yarr, zpnt=z, triangles=triangles)
    nptest.assert_array_equal(coords, expected)


@pytest.mark.parametrize('as_array', [True, False])
@pytest.mark.parametrize(('geom', 'geomtype', 'error'), [
    ([1, 2], 'Point', False),
    ([[1, 2], [5, 6], [5, 2]], 'LineString', False),
    ([[1, 2], [5, 6], [5, 2]], 'Polygon', False),
    ([[1, 2], [5, 6], [5, 2]], 'Circle', True),

])
def test_make_record(geom, geomtype, error, as_array):
    props = {'prop1': 'this string', 'prop2': 3.1415}
    expected_geoms = {
        'point': {
            'geometry': {
                'type': 'Point',
                'coordinates': [1, 2]
            },
            'id': 1,
            'properties': props
        },
        'linestring': {
            'geometry': {
                'type': 'LineString',
                'coordinates': [[[1, 2], [5, 6], [5, 2]]]
            },
            'id': 1,
            'properties': props
        },
        'polygon': {
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[1, 2], [5, 6], [5, 2]]]
            },
            'id': 1,
            'properties': props
        }
    }

    if as_array:
        geom = np.array(geom)

    if error:
        with pytest.raises(ValueError):
            misc.make_record(1, geom, geomtype, props)
    else:
        record = misc.make_record(1, geom, geomtype, props)
        assert record == expected_geoms[geomtype.lower()]


@pytest.mark.skipif(not HASPGG, reason='pygridgen unavailabile')
def test_interpolate_bathymetry(simple_bathy, simple_grid):
    elev1 = misc.interpolate_bathymetry(None, simple_grid.x_rho, simple_grid.y_rho)
    elev2 = misc.interpolate_bathymetry(simple_bathy, simple_grid.x_rho, simple_grid.y_rho)

    fake_elev = np.ma.MaskedArray(data=np.zeros(simple_grid.x_rho.shape), mask=simple_grid.x_rho.mask)
    real_elev = np.ma.masked_invalid(np.array([
        [100.15, 100.20,    nan,    nan,    nan,    nan],
        [100.20, 100.25, 100.65, 100.74, 100.83, 100.95],
        [100.25, 100.30, 100.35, 100.40, 100.45, 100.50],
        [100.30, 100.35, 100.40, 100.45, 100.50, 100.55],
        [100.35, 100.40,    nan,    nan,    nan,    nan],
        [100.40, 100.45,    nan,    nan,    nan,    nan],
        [100.45, 100.50,    nan,    nan,    nan,    nan],
        [100.50, 100.55,    nan,    nan,    nan,    nan]
    ]))

    nptest.assert_array_equal(elev1, fake_elev)
    assert (elev1.shape == simple_grid.x_rho.shape)
    nptest.assert_array_almost_equal(elev2, real_elev, decimal=2)


@pytest.fixture
def stackgrids():
    grids = {
        'input': {
            'g0': np.array([
                [13.7, 13.8],
                [14.7, 14.8],
                [15.7, 15.8],
                [16.7, 16.8],
                [17.7, 17.8],
            ]),
            'g1': np.array([
                [6.6, 6.7, 6.8],
                [7.6, 7.7, 7.8],
                [8.6, 8.7, 8.8],
                [9.6, 9.7, 9.8],
                [10.6, 10.7, 10.8],
                [11.6, 11.7, 11.8],
                [12.6, 12.7, 12.8],
            ]),
            'g2': np.array([
                [7.9, 7.10, 7.11, 7.12, 7.13],
                [8.9, 8.10, 8.11, 8.12, 8.13],
                [9.9, 9.10, 9.11, 9.12, 9.13],
            ]),
            'g3': np.array([
                [1.4, 1.5, 1.6, 1.7, 1.8],
                [2.4, 2.5, 2.6, 2.7, 2.8],
                [3.4, 3.5, 3.6, 3.7, 3.8],
                [4.4, 4.5, 4.6, 4.7, 4.8],
                [5.4, 5.5, 5.6, 5.7, 5.8],
            ]),
            'g4': np.array([
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
            ]),
            'g5': np.array([
                [7.14, 7.15, 7.16],
                [8.14, 8.15, 8.16],
            ])
        },
        'output': {
            'g1-g2Left': np.array([
                [nan,  nan,  nan,  nan,  nan,  6.6,  6.7,  6.8],
                [7.9, 7.10, 7.11, 7.12, 7.13,  7.6,  7.7,  7.8],
                [8.9, 8.10, 8.11, 8.12, 8.13,  8.6,  8.7,  8.8],
                [9.9, 9.10, 9.11, 9.12, 9.13,  9.6,  9.7,  9.8],
                [nan,  nan,  nan,  nan,  nan, 10.6, 10.7, 10.8],
                [nan,  nan,  nan,  nan,  nan, 11.6, 11.7, 11.8],
                [nan,  nan,  nan,  nan,  nan, 12.6, 12.7, 12.8],
            ]),
            'g1-g2Right': np.array([
                [6.6, 6.7, 6.8, nan, nan, nan, nan, nan],
                [7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 7.13],
                [8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13],
                [9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13],
                [10.6, 10.7, 10.8, nan, nan, nan, nan, nan],
                [11.6, 11.7, 11.8, nan, nan, nan, nan, nan],
                [12.6, 12.7, 12.8, nan, nan, nan, nan, nan],
            ]),
            'g0-g1': np.array([
                [nan, 6.6, 6.7, 6.8],
                [nan, 7.6, 7.7, 7.8],
                [nan, 8.6, 8.7, 8.8],
                [nan, 9.6, 9.7, 9.8],
                [nan, 10.6, 10.7, 10.8],
                [nan, 11.6, 11.7, 11.8],
                [nan, 12.6, 12.7, 12.8],
                [13.7, 13.8, nan,  nan],
                [14.7, 14.8, nan,  nan],
                [15.7, 15.8, nan,  nan],
                [16.7, 16.8, nan,  nan],
                [17.7, 17.8, nan,  nan],
            ]),
            'g0-g1-g2': np.array([
                [6.6,  6.7,  6.8, nan, nan, nan, nan, nan],
                [7.6,  7.7,  7.8, 7.9, 7.10, 7.11, 7.12, 7.13],
                [8.6,  8.7,  8.8, 8.9, 8.10, 8.11, 8.12, 8.13],
                [9.6,  9.7,  9.8, 9.9, 9.10, 9.11, 9.12, 9.13],
                [10.6, 10.7, 10.8, nan, nan, nan, nan, nan],
                [11.6, 11.7, 11.8, nan, nan, nan, nan, nan],
                [12.6, 12.7, 12.8, nan, nan, nan, nan, nan],
                [nan, 13.7, 13.8, nan, nan, nan, nan, nan],
                [nan, 14.7, 14.8, nan, nan, nan, nan, nan],
                [nan, 15.7, 15.8, nan, nan, nan, nan, nan],
                [nan, 16.7, 16.8, nan, nan, nan, nan, nan],
                [nan, 17.7, 17.8, nan, nan, nan, nan, nan],
            ]),
            'g1-g3': np.array([
                [nan, nan, 1.4, 1.5, 1.6, 1.7, 1.8],
                [nan, nan, 2.4, 2.5, 2.6, 2.7, 2.8],
                [nan, nan, 3.4, 3.5, 3.6, 3.7, 3.8],
                [nan, nan, 4.4, 4.5, 4.6, 4.7, 4.8],
                [nan, nan, 5.4, 5.5, 5.6, 5.7, 5.8],
                [6.6, 6.7, 6.8, nan, nan, nan, nan],
                [7.6, 7.7, 7.8, nan, nan, nan, nan],
                [8.6, 8.7, 8.8, nan, nan, nan, nan],
                [9.6, 9.7, 9.8, nan, nan, nan, nan],
                [10.6, 10.7, 10.8, nan, nan, nan, nan],
                [11.6, 11.7, 11.8, nan, nan, nan, nan],
                [12.6, 12.7, 12.8, nan, nan, nan, nan],
            ]),
            'g3-g4': np.array([
                [0.0, 0.1, 0.2, 0.3, nan, nan, nan, nan, nan],
                [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
                [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
                [nan, nan, nan, nan, 4.4, 4.5, 4.6, 4.7, 4.8],
                [nan, nan, nan, nan, 5.4, 5.5, 5.6, 5.7, 5.8],
            ]),
            'g-all': np.array([
                [0.0, 0.1, 0.2, 0.3, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                [1.0, 1.1, 1.2, 1.3, 1.4, 1.5,  1.6,  1.7,  1.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [2.0, 2.1, 2.2, 2.3, 2.4, 2.5,  2.6,  2.7,  2.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [3.0, 3.1, 3.2, 3.3, 3.4, 3.5,  3.6,  3.7,  3.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 4.4, 4.5,  4.6,  4.7,  4.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, 5.4, 5.5,  5.6,  5.7,  5.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan,  6.6,  6.7,  6.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan,  7.6,  7.7,  7.8, 7.9, 7.10, 7.11, 7.12, 7.13, 7.14, 7.15, 7.16],
                [nan, nan, nan, nan, nan, nan,  8.6,  8.7,  8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16],
                [nan, nan, nan, nan, nan, nan,  9.6,  9.7,  9.8, 9.9, 9.10, 9.11, 9.12, 9.13, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, 10.6, 10.7, 10.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, 11.6, 11.7, 11.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, 12.6, 12.7, 12.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, 13.7, 13.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, 14.7, 14.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, 15.7, 15.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, 16.7, 16.8, nan, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan, nan, 17.7, 17.8, nan, nan, nan, nan, nan, nan, nan, nan],
            ])
        }
    }

    return grids


@pytest.mark.parametrize('idx1, idx2, how, where, shift, expected', [
    ('g1', 'g3', 'v', '-', 2, 'g1-g3'),
    ('g3', 'g1', 'v', '+', -2, 'g1-g3'),
    ('g1', 'g2', 'h', '-', 1, 'g1-g2Left'),
    ('g1', 'g2', 'h', '+', 1, 'g1-g2Right'),
    ('g0', 'g1', 'v', '-', 1, 'g0-g1'),
], ids=['VA-', 'VB+', 'HL-', 'HR+', 'V-easy'])
def test_padded_stack_pairs(stackgrids, idx1, idx2, how, where, shift, expected):
    result = misc.padded_stack(
        stackgrids['input'][idx1],
        stackgrids['input'][idx2],
        how=how,
        where=where,
        shift=shift
    )
    nptest.assert_array_equal(result, stackgrids['output'][expected])


def test_padded_stack_three(stackgrids):
    step1 = misc.padded_stack(stackgrids['input']['g0'], stackgrids['input']['g1'],
                              how='v', where='-', shift=-1)
    step2 = misc.padded_stack(step1, stackgrids['input']['g2'],
                              how='h', where='+', shift=1)
    nptest.assert_array_equal(step2, stackgrids['output']['g0-g1-g2'])


def test_padded_stack_a_bunch(stackgrids):
    step1 = misc.padded_stack(stackgrids['input']['g0'], stackgrids['input']['g1'],
                              how='v', where='-', shift=-1)
    step2 = misc.padded_stack(step1, stackgrids['input']['g2'],
                              how='h', where='+', shift=1)
    step3 = misc.padded_stack(step2, stackgrids['input']['g3'],
                              how='v', where='-', shift=-2)
    step4 = misc.padded_stack(step3, stackgrids['input']['g4'],
                              how='h', where='-', shift=-1)
    step5 = misc.padded_stack(step4, stackgrids['input']['g5'],
                              how='h', where='+', shift=7)
    nptest.assert_array_equal(step5, stackgrids['output']['g-all'])


@pytest.mark.parametrize(('how', 'where'), [('junk', '+'), ('h', 'junk')])
def test_padded_stack_errors(stackgrids, how, where):
    with pytest.raises(ValueError):
        misc.padded_stack(stackgrids['input']['g1'], stackgrids['input']['g3'],
                          how=how, where=where, shift=2)


@pytest.mark.parametrize(('inside', 'expected'), [
    (True, np.array([
        [0, 0, 0, 0, 0], [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0], [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=bool)),
    (False, np.array([
        [1, 1, 1, 1, 1], [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1], [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]], dtype=bool))
], ids=['inside', 'outside'])
def test_mask_with_polygon(inside, expected):
    y, x = np.mgrid[:5, :5]
    polyverts = [
        (0.5, 2.5),
        (3.5, 2.5),
        (3.5, 0.5),
        (0.5, 0.5),
    ]
    mask = misc.mask_with_polygon(x, y, polyverts, inside=inside)
    nptest.assert_array_equal(mask, expected)


@pytest.mark.parametrize(('nodes', 'mask', 'triangles', 'expected'), [
    (
        np.array([
            [0, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 0],
        ]),
        np.zeros((4, 7)),
        True,
        np.array([
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
            [9, 3, 5, 5, 2, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 5, 5, 9],
            [9, 4, 5, 5, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 0, 0, 0],
        ])
    ),
    (
        np.ones((6, 6)),
        np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        False,
        np.array([
            [0, 0, 9, 9, 9, 9, 9],
            [0, 0, 9, 5, 5, 5, 9],
            [9, 9, 9, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 9, 9, 9, 9, 9, 9]
        ])
    ),
    (
        np.ones((6, 6)),
        np.zeros((5, 5)),
        False,
        np.array([
            [9, 9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 5, 5, 5, 5, 5, 9],
            [9, 9, 9, 9, 9, 9, 9]
        ])
    ),
    (
        np.array([
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        ]),
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        False,
        np.array([
            [0, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
            [0, 9, 5, 5, 5, 5, 5, 9, 0, 0, 0, 0],
            [0, 9, 9, 5, 5, 5, 9, 9, 0, 0, 0, 0],
            [0, 0, 9, 5, 5, 5, 9, 0, 0, 0, 0, 0],
            [0, 0, 9, 9, 5, 9, 9, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 5, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 5, 9, 9, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 9, 5, 9, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 0, 0, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 9, 9, 9, 5, 5, 5, 5, 5, 9],
            [0, 0, 0, 9, 5, 5, 5, 9, 9, 9, 9, 9],
            [0, 0, 0, 9, 5, 5, 5, 9, 0, 0, 0, 0],
            [0, 0, 0, 9, 5, 5, 5, 9, 0, 0, 0, 0],
            [0, 0, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0]
        ])
    ),
], ids=['with triangles', 'without triangles', 'simple no mask', 'complex no mask'])
def test_make_gefdc_cells(nodes, mask, triangles, expected):
    cells = misc.make_gefdc_cells(nodes, cell_mask=mask, triangles=triangles)
    nptest.assert_array_equal(cells, expected)
