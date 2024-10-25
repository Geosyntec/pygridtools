from importlib import resources

import pygridtools
from pygridgen.tests import requires

try:
    import pytest
except ImportError:
    pytest = None


@requires(pytest, 'pytest')
def test(*args):
    options = [resources('pygridtools', '')]
    options.extend(list(args))
    return pytest.main(options)


@requires(pytest, 'pytest')
def teststrict(*args):
    options = list(set([
        resources('pygridtools', ''),
        '--mpl',
        '--doctest-modules'
    ] + list(args)))
    return pytest.main(options)
