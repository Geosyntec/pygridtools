from pkg_resources import resource_filename

import pygridtools
from .helpers import requires


try:
    import pytest
except ImportError:
    pytest = None


@requires(pytest, 'pytest')
def test(*args):
    options = [resource_filename('pygridtools', '')]
    options.extend(list(args))
    return pytest.main(options)


@requires(pytest, 'pytest')
def teststrict():
    options = [resource_filename('pygridtools', ''), '--pep', '--mpl']
    return pytest.main(options)
