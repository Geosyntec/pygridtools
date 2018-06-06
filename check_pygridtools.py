import sys
import matplotlib
from matplotlib import style

import pygridtools

matplotlib.use('agg')
style.use('classic')

if '--strict' in sys.argv:
    sys.argv.remove('--strict')
    status = pygridtools.teststrict(*sys.argv[1:])
else:
    status = pygridtools.test(*sys.argv[1:])

sys.exit(status)