`pygridtools`
=============

.. image:: docs/_static/logo.png


.. image:: https://travis-ci.org/Geosyntec/pygridtools.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/pygridtools

.. image:: https://codecov.io/gh/Geosyntec/pygridtools/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/Geosyntec/pygridtools

  .. image:: https://readthedocs.org/projects/pygridtools/badge/?version=latest
    :target: https://pygridtools.readthedocs.io/en/latest/?badge=latest


A high-level interface for curvilinear-orthogonal grid generation, manipulation, and visualization.

Depends heavily on `gridgen <https://github.com/sakov/gridgen-c>`_ and `pygridgen <https://pygridgen.github.io/pygridgen>`_

The full documentation for this for library is `here <https://pygridtools.readthedocs.io/>`_.

Installation and Depedencies
----------------------------
`conda-forge <https:/github.com/conda-forge>`_ generously maintains Linux and Mac OS X conda builds of *pygridtools*.

Install with

::

   conda install pygridtools --channel=conda-forge

Building (``gridgen-c``) on Windows has been a tough nut to crack and help is very much wanted in that department.
Until we figure that out, you can do the following in the source directory.

::

    conda create --name=grid python=3.6 numpy scipy pandas matplotlib shapely geopandas --channel=conda-forge
    pip install -e .

You won't be able to generate new grids, but you should be able to manipulate existing grids.

To create new grids on Linux or Mac OS, you'll need ``pygridgen``

::

    conda activate grid
    conda install pygridgen --channel=conda-forge

If you want to use the interactive ipywidgets to manipulate grid parameters, you'll need a few elements of the jupyter ecosystem

::

    conda activate grid
    conda install notebook ipywidgets --channel=conda-forge

If you'd like to build the docs, you need a few more things

::

    conda activate grid
    conda install sphinx numpydoc sphinx_rtd_theme nbsphinx --channel=conda-forge

Finally, to fully run the tests, you need ``pytest`` and a few plugins

::

    conda activate grid
    conda install pytest pytest-mpl pytest-pep8 --channel=conda-forge


Grid Generation
~~~~~~~~~~~~~~~

If you wish to generate new grids from scratch, you'll need `pygridgen <https://github.com/pygridgen/pygridgen>`_, which is also available through the conda-forge channel.

::

   conda install pygridgen --channel=conda-forge

The documentation `pygridgen` has a `more detailed tutorial <http://pygridgen.github.io/pygridgen/tutorial/basics.html>`_ on generating new grids.

Testing
~~~~~~~

Tests are written using the `pytest` package.
From the source tree, run them simply with by invoking ``pytest`` in a terminal.
If you're editing the source code, it helps to have `pytest-pep8` installed to check code style.

Alternatively, from the source tree you can run ``python check_pygridtools.py --strict`` to run the units tests, style checker, and doctests.

Documentation
~~~~~~~~~~~~~
Building the HTML documentation requires:

* sphinx
* sphinx_rtd_theme
* numpydoc
* jupyter-notebook
* nbsphinx
* pandas
* seaborn


Source code and Issue Tracker
------------------------------

The source code is available on Github at `Geosyntec/pygridtools <https://github.com/Geosyntec/pygridtools/>`_.

Please report bugs, issues, and ideas there.

Contributing
------------
1. Feedback is a huge contribution
2. Get in touch by creating an issue to make sure we don't duplicate work
3. Fork this repo
4. Submit a PR in a separate branch
5. Write a test (or two (or three))
6. Stick to PEP8-ish -- I'm lenient on the 80 chars thing (<100 is probably a smart move though).
