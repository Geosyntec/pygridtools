`pygridtools`
=============

.. image:: http://i.imgur.com/JWdgAKk.png



.. image:: https://travis-ci.org/Geosyntec/pygridtools.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/pygridtools
.. image:: https://coveralls.io/repos/Geosyntec/pygridtools/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/Geosyntec/pygridtools?branch=master

A high-level interface for curvilinear-orthogonal grid generation, manipulation, and visualization.

Depends heavily on `gridgen <https://github.com/sakov/gridgen-c>`_ and Paul Hobson's fork of `pygridgen <https://phobson.github.io/pygridgen>`_

The full documentation for this for library is `here <https://Geosyntec.github.io/pygridtools>`_.

Installation
------------
`IOOS <https:/github.com/IOOS>`_ generously maintains Linux and Mac OS X conda builds of *pygridtools*.

Install with

::

   conda install --channel=IOOS pygridtools
   
Building (``gridgen-c``) on Windows has been a tough nut to crack and help is very much wanted in that department.
Until we figure that out, you can do the following in the source directory.

::

    conda install seaborn fiona=1.5
    pip install -e .

Other Python Dependencies
-------------------------

Basics
~~~~~~

The remaining python depedencies are the following:

* numpy
* matplotlib, seaborn
* pyproj (only if working with geographic coordinates)
* fiona (for shapfile I/O)
* pandas (for easy data I/O and manipulation)

Grid Generation
~~~~~~~~~~~~~~~

If you wish to generate new grids from scratch, you'll need `pygridgen <https://github.com/phobson/pygridgen>`_, which is also available through the IOOS conda channel.

::

   conda install --channel=IOOS pygridgen
   
The documentation `pygridgen` has a `more detailed tutorial <http://phobson.github.io/pygridgen/tutorial/basics.html>`_ on generating new grids.

Testing
~~~~~~~

Tests are written using the `nose` package.
From the source tree, run them simply with by invoking ``nosetests`` in a terminal.


Source code and Issue Tracker
------------------------------

The source code is available on Github at `Geosyntec/pygridtools <https://github.com/Geosyntec/pygridtools/>`_.

Please report bugs, issues, and ideas there.

Contributing
------------
1. Feedback is a huge contribution
2. Get in touch by creating an issue to make sure we don't duplicate work
3. Fork this repo
4. Submit a PR in a seperate branch
5. Write a test (or two (or three))
6. Stick to PEP8-ish -- I'm lenient on the 80 chars thing (<100 is probably a smart move though).
