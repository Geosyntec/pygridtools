`pygridtools`
=============
.. image:: http://i.imgur.com/2SeSNsR.png

.. image:: https://travis-ci.org/Geosyntec/pygridtools.svg?branch=master
    :target: https://travis-ci.org/Geosyntec/pygridtools
.. image:: https://coveralls.io/repos/Geosyntec/pygridtools/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/Geosyntec/pygridtools?branch=master


A high-level interface for curvilinear-orthogonal grid generation, manipulation, and visualization.

Depends heavily on `gridgen`_ and `pygridgen`_

.. _gridgen: https://github.com/sakov/gridgen-c
.. _pygridgen: https://Geosyntec.github.io/pygridgen

The full documentation for this for library is `here`_.

.. _here: https://Geosyntec.github.io/pygridtools


Installation
------------
`IOOS <https:/github.com/IOOS>`_ generously maintains Linux and Mac OS X conda builds of *pygridtools*.

Install with

::

   conda install --channel=IOOS pygridtools


Python Dependencies
-------------------

Basics
~~~~~~

Provided that all of the shared C libraries are installed, the remaining python depedencies are the following:

* numpy
* matplotlib, seaborn
* pyproj (only if working with geographic coordinates)
* fiona (for shapfile I/O)
* pandas (for easy data I/O and manipulation)

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
