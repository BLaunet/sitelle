.. sitelle documentation master file, created by
   sphinx-quickstart on Wed Jan 17 09:34:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:tocdepth: 3

*****************************************
Welcome to sitelle package documentation!
*****************************************

This package is a helper module designed for the analysis of M31 datacubes taken by SITELLE instrument.

It is mainly a prototype and has been developed for the specific reduction and analysis of Andromeda observations in the SN2 and SN3 filters.

.. warning::
   Even if the development is as generic as possible, some functions are specifically coded for M31 in the SN2 and SN3 filters and **should not be used as is for other observations**.

The package heavily relies on the `ORCS <http://132.203.11.199/orcs-doc/index.html>`_
and `ORB <http://132.203.11.199/orb-doc/index.html>`_ packages, and extends them in some way.

Most notable points are :
   * Algorithm of source detection and fitting
   * Interfacing with `Nburst method <https://arxiv.org/abs/0709.3047>`_ for galaxy continuum fitting. See also `examples <nburst_example.ipynb>`_

Installation
------------

The lastest version of the package is hosted on `pypi <https://pypi.org>`_ and can be installed with::

   pip install sitelle

Sitelle has the following strict requirements:

* `Python <http://www.python.org/>`_ 2.7
* `ORCS <http://132.203.11.199/orcs-doc/index.html>`_
* `ORB <http://132.203.11.199/orb-doc/index.html>`_
* `Numpy <http://www.numpy.org/>`_

Some specific tasks also rely on:

* `Astropy <http://astropy.org>`_
* `Photutils <http://photutils.readthedocs.io/en/stable/>`_
* `Matplotlib <http://matplotlib.org/>`_
* `Pandas <http://pandas.pydata.org>`_
* `Nburst <https://arxiv.org/abs/0709.3047>`_

User documentation
------------------
.. toctree::
   :maxdepth: 1

   tips-tricks.ipynb
   sources_example.ipynb
   nburst_example.ipynb



Modules specific documentation
------------------------------

.. toctree::
   :maxdepth: 1

   modules/nburst
   modules/process
   modules/plot
   modules/parallel
   modules/region
   modules/source
   modules/utils
   modules/fit
   modules/constants
   modules/calibration

Publications
------------

This package has been used in the following works :
   * :download:`Master Thesis <_static/Master_Thesis_BLaunet.pdf>` of Barthélémy Launet, ETH Zürich
   * :download:`Poster <_static/EWASS_poster.pdf>` presentend at EWASS 2018 in Liverpool, Barthélémy Launet & Anne-Laure Melchior
