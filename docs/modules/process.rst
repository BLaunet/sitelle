.. _sitelle-process:

*********************************
Process (`sitelle.process`)
*********************************

.. currentmodule:: sitelle.process

Introduction
============

This module overrides the SpectralCube methods of ORCS, to correct:

* the `integrate` function, that was not using the calibrated datacube
* the median extraction, that was not properly computed
* the parallel extraction, that can be slower than non-parallel when only a few spectra are extracted
* the possibility to extract non-integrated spectrum, if we want to analyse them separately without extracting them one by one


Reference/API
=============
.. automodapi:: sitelle.process
