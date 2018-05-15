.. _sitelle-parallel:

*********************************
Parallel (`sitelle.parallel`)
*********************************

.. currentmodule:: sitelle.parallel

Introduction
============

This module contains helper methods for parallelization of process.
Because we mostly work on :class:`~numpy:numpy.ndarray` and :class:`~pandas:pandas.DataFrame`, we build three main parallelized methods:

* `parallel_apply_along_axis` that mimicks :func:`numpy:numpy.apply_along_axis` signature
* `parallel_apply_over_frame` to iterate over 2D frames of the datacubes (analyse images)
* `parallel_apply_over_df` that mimicks :func:`pandas:pandas.DataFrame.apply`

Reference/API
=============
.. automodapi:: sitelle.parallel
