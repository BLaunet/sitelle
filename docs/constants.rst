.. _sitelle-constants:

*********************************
Constants (`sitelle.constants`)
*********************************

.. currentmodule:: sitelle.constants

Introduction
============

``sitelle.constants`` contains a number of physical constants heavily used
in the project.

They can be imported directly from
the ``sitelle.constants`` sub-package::

    >>> from sitelle.constants import M31_CENTER

or, if you want to avoid having to explicitly import all the constants you
need, you can simply do:

    >>> from sitelle import constants as const

and then subsequently use for example ``const.M31_CENTER``.

Caveats
=======
The ``FITS_DIR`` constant is more a convenient variable to avoid typing over and over the path hosting the fits files, but it is not used anywhere in the sitelle module, only in notebooks examples.
For now it has been implemented only for B. Launet on different machines.

Reference/API
=============
.. automodapi:: sitelle.constants
