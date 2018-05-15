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
need, you can simply do::

    >>> from sitelle import constants as const

and then subsequently use for example ``const.M31_CENTER``.

Caveats
=======
The ``FITS_DIR`` constant is more a convenient variable to avoid typing over and over the path hosting the fits files, but it is not used anywhere in the sitelle module, only in notebooks examples.
For now it has been implemented only for B. Launet on different machines.

Once a stable directory structure has been set, it is very useful to avoid typing the whole paths each time::

    >>> from sitelle.constants import FITS_DIR
    >>> from orcs.process import SpectralCube
    >>> SN2_ORCS = SpectralCube(FITS_DIR/'orig/M31_SN2.merged.cm1.1.0.hdf5')
    >>> SN2_ORCS.set_wcs(FITS_DIR/'M31_SN2.1.0.ORCS/M31_SN2.1.0.wcs.deep_frame.fits')
    >>> SN2_ORCS.set_dxdymaps(FITS_DIR/'M31_SN2.1.0.ORCS/M31_SN2.1.0.wcs.dxmap.fits', FITS_DIR/'M31_SN2.1.0.ORCS/M31_SN2.1.0.wcs.dymap.fits')
    >>> SN2_ORCS.correct_wavelength(FITS_DIR/'M31_SN2.1.0.ORCS/M31_SN2.1.0.skymap.fits')

Reference/API
=============
.. automodapi:: sitelle.constants
