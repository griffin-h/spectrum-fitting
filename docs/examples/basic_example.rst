.. _basicexamples:

Basic Example
========================

Simple usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In here we show how to use spectrum-fitting in the simplest way possible.

.. code:: python

	specvels.py spectrum1.txt spectrum2.fits -l 656 -p emis

This will attempt to fit an emission line (``emis``) the Hα line (at 656 nm) in the two spectra you provide. To see other line profiles and other optional parameters, ``run specvels.py -h``.
