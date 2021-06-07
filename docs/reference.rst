Reference
=========

The ``unequalpy`` package contains the following modules:

Matter power spectrum
---------------------
This module computes the equal- and unequal-time matter power spectrum for
different formalisms: Standard Perturbation Theory and Effective Field Theory.

- :func:`~unequalpy.matter.matter_power_spectrum_1loop`
- :func:`~unequalpy.matter.matter_unequal_time_power_spectrum`

----

.. autofunction:: unequalpy.matter.matter_power_spectrum_1loop

.. autofunction:: unequalpy.matter.matter_unequal_time_power_spectrum



Approximation to the power spectrum
-----------------------------------
This module prepares the power spectra for the lensing analysis,
using different approximations to the unequal-time power spectrum.

- :func:`~unequalpy.approximation.geometric_approx`
- :func:`~unequalpy.approximation.midpoint_approx`
- :func:`~unequalpy.approximation.growth_midpoint`

----

.. autofunction:: unequalpy.approximation.geometric_approx

.. autofunction:: unequalpy.approximation.midpoint_approx



Lens filters
------------
This module computes different filters for the computation of the angular power
spectrum for the lensing potential

- :func:`~unequalpy.lens_filter.lensing_efficiency`
- :func:`~unequalpy.lens_filter.lensing_efficiency_cmb`
- :func:`~unequalpy.lens_filter.filter_shear`
- :func:`~unequalpy.lens_filter.filter_convergence`
- :func:`~unequalpy.lens_filter.filter_galaxy_clustering`

----

.. autofunction:: unequalpy.lens_filter.lensing_efficiency

.. autofunction:: unequalpy.lens_filter.lensing_efficiency_cmb

.. autofunction:: unequalpy.lens_filter.filter_shear

.. autofunction:: unequalpy.lens_filter.filter_convergence

.. autofunction:: unequalpy.lens_filter.filter_galaxy_clustering
