===============================================================
unequalPy: A package for the unequal-time matter power spectrum
===============================================================

|Zenodo Badge| |PyPI Status| |Documentation Status|

This package contains functions to obtain the unequal-time power spectrum at one-loop
in standard perturbation theory an effective field theory. It also provides functions
to reproduce the analysis in [1]_:

* Approximations to the unequal-time matter power spectrum: geometric and the midpoint approximations.
* Weak lensing functions: filters and lensing efficiency.
* Tests using DES-Y1 data and cosmoSIS.

The full list of features can be found in the `unequalPy Documentation`_.

If you use UnequalPy for work or research presented in a publication please follow
our `Citation Guidelines`_.

.. _unequalPy Documentation: https://unequalpy.readthedocs.io/en/latest/
.. _Citation Guidelines: CITATION


Getting Started
---------------

unequalPy is distributed through PyPI_. To install UnequalPy and its
dependencies_ using pip_:

.. code:: bash

    $ pip install unequalpy

The unequalPy library can then be imported from python:

.. code:: python

    >>> import unequalpy
    >>> help(unequalpy)

.. _PyPI: https://pypi.org/project/unequalpy/
.. _dependencies: setup.cfg
.. _pip: https://pip.pypa.io/en/stable/


References
----------
.. [1] de la Bella, L. and Tessore, N. and Bridle, S., 2020.


.. |Zenodo Badge| image:: https://zenodo.org/badge/221432358.svg
   :target: https://zenodo.org/badge/latestdoi/221432358
   :alt: DOI of Latest unequalPy Release

.. |PyPI Status| image:: https://img.shields.io/pypi/v/unequalpy.svg
    :target: https://pypi.python.org/pypi/unequalpy
    :alt: unequalPy's PyPI Status

.. |Documentation Status| image:: https://readthedocs.org/projects/githubapps/badge/?version=latest
    :target: https://unequal.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
