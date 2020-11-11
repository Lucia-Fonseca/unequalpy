"""Lensing filters.
This module computes different filters for the computation of the angular power
spectrum for the lensing potential.
"""

__version__ = '0.1.0'

__author__ = 'Lucia F. de la Bella'
__email__ = 'lucia.fonsecadelabella@manchester.ac.uk'
__license__ = 'MIT'
__copyright__ = '2020, Lucia Fonseca de la Bella'

__all__ = [
    'lensing_efficiency',
    'lensing_efficiency_cmb',
    'filter_shear',
    'filter_convergence',
    'filter_galaxy_clustering',
]

import numpy as np

THREE_2C2 = 1.669e-11
INVERSE_C = 3.333e-6

mu = np.array([0.929, 1.278, 1.860, 2.473]) * 1000 / 0.7
eta = np.array([1.235, 0.832, 0.551, 0.412]) * 0.7 / 1000

alpha = 1.0 / (mu * eta - 1.0)
beta = eta / (mu * eta - 1.0)


def lensing_efficiency(x, zx, nz):
    """Lensing efficiency.
    This function computes the general form of the lensing efficiency
    function given in equation x in [1]_.

    Parameters
    ----------
    x: (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    zx: array_like
        Array of redshift as a function of comoving distance.
    nz : (nx, ) array_like
        Array of redshift distribution of galaxies.

    Returns
    -------
    efficiency: (nx,) array_like
        Array of lensing efficiency values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> qf = lensing_efficiency(xl, 1)

    References
    ----------
    ..[1] Lemos, P. and Challinor, A. and Efstathiou, G., 2017,
        arXiv: 1704.01054.
    """
    TINY = np.finfo(0.).tiny

    return np.trapz(np.clip(x - x[:, None], 0, None) / (x + TINY) * nz, zx)


def lensing_efficiency_cmb(x, xs):
    """Parametric lensing efficiency cmb.
    This function computes the  cmb lensing efficiency
    function given in equation x in [1]_.

    Parameters
    ----------
    x : (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    xs : float
        Value of the comoving distance at the last scatering surface.

    Returns
    -------
    efficiency: (nx,) array_like
        Array of lensing efficiency values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xs = 14000
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> qf = lensing_efficiency_cmb(xl, xs)

    References
    ----------
    ..[1] Hanson, D. and Challinor, A. and Lewis, A., 2010,
        doi: 10.1007/s10714-010-1036-y.
    """
    return 1.0 - x / xs


def filter_shear(x, zx, lens_efficiency, cosmology):
    """Filter for shear.
    This function computes the filter for the shear power spectra,
    described in [1]_.

    Parameters
    ----------
    x: (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    zx: array_like
        Array of redshift as a function of comoving distance.
    lens_efficiency: (nx,) array_like
        Array of lensing efficiency values.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    filter: (nx,) array_like
        Array of filter values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> q1 = lensing_efficiency(xl, 1)
    >>> f1 = filter(xl, z_list, q1, cosmology)

    References
    ----------
    ..[1] Lemos, P. and Challinor, A. and Efstathiou, G., 2017,
        arXiv: 1704.01054.
    """
    factor = THREE_2C2 * np.square(cosmology.H0) * cosmology.Om0
    return factor.value * (1.0 + zx) * lens_efficiency / x


def filter_convergence(x, zx, lens_efficiency, cosmology):
    """Filter for convergence.
    This function computes the filter for the convergenge power spectra,
    described in [1]_.

    Parameters
    ----------
    x: (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    zx: array_like
        Array of redshift as a function of comoving distance.
    lens_efficiency: (nx,) array_like
        Array of lensing efficiency values.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    filter: (nx,) array_like
        Array of filter values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> q1 = lensing_efficiency(xl, 1)
    >>> f1 = filter(xl, z_list, q1, cosmology)

    References
    ----------
    ..[1] Kilbinger, M., 2014, doi: 10.1088/0034-4885/78/8/086901.
    """
    factor = THREE_2C2 * np.square(cosmology.H0) * cosmology.Om0
    return factor.value * (1.0 + zx) * lens_efficiency * x


def filter_galaxy_clustering(x, zx, nz, linear_bias, cosmology):
    """Filter for galaxy clustering.
    This function computes filter for the galaxy clustering power spectra,
    described in [1]_.

    Parameters
    ----------
    x : (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    zx : array_like
        Array of redshift as a function of comoving distance.
    nz: (nx,) array_like
        Array of redshift distribution of galaxies.
    linear_bias : float
        Float for the value of the linear bias parameter.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing methods for the evolution history of
        omega_matter and omega_lambda with redshift.

    Returns
    -------
    filter: (nx,) array_like
        Array of filter values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> n1 = redshift_distribution_galaxies(xl, 1)
    >>> f1 = filter(xl, z_list, n1, cosmo)

    References
    ----------
    ..[1] Kilbinger, M., 2014, doi: 10.1088/0034-4885/78/8/086901.
    """
    H = cosmology.H0 * cosmology.efunc(zx)
    return H.value * linear_bias * nz * INVERSE_C
