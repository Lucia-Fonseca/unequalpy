""" Power spectrum.
This module prepares the power spectra for the lensing analysis,
using different approximations to the unequal-time power spectrum.
"""

__version__ = '0.1.0'

__author__ = 'Lucia F. de la Bella'
__email__ = 'lucia.fonsecadelabella@manchester.ac.uk'
__license__ = 'MIT'
__copyright__ = '2020, Lucia Fonseca de la Bella'

__all__ = [
    'geometric_approx',
    'midpoint_approx',
]

import numpy as np


def geometric_approx(power_spectrum):
    r"""Geometric approximation for the power spectrum.
    This function computes the unequal-time geometric mean approximation
    for any power spectrum, as described in equation 1.3 in [1]_.

    Parameters
    ----------
    power_spectrum : (nz, k) array_like
        Array of power spectra at different redshifts.

    Returns
    -------
    power_power : (nz, nz, nk) array_like
        The geometric approximation of the unequal-time
        power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`{\rm Mpc}^{3}`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.approximation import geometric_approx as Pgeom
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)

    We use precomputed values from the FAST-PT code:

    >>> d = np.loadtxt('../Pfastpt.txt',unpack=True)
    >>> ks = d[:, 0]
    >>> pk, p22, p13 = d[:, 1], d[:, 2], d[:, 3]

    >>> p11_int = interp1d( ks, pk, fill_value="extrapolate")
    >>> p22_int = interp1d( ks, p22, fill_value="extrapolate")
    >>> p13_int = interp1d( ks, p13, fill_value="extrapolate")
    >>> powerk = (p11_int, p22_int, p13_int)

    The normalised growth function from SkyPy:

    >>> g0 = growth_function(0, cosmo)
    >>> z = np.array([0,1])
    >>> Dz = growth_function(z, cosmo)]) / g0

    The equal-time matter power spectrum

    >>> pet = P1loop(ks, Dz, powerk)

    And finally, the geometric approximation to the
    unequal-time matter power spectrum:

    >>> pg_spt = Pgeom(pet)

    References
    ----------
    ..[1] de la Bella, L. and Tessore, N. and Bridle, S., 2020.
    """

    return np.sqrt(power_spectrum[:, None] * power_spectrum[None, :])


def midpoint_approx(wavenumber, growth1, growth2, powerk,
                    counterterm1=0, counterterm2=0, model='spt'):
    r"""Midpoint approximation for the power spectrum.
    This function computes the unequal-time midpoint approximation for any
    power spectrum, as described in equation 2.14 in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`{\rm Mpc}^{-1}`
        at which to evaluate the matter power spectrum.
    growth1, growth2 : (nz,), (nz2,) array_like
        Arrays of linear growth function at two redshifts.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    counterterm1, counterterm2 : (nz1,), (nz2,) array_like
        Array of counterterms dealing with deviations
        from perfect fluid,  as described in equation 2.55 in [1],
        in units of :math:`{\rm Mpc}^{2}`. Default is 0.
    model : string, optional
        You can choose from two perturbation frameworks:
        {'spt': standard perturbation theory} or
        {'eft': effective field theory}. Default is 'spt'.

    Returns
    -------
    power : (nz1 * nz2, nk) array_like
        The midpoint power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`{\rm Mpc}^{3}`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.approximation import midpoint_approx as Pmid
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)

    We use precomputed values from the FAST-PT code:

    >>> d = np.loadtxt('../Pfastpt.txt',unpack=True)
    >>> ks = d[:, 0]
    >>> pk, p22, p13 = d[:, 1], d[:, 2], d[:, 3]

    >>> p11_int = interp1d( ks, pk, fill_value="extrapolate")
    >>> p22_int = interp1d( ks, p22, fill_value="extrapolate")
    >>> p13_int = interp1d( ks, p13, fill_value="extrapolate")
    >>> powerk = (p11_int, p22_int, p13_int)

    The normalised growth function from SkyPy:

    >>> g0 = growth_function(0, cosmo)
    >>> D0, D1 = 1.0, growth_function(1, cosmo)]) / g0

    Finally, the midpoint approximation to the
    unequal-time matter power spectrum:

    >>> pspt = Pmid(ks, D0, D1, powerk)

    References
    ----------
    ..[1] de la Bella, L. and Tessore, N. and Bridle, S., 2020.
    """
    p11, p22, p13 = powerk

    if np.ndim(wavenumber) == 1 and \
            (np.ndim(growth1) == 1 and np.ndim(growth2) == 1):
        wavenumber = wavenumber[:, np.newaxis][:, np.newaxis]
        growth1 = growth1[np.newaxis, :]
        growth2 = growth2[:, np.newaxis]
        if (np.ndim(counterterm1) == 1 and np.ndim(counterterm2) == 1):
            counterterm1 = counterterm1[np.newaxis, :]
            counterterm2 = counterterm2[:, np.newaxis]
    elif np.ndim(wavenumber) == 1 and \
            (np.ndim(growth1) == 1 or np.ndim(growth2) == 1):
        wavenumber = wavenumber[:, np.newaxis]

    D_mean = 0.5 * (growth1 + growth2)
    D_means = np.square(D_mean)

    P11 = D_means * p11(wavenumber)
    P22 = D_means * D_means * p22(wavenumber)
    P13 = D_means * D_means * p13(wavenumber)

    if model == 'spt':
        power_spectrum = P11 + P22 + P13
    elif model == 'eft':
        c_mean = 0.5 * (counterterm1 + counterterm2)
        Pct = - c_mean * D_means * np.square(wavenumber) * p11(wavenumber)
        power_spectrum = P11 + P22 + P13 + Pct
    else:
        raise ValueError('Such model does not exist.\
                      Choose between "spt" or "eft"')
    return power_spectrum.T
