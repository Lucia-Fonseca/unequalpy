""" Power.
This module prepares the power spectra for the lensing analysis,
using different approximations and approaches.
"""

import numpy as np
import sys
sys.path.append("../power/EDS/")
from matter import matter_power_spectrum_1loop as P1loop
from matter import matter_unequal_time_power_spectrum as Puetc

__all__ = [
    'unequal_power_squared',
    'geometric_approx_squared',
    'midpoint_approx',
]


def unequal_power_squared(wavenumber, growth, powerk,
                    counterterm=0, model='spt'):
    r"""Squared of the exact unequal-time power spectrum.
    This function computes the squared unequal-time
    power spectrum, as described in equation 3.5 and 3.9 in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`[{\rm Mpc}^{-1}]`
        at which to evaluate the matter power spectrum.
    growth : (nz, 2) array_like
        Array of pair values of linear growth function at two redshifts.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    counterterm : (nz, 2) array_like
        Array of pairs of counterterms dealing with deviations
        from perfect fluid,  as described in equation 2.55 in [1],
        in units of :math:`[{\rm Mpc}^{2}]`. Default is 0.
    model : string, optional
        You can choose from two perturbation frameworks:
        {'spt': standard perturbation theory} described in [2] or
        {'eft': effective field theory} described in [2]. Default is 'spt'.

    Returns
    -------
    power_squared : (nz,nk) array_like
        The squared of the exact unequal-time
        power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`[{\rm Mpc}^{3}]`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.analysis import unequal_power_squared as Puneq2
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
    >>> D0 = growth_function(0, cosmo) / g0
    >>> vec0 = np.array([D0,D0])

    The best-fit counterterm using the Quijote simulations:

    >>> ct0 = -0.4
    >>> ct2 = np.array([ct0,ct0])

    And finally, the SPT and EFT equal-time matter power spectra:

    >>> pu2spt = Puneq2(ks, vec0, powerk)
    >>> pu2eft = Puneq2(ks, vec0, powerk, ct2, 'eft')

    References
    ----------
    ..[1] de la Bella, L., 2020.
    """
    p11, p22, p13 = powerk

    if np.ndim(growth) == 2 and np.ndim(wavenumber) == 1:
        growth = growth[np.newaxis, :]

    D1, D2 = growth.T
    D1s, D2s = np.square(growth).T

    P11_11 = (D1s * D2s) * np.square(p11(wavenumber))
    P11_22 = 2 * (D1s * D2s) * (D1 * D2) * p11(wavenumber) * p22(wavenumber)
    P11_13 = 2 * (D1s * D2s) * (D1s + D2s) *\
             p11(wavenumber) * 0.5 * p13(wavenumber)
    p2 = P11_11 + P11_22 + P11_13

    if model == 'spt':
        power_squared = p2
    elif model == 'eft':
        if np.ndim(counterterm) == 2 and np.ndim(wavenumber) == 1:
            counterterm = counterterm[np.newaxis, :]
        c2z1, c2z2 = counterterm.T
        ct = 0.5 * (c2z1 + c2z2)
        Pct = - 2 * ct * D1 * D2 * np.square(wavenumber) *\
               p11(wavenumber) * Puetc(wavenumber, growth, powerk)
        power_squared = p2 + Pct
    else:
        raise ValueError('Such model does not exist.\
                          Choose between "spt" or "eft"')
    return power_squared


def geometric_approx_squared(wavenumber, growth, powerk,
                    counterterm=0, model='spt'):
    r"""Geometric approximation for the power spectrum.
    This function computes the unequal-time geometric mean approximation
    for any power spectrum, as described in equation 3.7 and 3.11 in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`[{\rm Mpc}^{-1}]`
        at which to evaluate the matter power spectrum.
    growth : (nz, 2) array_like
        Array of pair values of linear growth function at two redshifts.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    counterterm : (nz, 2) array_like
        Array of pairs of counterterms dealing with deviations
        from perfect fluid,  as described in equation 2.55 in [1],
        in units of :math:`[{\rm Mpc}^{2}]`. Default is 0.
    model : string, optional
        You can choose from two perturbation frameworks:
        {'spt': standard perturbation theory} described in [2] or
        {'eft': effective field theory} described in [2]. Default is 'spt'.

    Returns
    -------
    power_power : (nz,nk) array_like
        The squared of the geometric approximation of the unequal-time
        power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`[{\rm Mpc}^{3}]`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.analysis import geometric_approx_squared as PPgeom
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
    >>> D0 = growth_function(0, cosmo) / g0
    >>> vec0 = np.array([D0,D0])

    The best-fit counterterm using the Quijote simulations:

    >>> ct0 = -0.4
    >>> ct2 = np.array([ct0,ct0])

    And finally, the SPT and EFT equal-time matter power spectra:

    >>> ppgspt = PPgeom(ks, vec0, powerk)
    >>> ppgeft = PPgeom(ks, vec0, powerk, ct2, 'eft')

    References
    ----------
    ..[1] de la Bella, L., 2020.
    """
    p11, p22, p13 = powerk

    if np.ndim(growth) == 2 and np.ndim(wavenumber) == 1:
        growth = growth[np.newaxis, :]

    D1, D2 = growth.T
    D1s, D2s = np.square(growth).T

    P11_11 = (D1s * D2s) * np.square(p11(wavenumber))
    P11_22 = (D1s * D2s) * (D1s + D2s) * p11(wavenumber) * p22(wavenumber)
    P11_13 = 2 * (D1s * D2s) * (D1s + D2s) *\
             p11(wavenumber) * 0.5 * p13(wavenumber)
    p_p = P11_11 + P11_22 + P11_13

    if model == 'spt':
        power_power = p_p
    elif model == 'eft':
        if np.ndim(counterterm) == 2 and np.ndim(wavenumber) == 1:
            counterterm = counterterm[np.newaxis, :]
        c2z1, c2z2 = counterterm.T
        Pct1 = - c2z1 * D1s * np.square(wavenumber) *\
               p11(wavenumber) * P1loop(wavenumber, D2, powerk)
        Pct2 = - c2z2 * D2s * np.square(wavenumber) *\
               p11(wavenumber) * P1loop(wavenumber, D1, powerk)
        power_power = p_p + Pct1 + Pct2
    else:
        raise ValueError('Such model does not exist.\
                          Choose between "spt" or "eft"')
    return power_power


def midpoint_approx(wavenumber, growth, powerk,
                    counterterm=0, model='spt'):
    r"""Midpoint approximation for the power spectrum.
    This function computes the unequal-time midpoint approximation for any
    power spectrum, as described in equation 2.16 in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`[{\rm Mpc}^{-1}]`
        at which to evaluate the matter power spectrum.
    growth : (nz, 2) array_like
        Array of pair values of linear growth function at two redshifts.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    counterterm : (nz, 2) array_like
        Array of pairs of counterterms dealing with deviations
        from perfect fluid,  as described in equation 2.55 in [1],
        in units of :math:`[{\rm Mpc}^{2}]`. Default is 0.
    model : string, optional
        You can choose from two perturbation frameworks:
        {'spt': standard perturbation theory} described in [2] or
        {'eft': effective field theory} described in [2]. Default is 'spt'.

    Returns
    -------
    power_spectrum : (nz,nk) array_like
        The midpoint power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`[{\rm Mpc}^{3}]`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.analysis import midpoint_approx as Pmid
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
    >>> D0 = growth_function(0, cosmo) / g0
    >>> vec0 = np.array([D0,D0])

    The best-fit counterterm using the Quijote simulations:

    >>> ct0 = -0.4
    >>> ct2 = np.array([ct0,ct0])

    And finally, the SPT and EFT equal-time matter power spectra:

    >>> pspt = Pmid(ks, vec0, powerk)
    >>> peft = Pmid(ks, vec0, powerk, ct2, 'eft')

    References
    ----------
    ..[1] de la Bella, L., 2020.
    """
    D1, D2 = growth.T
    D_mean = 0.5 * (D1 + D2)
    if np.ndim(counterterm):
        c2z1, c2z2 = counterterm.T
        c_mean = 0.5 * (c2z1 + c2z2)
    else:
        c_mean = 0.0


    return P1loop(wavenumber, D_mean, powerk, counterterm=c_mean, model=model)
