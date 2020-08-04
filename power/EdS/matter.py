"""Power spectrum.
This module computes different versions of the matter power spectrum.
"""

import numpy as np

__all__ = [
    'matter_power_spectrum_1loop',
    'matter_unequal_time_power_spectrum',
]


def matter_power_spectrum_1loop(wavenumber, growth, powerk,
                                counterterm=0, model='spt'):
    r"""One-loop matter power spectrum.
    This function computes the one-loop matter power spectrum in real space in
    standard perturbation theory at a single redshift, as described in [1]_
    and [2]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`[{\rm Mpc}^{-1}]` at which to
        evaluate the matter power spectrum.
    growth : (nz,) array_like
        Array of linear growth function at one single redshift.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    counterterm : (nz,) array_like
        Array of counterterms dealing with deviations from perfect fluid, as
        described in equation 2.55 in [2], in units of :math:`[{\rm Mpc}^{2}]`.
        Default is 0.
    model : string, optional
        You can choose from two perturbation frameworks:
        {'spt': standard perturbation theory} described in [1] or
        {'eft': effective field theory} described in [2]. Default is 'spt'.

    Returns
    -------
    power_spectrum : (nz,nk) array_like
        The matter power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`[{\rm Mpc}^{3}]`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
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

    The best-fit counterterm using the Quijote simulations:

    >>> ct2 = -0.4

    And finally, the SPT and EFT equal-time matter power spectra:

    >>> pspt = matter_power_spectrum_1loop(k, D0, powerk)
    >>> peft = matter_power_spectrum_1loop(k, D0, ct2, model='eft')

    References
    ----------
    ..[1] astro-ph/0604075.
    ..[2] de la Bella, L. et al., 2017, doi:10.1088/1475-7516/2017/11/039.
    """
    p11, p22, p13 = powerk

    if np.ndim(wavenumber) == 1 and np.ndim(growth) == 1:
        growth = growth[:, np.newaxis]

    Ds = np.square(growth)

    P11 = Ds * p11(wavenumber)
    P22 = Ds * Ds * p22(wavenumber)
    P13 = Ds * Ds * p13(wavenumber)
    if model == 'spt':
        power_spectrum = P11 + P22 + P13
    elif model == 'eft':
        Pct = - counterterm * Ds * np.square(wavenumber) * p11(wavenumber)
        power_spectrum = P11 + P22 + P13 + Pct
    else:
        raise ValueError('Such model does not exist.\
                          Choose between "spt" or "eft"')
    return power_spectrum


def matter_unequal_time_power_spectrum(wavenumber, growth, powerk,
                                       counterterm=0, model='spt'):
    r"""Unequal-time one-loop matter power spectrum.
    This function computes the unequal-time one-loop matter power spectrum in
    real space in standard perturbation theory, as described in [1]_.

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
        The unequal-time matter power spectrum evaluated at the input
        redshifts and wavenumbers for the given cosmology,
        in units of :math:`[{\rm Mpc}^{3}]`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
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
    >>> Dv = np.array([g0, growth_function(1, cosmo)]) / g0

    The best-fit counterterm using the Quijote simulations:

    >>> ct2 = np.array([-0.4,-1])

    And finally, the SPT and EFT unequal-time matter power spectra:

    >>> pspt = matter_unequal_time_power_spectrum(0.1, Dv, powerk)
    >>> peft = matter_unequal_time_power_spectrum(0.1, Dv, powerk, ct2, 'eft')

    References
    ----------
    ..[1] de la Bella, L., 2020.
    """
    p11, p22, p13 = powerk

    if np.ndim(growth) == 2 and np.ndim(wavenumber) == 1:
        growth = growth[np.newaxis, :]

    D1, D2 = growth.T
    D1s, D2s = np.square(growth).T

    P11 = (D1 * D2) * p11(wavenumber)
    P22 = (D1s * D2s) * p22(wavenumber)
    P13 = 0.5 * D1 * D2 * (D1s + D2s) * p13(wavenumber)
    if model == 'spt':
        power_spectrum = P11 + P22 + P13
    elif model == 'eft':
        if np.ndim(counterterm) == 2 and np.ndim(wavenumber) == 1:
            counterterm = counterterm[np.newaxis, :]
        c2z1, c2z2 = counterterm.T
        ct = 0.5 * (c2z1 + c2z2)
        Pct = - ct * D1 * D2 * np.square(wavenumber) * p11(wavenumber)
        power_spectrum = P11 + P22 + P13 + Pct
    else:
        raise ValueError('Such model does not exist.\
                          Choose between "spt" or "eft"')
    return power_spectrum
