"""Power spectrum.
This module computes different versions of the power spectrum.
"""

import numpy as np
from scipy.interpolate import interp1d
from skypy import power_spectrum as ps
from astropy import units
from astropy import constants as const

def matter_power_spectrum_1loop(wavenumber, redshift, cosmology):
    """One-loop matter power spectrum.
    This function computes the one-loop matter power spectrum in real space in
    standard perturbation theory, as described in [1]_.

    Parameters
    ----------
    wavenumber : float or (nk,) numpy.ndarray
        Float or array of wavenumbers in units of [Mpc^-1] at which to evaluate the
        matter power spectrum.
    redshift : float or (nz,) numpy.ndarray
        Float or array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble parameter
        and CMB temperature at the present day.

    Returns
    -------
    power_spectrum : (nk,nz) numpy.ndarray, or (nk,) or (nz,) or float if input scalar
        The matter power spectrum evaluated at the input redshifts and wavenumbers
        for the given cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> pspt = matter_power_spectrum_1loop(0.1, 0, cosmo)

    References
    ----------
    ..[1] de la Bella, L. et al., 2017, doi:10.1088/1475-7516/2017/11/039.
    """
    g0 = ps.growth_function(0, cosmology)

    D = ps.growth_function(redshift, cosmology) / g0
    Ds = D * D

    P11 = Ds * p11_int(wavenumber)
    P22 = Ds * Ds * p22_int(wavenumber)
    P13 = Ds * Ds * p13_int(wavenumber)
    return P11 + P22 + P13

def matter_unequal_time_power_spectrum(wavenumber, redshift1, redshift2, cosmology):
    """Unequal-time one-loop matter power spectrum.
    This function computes the unequal-time one-loop matter power spectrum in
    real space in standard perturbation theory, as described in [1]_.

    Parameters
    ----------
    wavenumber : float or (nk,) numpy.ndarray
        Float or array of wavenumbers in units of [Mpc^-1] at which to evaluate the
        matter power spectrum.
    redshift1, redshift2 : float or (nz,) numpy.ndarray
        Float or array of redshifts at which to evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble parameter
        and CMB temperature at the present day.

    Returns
    -------
    power_spectrum : (nk,nz) numpy.ndarray, or (nk,) or (nz,) or float if input scalar
        The matter power spectrum evaluated at the input redshifts and wavenumbers
        for the given cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> pspt_uetc = matter_unequal_time_power_spectrum(0.1, 0, cosmo)

    References
    ----------
    ..[1] de la Bella, L. et al., 2020.
    """
    g0 = ps.growth_function(0, cosmology)

    D1 = ps.growth_function(redshift1, cosmology) / g0
    D2 = ps.growth_function(redshift1, cosmology) / g0
    D1s = D1 * D1
    D2s = D2 * D2

    P11 = D1 * D2 * p11_int(wavenumber)
    P22 = D1s * D2s * p22_int(wavenumber)
    P13 = 0.5 * D1 * D2 * (D1s + D2s) * p13_int(wavenumber)
    return P11 + P22 + P13


def lensing_power_spectrum_1loop(wavenumber, x, cosmology, variable=True):
    """One-loop lensing power spectrum.
    This function computes the lensing power spectrum from the one-loop lensing
    power spectrum in real space in standard perturbation theory through the
    Poisson equation, as described in [1]_.

    Parameters
    ----------
    wavenumber : float or (nk,) numpy.ndarray
        Float or array of wavenumbers in units of [Mpc^-1] at which to evaluate the
        matter power spectrum.
    x : float or (nz,) numpy.ndarray
        Float or array of points (redshift or comoving distances) at which to
        evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble parameter
        and CMB temperature at the present day.
    variable: boolean
        This argument allows you to work with redshift values (True) or comoving
        distances (False).

    Returns
    -------
    power_spectrum : (nk,nz) numpy.ndarray, or (nk,) or (nz,) or float if input scalar
        The matter power spectrum evaluated at the input redshifts and wavenumbers
        for the given cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> lensing_power = lensing_power_spectrum_1loop(0.1, 0, cosmo)

    References
    ----------
    ..[1] Lemos, P. and Challinor, A. and Efstathiou, G., 2017,
        arXiv: 1704.01054.
    """
    if variable:
        y = x
    else:
        y = z_of_chi(x, cosmology)

    map = np.power(_poisson_mapping(wavenumber,y, cosmology, variable=variable) , 2)
    return map * matter_power_spectrum_1loop(wavenumber, y, cosmology)


def lensing_unequal_time_power_spectrum(wavenumber, x1, x2, cosmology, variable=True):
    """Unequal-time one-loop lensing power spectrum.
    This function computes the unequal-time lensing power spectrum from the
    unequal-time one-loop matter power spectrum in real space in standard
    perturbation theory through the Poisson equation, as described in [1]_.

    Parameters
    ----------
    wavenumber : float or (nk,) numpy.ndarray
        Float or array of wavenumbers in units of [Mpc^-1] at which to evaluate the
        matter power spectrum.
    x1, x2 : float or (nz,) numpy.ndarray
        Float or array of points (redshift or comoving distances) at which to
        evaluate the growth function.
    cosmology : astropy.cosmology.Cosmology
        Cosmology object providing omega_matter, omega_baryon, Hubble parameter
        and CMB temperature at the present day.
    variable: boolean
        This argument allows you to work with redshift values (True) or comoving
        distances (False). Default is True.

    Returns
    -------
    power_spectrum : (nk,nz) numpy.ndarray, or (nk,) or (nz,) or float if input scalar
        The matter power spectrum evaluated at the input redshifts and wavenumbers
        for the given cosmology.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> pspt_uetc = unequal_time_matter_power_spectrum(0.1, 0, cosmo)

    References
    ----------
    ..[1] de la Bella, L. et al., 2020.
    """
    if variable:
        y1, y2 = x1, x2
    else:
        y1, y2 = z_of_chi(x1, cosmology), z_of_chi(x2, cosmology)

    map = _poisson_mapping(wavenumber, y1, cosmology, variable=variable) * \
        _poisson_mapping(wavenumber, y2, cosmology, variable=variable)
    return map * matter_unequal_time_power_spectrum(wavenumber, y1, y2, cosmology)



k11, p11 = np.loadtxt("/Users/c49734lf/Workspace/2019-2020/Ongoing projects/Unequal-time EFT/UETC_SPT_notebooks/EDS/p11.text", unpack=True)
k22, p22 = np.loadtxt("/Users/c49734lf/Workspace/2019-2020/Ongoing projects/Unequal-time EFT/UETC_SPT_notebooks/EDS/p22.dat", unpack=True)
k13, p13 = np.loadtxt("/Users/c49734lf/Workspace/2019-2020/Ongoing projects/Unequal-time EFT/UETC_SPT_notebooks/EDS/p13.dat", unpack=True)

p11_int = interp1d( k11, p11, fill_value="extrapolate")
p22_int = interp1d( k22, p22, fill_value="extrapolate")
p13_int = interp1d( k13, p13, fill_value="extrapolate")

def _z_of_chi(x, cosmology):
    z_list = np.linspace(0, 1100,num=1000000)
    chi_list = cosmology.comoving_distance(z_list).value
    return interp1d( chi_list, z_list, fill_value="extrapolate")


def _poisson_mapping(wavenumber, x, cosmology, variable=True):

    if variable:
        y = x
    else:
        y = z_of_chi(x, cosmology)

    omz = cosmology.Om(y)
    Hz = cosmology.H(y).value
    one_plus_z2 = np.power( 1.0 + y, 2)
    c = const.c.to('km/s').value

    k2 = np.power(wavenumber, 2)
    mapping = 1.5 * omz * np.power(Hz * one_plus_z2 / c, 2) / k2

    return  mapping
