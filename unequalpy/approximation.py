""" Power spectrum.
This module prepares the power spectra for the lensing analysis,
using different approximations to the unequal-time power spectrum.
"""

__version__ = '0.1.6'

__author__ = 'Lucia F. de la Bella'
__email__ = 'lucia.fonsecadelabella@manchester.ac.uk'
__license__ = 'MIT'
__copyright__ = '2020, Lucia Fonseca de la Bella'

__all__ = [
    'geometric_approx',
    'midpoint_approx',
    'growth_midpoint',
]

import numpy as np
from unequalpy.matter import matter_power_spectrum_1loop as Petc


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
    ..[1] de la Bella, L. and Tessore, N. and Bridle, S., 2020,
        arXiv 2011.06185.
    """

    return np.sqrt(power_spectrum[:, None] * power_spectrum[None, :])


def midpoint_approx(wavenumber, growth_mean, powerk):
    r"""Midpoint approximation for the power spectrum.
    This function computes the unequal-time midpoint approximation for any
    power spectrum, as described in equation 2.14 in [1]_.

    Parameters
    ----------
    wavenumber : (nk,) array_like
        Array of wavenumbers in units of :math:`{\rm Mpc}^{-1}`
        at which to evaluate the matter power spectrum.
    growth_mean : (nz1, nz2), array_like
        Growth function array evaluated at the midpoint redshift.
    powerk: tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.

    Returns
    -------
    power : (nz1, nz2, nk), array_like
        The midpoint power spectrum evaluated at the input redshifts
        and wavenumbers for the given cosmology.
        Units of :math:`{\rm Mpc}^{3}`.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.approximation import midpoint_approx
    >>> from unequalpy.approximation import growth_midpoint
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)

    We use precomputed values from the FAST-PT code:

    >>> d = np.loadtxt('../Pfastpt.txt',unpack=True)
    >>> ks = d[:, 0]
    >>> pk, p22, p13 = d[:, 1], d[:, 2], d[:, 3]

    >>> p11_int = interp1d( ks, pk, fill_value="extrapolate")
    >>> p22_int = interp1d( ks, p22, fill_value="extrapolate")
    >>> p13_int = interp1d( ks, p13, fill_value="extrapolate")
    >>> powerk = (p11_int, p22_int, p13_int)

    The normalised growth function from SkyPy
    at some midoint redshift between z1 and z2:

    >>> z1, z2 = 0, 2
    >>> D12  = growth_midpoint(z1, z2, growth_function, cosmo)

    Finally, the midpoint approximation to the
    unequal-time matter power spectrum:

    >>> pspt = midpoint_approx(ks, D12, powerk)

    References
    ----------
    ..[1] de la Bella, L. and Tessore, N. and Bridle, S., 2020,
        arXiv 2011.06185.
    """
    p11, p22, p13 = powerk

    power_spectrum = []
    for gzm in growth_mean:
        power_spectrum.append(Petc(wavenumber, gzm, powerk))
    return np.asarray(power_spectrum)


def growth_midpoint(redshift1, redshift2, growth_function, *args):
    r"""Growth function at the midpoint.
    This function computes the linear growth function evaluated at the mean of
    two given redshift values.

    Parameters
    ----------
    redshift1, redshift2 : (nz1,), (nz2,), array_like
        Array of wavenumbers in units of :math:`{\rm Mpc}^{-1}`
        at which to evaluate the matter power spectrum.
    growth_function : function
        Method to compute the growth function.
    *args: tuple
        Arguments for the growth function method.

    Returns
    -------
    growth : (nz1, nz2), array_like
        The growth function evaluated at the midpoint redshift.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.approximation import growth_midpoint
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)

    The normalised growth function from SkyPy (you can choose any other method)
    at some midoint redshift between z1 and z2:

    >>> z1, z2 = 0, 2
    >>> D12  = growth_midpoint(z1, z2, growth_function, cosmo)

    """
    zm = 0.5 * np.add.outer(redshift1, redshift2)
    return growth_function(zm, *args) / growth_function(0, *args)
