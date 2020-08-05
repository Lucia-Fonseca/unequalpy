""" Angular Power Spectrum.
This module computes the angular power spectrum by using different
approximations.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
sys.path.append("../power/EDS/")
from matter import matter_power_spectrum_1loop as P1loop

__all__=[
    'cl_dirac',
]

def cl_dirac(ell, filter1, filter2, powerk,
             growth_function, cosmology, zatchi=None,
             counterterm=0, model='spt'):
    r''' Dirac approximation.
    This function computes the angular power spectrum by approximating the
    spherical Bessel functions with the amplitude of their first peak. This
    has been described in [1]_, for example.

    Parameters
    ----------
    ell : (nl,) array_like
        Angular mode array.
    filter1, filter2 : function
        Filters corresponding to the weak lensing observable and the galaxy
        survey selection function. They are function of the comoving distance.
    powerk : tuple, function
        Tuple of functions for the linear, 22-type and 13-type power spectra
        at redshift zero.
    cosmology : astropy.cosmology.Cosmology
                 Cosmology object providing method for the evolution of
                 omega_matter with redshift.
    zatchi : function
        This function returns redshift at a given comoving distance.
        Default is None.
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
    angular_power : (nl,) array_like
        The angular power spectrum evaluated at the input angular modes.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units
    >>> from astropy import constants as const
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> from skypy.power_spectrum import growth_function
    >>> from unequalpy.filter import lensing_efficiency as le
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)

    We use precomputed values from the FAST-PT code:

    >>> d = np.loadtxt('../Pfastpt.txt',unpack=True)
    >>> ks = d[:, 0]
    >>> pk, p22, p13 = d[:, 1], d[:, 2], d[:, 3]

    >>> p11_int = interp1d( ks, pk, fill_value="extrapolate")
    >>> p22_int = interp1d( ks, p22, fill_value="extrapolate")
    >>> p13_int = interp1d( ks, p13, fill_value="extrapolate")
    >>> powerk = (p11_int, p22_int, p13_int)

    The filters at bins 1 and 2:

    >>> c = const.c.to('km/s')
    >>> H0= cosmo.H(0)
    >>> constant = (1.5 * cosmo.Om0 * H0**2 / c**2).value
    >>> def le1(x):
            return constant*le(x, 1)
    >>> def le2(x):
            return constant*le(x, 2)

    And finally, the angular power spectrum:

    >>> ell = 2 * np.logspace(-2,3)
    >>> cl_D = cl_dirac(ell,le1, le2, powerk, growth_function, cosmo)

    References
    ----------
    ..[1] Lemos, P. and Challinor, A. and Efstathiou, G., 2017,
        arXiv: 1704.01054.
    '''
    if zatchi:
        zatx = zatchi
        # zatx = results.redshift_at_comoving_radial_distance(x)
    else:
        z_list = np.linspace(0, 1100,num=1000000)
        xl = cosmology.comoving_distance(z_list).value
        zatx = interp1d( xl, z_list, fill_value="extrapolate")

    g0 = growth_function(0, cosmology)

    def integrand(x, ell):
        k = ell / x

        D_x = growth_function(np.float(zatx(x)), cosmology) / g0

        filters = filter1(x) * filter2(x)
        power = P1loop(k, D_x, powerk, counterterm, model)
        a_inverse2 = np.square(1 + zatx(x))
        return filters * a_inverse2 * power

    if np.ndim(ell):
        integral = np.empty(len(ell))
        for i in range(len(ell)):
            integral[i] = quad(integrand, 1.0e-4, 1.0e4, args=ell[i])[0]
    else:
        integral = quad(integrand, 1.0e-4, 1.0e4, args=ell)[0]

    return integral
