"""Lensing filters.
This module computes different filters for the computation of the angular power
spectrum for the lensing potential.
"""
import numpy as np
import scipy.special as sc

__all_=[
    'lensing_efficiency',
    'filter_shear',
    'filter_convergence',
]


def lensing_efficiency(x, ibin):
    """Parametric lensing efficiency.
    This function computes the parametric form of the lensing efficiency
    function given in equation 6 in [1]_.
    Parameters
    ----------
    x: (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    ibin: integer
        Integer associated with a particular redshift bin.
    Returns
    -------
    efficiency: (nx,) array_like
        Array of lensing efficiency values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> qf = lensing_efficiency(xl, 1)

    References
    ----------
    ..[1] Tessore, N. and Harrison, I., 2020, arXiv:2003.11558.
    """
    mu = np.array([0.929, 1.278, 1.860, 2.473]) * 1000 / 0.7
    eta = np.array([1.235, 0.832, 0.551, 0.412]) * 0.7 / 1000

    alpha = 1.0 / (mu * eta - 1.0)
    beta = eta / (mu * eta -1.0)

    a = alpha[ibin - 1]
    b = beta[ibin - 1]

    t1 = sc.gammaincc(a + 1, b * x)
    t2 = b * x * sc.gammaincc(a, b * x) / a

    return t1 - t2


def filter_shear(x, ibin):
    """Filter for shear.
    This function filter for the shear power spectra, described in [1]_.
    Parameters
    ----------
    x: (nx,) array_like
        Array of comoving distances at which evaluate the lensing
        efficiency function.
    ibin: integer
        Integer associated with a particular redshift bin.
    Returns
    -------
    filter: (nx,) array_like
        Array of filter values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> fx = filter(xl, 1)

    References
    ----------
    ..[1] Lemos, P. and Challinor, A. and Efstathiou, G., 2017,
        arXiv: 1704.01054.
    """
    return lensing_efficiency(x, ibin) / x


def filter_convergence(x, z, ibin):
    """Filter for convergence.
    This function filter for the convergenge power spectra, described in [1]_.

    Parameters
    ----------
    x: float, numpy.ndarray
        Float or array of comoving distances at which evaluate the lensing
        efficiency function.
    z: array_like
        Redshift
    ibin: integer
        Integer associated with a particular redshift bin.
    Returns
    -------
    filter: float, numpy.ndarray
        FLoat or array of filter values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.comsology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=67.11, Ob0=0.049, Om0= 0.2685)
    >>> z_list = np.linspace(0, 1100,num=1000000)
    >>> xl = cosmo.comoving_distance(z_list).value
    >>> fx = filter(xl, 1)

    References
    ----------
    ..[1] Kilbinger, M., 2014, doi: 10.1088/0034-4885/78/8/086901.
    """
    z = np.asarray(z)
    return lensing_efficiency(x, ibin) * (1 + z) * x
