"""SIZE.

:Name: size.py

:Description: Conversions between the Gaussian object-size measures
    used across the UNIONS / ShapePipe stack.

    All conversions assume a circular Gaussian profile, for which the
    measures are related through the scale parameter ``sigma``:

    - ``T = 2 sigma^2`` — the ngmix / DES area parameter (arcsec^2),
    - ``r50 = sqrt(2 ln 2) sigma = 1.17741 sigma`` — the half-light
      radius (the primary size in the UNIONS shape-catalogue papers),
    - ``FWHM = 2 sqrt(2 ln 2) sigma = 2.35482 sigma``.

    These functions are the single source of truth for the size web;
    producers (ShapePipe) and consumers (sp_validation) should import
    from here rather than re-deriving the factors locally.

:Author: Cail Daley

"""

import numpy as np

# Half-light radius of a circular Gaussian per unit sigma:
# r50 = sqrt(2 ln 2) * sigma
SIGMA_TO_R50 = np.sqrt(2 * np.log(2))

# Full width at half maximum of a Gaussian per unit sigma:
# FWHM = 2 sqrt(2 ln 2) * sigma
SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))


def T_to_sigma(T):
    """T To Sigma.

    Gaussian scale ``sigma = sqrt(T / 2)`` from the ngmix area
    parameter ``T = 2 sigma^2``.

    Parameters
    ----------
    T : float or numpy.ndarray
        ngmix area parameter, ``T = 2 sigma^2``

    Returns
    -------
    float or numpy.ndarray
        Gaussian scale ``sigma``, in units of ``sqrt(T)``

    """
    return np.sqrt(T / 2)


def sigma_to_T(sigma):
    """Sigma To T.

    ngmix area parameter ``T = 2 sigma^2`` from the Gaussian scale.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Gaussian scale

    Returns
    -------
    float or numpy.ndarray
        ngmix area parameter ``T``, in units of ``sigma^2``

    """
    return 2 * sigma ** 2


def sigma_to_r50(sigma):
    """Sigma To R50.

    Half-light radius ``r50 = sqrt(2 ln 2) sigma`` of a circular
    Gaussian.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Gaussian scale

    Returns
    -------
    float or numpy.ndarray
        half-light radius, in units of ``sigma``

    """
    return SIGMA_TO_R50 * sigma


def r50_to_sigma(r50):
    """R50 To Sigma.

    Gaussian scale from the half-light radius.

    Parameters
    ----------
    r50 : float or numpy.ndarray
        half-light radius

    Returns
    -------
    float or numpy.ndarray
        Gaussian scale ``sigma``, in units of ``r50``

    """
    return r50 / SIGMA_TO_R50


def sigma_to_fwhm(sigma):
    """Sigma To Fwhm.

    Full width at half maximum ``FWHM = 2 sqrt(2 ln 2) sigma`` of a
    Gaussian.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Gaussian scale

    Returns
    -------
    float or numpy.ndarray
        FWHM, in units of ``sigma``

    """
    return SIGMA_TO_FWHM * sigma


def fwhm_to_sigma(fwhm):
    """Fwhm To Sigma.

    Gaussian scale from the full width at half maximum.

    Parameters
    ----------
    fwhm : float or numpy.ndarray
        full width at half maximum

    Returns
    -------
    float or numpy.ndarray
        Gaussian scale ``sigma``, in units of ``fwhm``

    """
    return fwhm / SIGMA_TO_FWHM


def T_to_r50(T):
    """T To R50.

    Half-light radius ``r50 = sqrt(2 ln 2) * sqrt(T / 2)
    = sqrt(ln 2 * T)`` from the ngmix area parameter.

    Parameters
    ----------
    T : float or numpy.ndarray
        ngmix area parameter, ``T = 2 sigma^2``

    Returns
    -------
    float or numpy.ndarray
        half-light radius, in units of ``sqrt(T)``

    """
    return sigma_to_r50(T_to_sigma(T))


def r50_to_T(r50):
    """R50 To T.

    ngmix area parameter ``T = 2 (r50 / sqrt(2 ln 2))^2 = r50^2 / ln 2``
    from the half-light radius.

    Parameters
    ----------
    r50 : float or numpy.ndarray
        half-light radius

    Returns
    -------
    float or numpy.ndarray
        ngmix area parameter ``T``, in units of ``r50^2``

    """
    return sigma_to_T(r50_to_sigma(r50))


def T_to_fwhm(T):
    """T To Fwhm.

    Full width at half maximum ``FWHM = 2 sqrt(2 ln 2) * sqrt(T / 2)``
    from the ngmix area parameter.

    Note that ``T`` is an *area*: the conversion to the length ``FWHM``
    involves a square root, not a linear factor.

    Parameters
    ----------
    T : float or numpy.ndarray
        ngmix area parameter, ``T = 2 sigma^2``

    Returns
    -------
    float or numpy.ndarray
        FWHM, in units of ``sqrt(T)``

    """
    return sigma_to_fwhm(T_to_sigma(T))
