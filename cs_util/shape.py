"""SHAPE.

:Name: shape.py

:Description: Conversions between the two ellipticity conventions used
    across the UNIONS / ShapePipe stack.

    A source's shape is a spin-2 quantity that can be expressed either
    as the *distortion* (e-type) or the *reduced shear* (g-type). For
    an ellipse of axis ratio ``q = b / a`` they have magnitudes

    - ``|e| = (1 - q^2) / (1 + q^2)`` — distortion, and
    - ``|g| = (1 - q)   / (1 + q)``   — reduced shear,

    related (at fixed position angle) by

    - ``g = e / (1 + sqrt(1 - e^2))``  and
    - ``e = 2 g / (1 + g^2)``.

    Both components share the orientation, so the conversion scales the
    complex ellipticity ``(comp1, comp2)`` by the magnitude ratio; the
    two-component functions below apply that ratio component-wise.

    Convention note: ShapePipe's HSM moments (``*_E*_*_HSM`` columns)
    store the distortion ``e``, while ngmix stores the reduced shear
    ``g``; this module is the single bridge between them, so producers
    (ShapePipe) and consumers (sp_validation) import from here rather
    than re-deriving the factors locally.

:Author: Cail Daley <cailmdaley@gmail.com>

"""

import numpy as np


def e_to_g(e1, e2):
    """E To G.

    Reduced shear ``g`` (g-type) from the distortion ``e`` (e-type),
    component-wise: ``g_i = e_i / (1 + sqrt(1 - |e|^2))`` with
    ``|e|^2 = e1^2 + e2^2``.

    Parameters
    ----------
    e1 : float or numpy.ndarray
        First distortion component
    e2 : float or numpy.ndarray
        Second distortion component

    Returns
    -------
    tuple of float or numpy.ndarray
        Reduced-shear components ``(g1, g2)``

    """
    factor = 1 / (1 + np.sqrt(1 - e1 ** 2 - e2 ** 2))
    return e1 * factor, e2 * factor


def g_to_e(g1, g2):
    """G To E.

    Distortion ``e`` (e-type) from the reduced shear ``g`` (g-type),
    component-wise: ``e_i = 2 g_i / (1 + |g|^2)`` with
    ``|g|^2 = g1^2 + g2^2``.

    Parameters
    ----------
    g1 : float or numpy.ndarray
        First reduced-shear component
    g2 : float or numpy.ndarray
        Second reduced-shear component

    Returns
    -------
    tuple of float or numpy.ndarray
        Distortion components ``(e1, e2)``

    """
    factor = 2 / (1 + g1 ** 2 + g2 ** 2)
    return g1 * factor, g2 * factor
