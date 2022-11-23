"""CFIS.

:Name: cfis.py

:Description: This script contains CFIS-specific methods.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""


import re
import numpy as np
from astropy import units
from astropy import coordinates as coords


class Cfis(object):
    """Cfis

    Class for CFIS image properties.

    """
    size = {'tile' : 0.5 * units.deg}


def get_tile_number(tile_name):
    """Get Tile Number.

    Return tile number of given image tile name.

    Parameters
    ----------
    str
        tile name

    Raises
    ------
    ValueError
        if tile name does not match expected pipeline numbering scheme

    Returns
    -------
    tuple
        tile number for x and tile number for y

    """
    m = re.search(r'(\d{3})[\.-](\d{3})', tile_name)
    if m is None or len(m.groups()) != 2:
        raise ValueError(
            f'Image name \'{tile_name}\' does not match tile name syntax'
        )

    nix = m.groups()[0]
    niy = m.groups()[1]

    return nix, niy


def get_tile_coord_from_nixy(nix, niy):
    """Get Tile Coord From Nixy.

    Return coordinates corresponding to tile with number (nix,niy).

    Parameters
    ----------
    nix : str or int
        tile number for x, can be list
    niy : str or int
        tile number for y, can be list

    See also
    --------
    get_tile_number_from_coord

    Returns
    -------
    tuple
        right ascension and declination

    """
    if not np.isscalar(nix):
        # Transform to int
        xi = np.array(nix).astype(int)
        yi = np.array(niy).astype(int)
    else:
        xi = int(nix)
        yi = int(niy)

    # Declination
    d = yi / 2 - 90
    dec = coords.Angle(d, unit='deg')

    # Right ascension
    r = xi / 2 / np.cos(dec.radian)
    ra = coords.Angle(r, unit='deg')

    return ra, dec