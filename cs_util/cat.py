"""CAT.

:Name: cat.py

:Description: This script contains methods to read and write galaxy catalogues.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import os
from datetime import datetime
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table


def write_header_info_sp(primary_header, name="unknown", version="unknown"):
    """Write Header Info sp_validation.

    Write information about software and run to FITS header

    Parameters
    ----------
    primary_header : dict
       FITS header information
    name : str
        software name, default is 'unknown'
    version : str
        version, default is 'unknown'

    Returns
    -------
    dict
        updated FITS header information

    """
    if "USER" in os.environ:
        author = os.environ["USER"]
    else:
        author = "unknown"
    primary_header["AUTHOR"] = (author, "Who ran the software")
    primary_header["SOFTNAME"] = (name, "Name of the software")
    primary_header["SOFTVERS"] = (version, "Version of the software")
    primary_header["DATE"] = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "When it was started",
    )

    return primary_header


def add_shear_bias_to_header(primary_header, R, R_shear, R_select, c):
    """Add Shear Bias To Header.

    Add information about multiplicative and additive shear bias
    from metacalibration to FITS header.

    Parameters
    ----------
    primary_header : dict
        FITS header information
    R : 2x2 matrix
        full response matrix
    R_shear : 2x2-matrix
        shear response matrix
    R_select : 2x2-matrix
        selection response matrix
    c : 2-tuple
        additive bias

    """
    primary_header["R"] = (r"<R>", r"Mean full response <R_shear> + <R_select>")
    primary_header["R_11"] = (R[0, 0], "Full response matrix comp 1 1")
    primary_header["R_12"] = (R[0, 1], "Full response matrix comp 1 2")
    primary_header["R_21"] = (R[1, 0], "Full response matrix comp 2 1")
    primary_header["R_22"] = (R[1, 1], "Full response matrix comp 2 2")

    primary_header["R_g"] = (r"<R_g>", r"Mean shear response matrix <R_shear>")
    primary_header["R_g11"] = (R_shear[0, 0], "Mean shear resp matrix comp 1 1")
    primary_header["R_g12"] = (R_shear[0, 1], "Mean shear resp matrix comp 1 2")
    primary_header["R_g21"] = (R_shear[1, 0], "Mean shear resp matrix comp 2 1")
    primary_header["R_g22"] = (R_shear[1, 1], "Mean shear resp matrix comp 2 2")

    primary_header["R_S"] = (
        r"<R_S>",
        r"Global selection response matrix <R_select>",
    )
    primary_header["R_S11"] = (
        R_select[0, 0],
        "Global selection resp matrix comp 1 1",
    )
    primary_header["R_S12"] = (
        R_select[0, 1],
        "Global selection resp matrix comp 1 2",
    )
    primary_header["R_S21"] = (
        R_select[1, 0],
        "Global selection resp matrix comp 2 1",
    )
    primary_header["R_S22"] = (
        R_select[1, 1],
        "Global selection resp matrix comp 2 2",
    )

    primary_header["c_1"] = (c[0], "Additive bias 1st comp")
    primary_header["c_2"] = (c[1], "Additive bias 2nd comp")


def write_ascii_table_file(cols, names, fname):
    """Write Ascii Table File.

    Write ASCII file with table data.

    Parameters
    ----------
    cols : list
        data columns
    names : list of str
        column names
    fname : str
        output file name

    """
    t = Table(cols, names=names)
    with open(fname, "w") as fout:
        ascii.write(t, fout, delimiter="\t")


def write_fits_BinTable_file(
    cols,
    output_path,
    R=None,
    R_shear=None,
    R_select=None,
    c=None,
):
    """Write Fits Bin Table File.

    Write columns to FITS file as BinaryTable

    Parameters
    ----------
    cols : list of fits.Column
        column data
    output_path : str
        output file path
    R : np.matrix(2, 2), optional
        total response matrix
    R_shear : np.matrix(2, 2), optional
        shear response matrix
    R_select : np.matrix(2, 2), optional
        selection response matrix
    c : np.array(2), optional
        additive bias components

    """
    table_hdu = fits.BinTableHDU.from_columns(cols)

    # Primary HDU with information in header
    primary_header = fits.Header()
    primary_header = write_header_info_sp(primary_header)
    if R is not None:
        add_shear_bias_to_header(primary_header, R, R_shear, R_select, c)
    primary_hdu = fits.PrimaryHDU(header=primary_header)

    hdu_list = fits.HDUList([primary_hdu, table_hdu])
    hdu_list.writeto(output_path, overwrite=True)


def bin_edges2centers(bin_edges):
    """Bin Edges To Centers.

    Transform bin edge values to central values.

    Parameters
    ----------
    bin_edges : list
        bin edge values

    Returns
    -------
    list
        bin central values

    """
    bin_means = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_means


def read_dndz(file_path):
    """Read Dndz.

    Read redshift histogram from file.

    Parameters
    ----------
    file_path : str
        input file path

    Returns
    -------
    list :
        redshift bin centers
    list :
        number densities
    list :
        redshift bin edges

    """
    dat = ascii.read(file_path, format="commented_header")

    # Remove last n(z) value which is zero, to match bin centers
    nz = dat["dn_dz"][:-1]
    z_edges = dat["z"]
    z_centers = bin_edges2centers(z_edges)

    return z_centers, nz, z_edges
