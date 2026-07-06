"""COSMO.

:Name: cosmo.py

:Description: This file contains methods for cosmological quantities:
              lensing critical surface mass density (``sigma_crit`` family)
              and shear two-point theory predictions. The theory-curve
              machinery — flexible cosmology construction (``get_cosmo``),
              angular power spectra and correlation functions
              (``get_theo_c_ell``, ``c_ell_to_xi``, ``get_theo_xi``), and
              CCL/CAMB/CosmoCov parameter conversion — was consolidated here
              from ``sp_validation.cosmology`` so that ``cs_util`` is the
              single home for the collaboration's cosmology primitives.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>
          Axel Guinot
          Cail Daley
          Sacha Guerrini
"""

import itertools

import camb
import numpy as np

from astropy import constants
from astropy import units
from astropy.cosmology import Planck18

import pyccl as ccl


def get_cosmo_default():
    """Get Cosmo Default.

    Return default cosmology.

    Returns
    -------
    Cosmology
        pyccl cosmology object

    """
    cos = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
    )

    # Set ell_max to large value, for spline interpolation (in integral over
    # C _ell to get real-space correlation functions). Avoid aliasing
    # ( oscillations)
    cos.cosmo.spline_params.ELL_MAX_CORR = 10_000_000
    cos.cosmo.spline_params.N_ELL_CORR = 5_000

    return cos


def sigma_crit(z_lens, z_source, cos, d_lens=None, d_source=None):
    """Sigma Crit.

    Critical surface mass density.

    Parameters
    ----------
    z_lens : float
        lens redshift
    z_source : float
        source redshift
    cos : pyccl.core.Cosmology
        cosmological parameters
    d_lens : astropy.units.Quantity, optional
        precomputed anguar diameter distance to lens, computed from z_lens
        if ``None`` (default)
    d_source : astropy.units.Quantity, optional
        precomputed anguar diameter distance to sourcce, computed from z_source
        if ``None`` (default)

    Returns
    -------
    astropy.units.Quantity
        critical surface mass density with units of M_sol / pc^2

    """
    unit_return = units.Msun / units.pc**2

    # Return 0 if lens behind source
    if z_lens >= z_source:
        return 0.0 * unit_return

    a_lens = 1 / (1 + z_lens)
    a_source = 1 / (1 + z_source)
    if d_lens is None:
        d_lens = cos.angular_diameter_distance(a_lens) * units.Mpc
    if d_source is None:
        d_source = cos.angular_diameter_distance(a_source) * units.Mpc

    d_lens_source = cos.angular_diameter_distance(a_lens, a_source) * units.Mpc

    frac = d_source / (d_lens_source * d_lens)
    pref = constants.c**2 / (4 * np.pi * constants.G)

    sigma_cr = (pref * frac).to(unit_return)

    return sigma_cr


def sigma_crit_eff(
    z_lens,
    z_source_arr,
    nz_source_arr,
    cos,
    d_lens=None,
    d_source_arr=None,
):
    """Sigma Crit Eff.

    Effective critical surface mass density, which
    is sigma_crit(z_lens, z_source) weighted by nz_source.

    Parameters
    ----------
    z_lens : float
        lens redshift
    z_source_arr : list
        source redshifts
    nz_source_arr : list
        number of galaxies at z_source
    cos : pyccl.core.Cosmology
        cosmological parameters
    d_lens : astropy.units.Quantity, optional
        precomputed anguar diameter distance to lens;
        computed from z_lens if ``None`` (default)
    d_source_arr : list, optional
        precompuated angular diameter distances to sources;
        computed from z_source_arr if ``None`` (default);
        needs to be list of astropy.units.Quantity

    Raises
    ------
    IndexError
        If lists ``z_source_arr``, ``nz_source_arr``, and d_source_arr
        do not match

    Returns
    -------
    astropy.units.Quantity
        effective critical surface mass density with units of M_sol / pc^2

    """
    n_source = len(z_source_arr)

    if d_source_arr is None:
        d_source_arr = [None] * n_source

    if (len(nz_source_arr) != n_source) or (len(d_source_arr) != n_source):
        raise IndexError(
            "Lists for source z, n(z), and/or d_ang have different lenghts"
        )

    sigma_cr_arr = []
    for idx in range(n_source):
        sigma_cr = sigma_crit(
            z_lens,
            z_source_arr[idx],
            cos,
            d_lens=d_lens,
            d_source=d_source_arr[idx],
        )

        # Get unit
        if len(sigma_cr_arr) == 0:
            unit = sigma_cr.unit

        sigma_cr_arr.append(sigma_cr.value)

    # Mean sigma_cr weighted by source redshifts.
    # np.average can only deal with unitless quantities.
    sigma_cr_eff = np.average(sigma_cr_arr, weights=nz_source_arr)

    return sigma_cr_eff * unit


def sigma_crit_m1_eff(
    z_lens,
    z_source_arr,
    nz_source_arr,
    cos,
    d_lens=None,
    d_source_arr=None,
):
    """Sigma Crit M1 Eff.

    Effective inverse critical surface mass density, which
    is sigma_crit^{-1}(z_lens, z_source) weighted by nz_source.
    See Eq. (17) in :cite:`2004AJ....127.2544S`.

    Parameters
    ----------
    z_lens : float
        lens redshift
    z_source_arr : list
        source redshifts
    nz_source_arr : list
        number of galaxies at z_source
    cos : pyccl.core.Cosmology
        cosmological parameters
    d_lens : astropy.units.Quantity, optional
        precomputed anguar diameter distance to lens;
        computed from z_lens if ``None`` (default)
    d_source_arr : float, optional
        precomputed anguar diameter distance to sources;
        computed from z_source_arr if ``None`` (default);
        needs to be list of astropy.units.Quantity

    Raises
    ------
    IndexError
        If lists ``z_source_arr``, ``nz_source_arr``, and ``d_source_arr``
        do not match

    Returns
    -------
    astropy.units.Quantity
        effective inverse critical surface mass density with units of
        M_sol / pc^2

    """
    n_source = len(z_source_arr)

    if d_source_arr is None:
        d_source_arr = [None] * n_source

    if (len(nz_source_arr) != n_source) or (len(d_source_arr) != n_source):
        raise IndexError(
            "Lists for source z, n(z), and/or d_ang have different lenghts"
        )

    sigma_cr_m1_arr = []
    weights = []

    for idx in range(n_source):
        sigma_cr = sigma_crit(
            z_lens,
            z_source_arr[idx],
            cos,
            d_lens=d_lens,
            d_source=d_source_arr[idx],
        )

        # Get unit
        if len(sigma_cr_m1_arr) == 0:
            unit = 1 / sigma_cr.unit

        # If lens behind source: continue
        if sigma_cr == 0:
            continue

        sigma_cr_m1 = 1 / sigma_cr

        sigma_cr_m1_arr.append(sigma_cr_m1.value)
        weights.append(nz_source_arr[idx])

    sigma_cr_m1_eff = np.average(sigma_cr_m1_arr, weights=weights)

    return sigma_cr_m1_eff * unit


def xipm_theo(
    theta,
    cos,
    z,
    dndz,
):
    """Xipm Theo.

    Return theoretical prediction of the shear two-point correlation function.

    Parameters
    ----------
    theta : list
        angular scales, list of type astropy.units.Quantity
    cos : pyccl.core.Cosmology
        cosmological parameters
    z : list
        redshift centers
    dndz : list
        number of galaxies for each z (arbitrary normalisation)

    Returns
    -------
    numpy.ndarray
        xi_+
    numpy.ndarray
        xi_-

    """
    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)
    lens_tr = ccl.WeakLensingTracer(cos, dndz=(z, dndz))

    # Calculate the angular cross-spectrum of the two tracers as a function
    # of ell
    # MKDEBUG TODO: vary, use unions-shear-ustc-cea/unions_wl/defaults.py
    ell = np.logspace(0, np.log10(10000), 1000)
    cl = ccl.angular_cl(cos, lens_tr, lens_tr, ell)

    method = "Bessel"

    xipm = {}
    for corr_type in ("GG+", "GG-"):
        xipm[corr_type] = ccl.correlation(
            cos,
            ell=ell,
            C_ell=cl,
            theta=theta.to("deg"),
            type=corr_type,
        )

    return xipm["GG+"], xipm["GG-"]


# ============================================================================
# Theory-curve machinery consolidated from sp_validation.cosmology
# (flexible cosmology construction, C_ell / xi_pm predictions, and
# CCL/CAMB/CosmoCov parameter conversion). cs_util is the single home.
# ============================================================================
# =============================================================================
# Fiducial Cosmology: astropy Planck18
# =============================================================================
# Source: Planck 2018 Paper VI, Table 2 (TT,TE,EE+lowE+lensing+BAO)
# Reference: Planck Collaboration 2020, A&A, 641, A6
#
# Note on sigma_8 / A_s consistency:
# CAMB with A_s=2.1e-9 and m_nu=0.06 eV derives sigma_8 ~ 0.806, not 0.8102.
# This ~0.5% difference arises from Planck's MCMC marginalization details.
# Policy: Use sigma_8=0.8102 for codes taking sigma_8 directly (CosmoCov, CCL);
#         use A_s=2.1e-9 for CAMB-based predictions.
# =============================================================================
PLANCK18 = {
    "Omega_m": Planck18.Om0,  # 0.30966
    # Flat-universe dark-energy density (1 - Omega_m = 0.69034, matching the
    # paper-era planck18.json and CosmoCov's flat-LCDM convention; astropy's
    # Ode0 = 0.68885 differs by the radiation/neutrino contributions).
    "Omega_v": 1.0 - Planck18.Om0,  # 0.69034
    "Omega_b": Planck18.Ob0,  # 0.04897
    "h": Planck18.h,  # 0.6766
    "n_s": Planck18.meta["n"],  # 0.9665
    "sigma_8": Planck18.meta["sigma8"],  # 0.8102
    "A_s": 2.1e-9,  # ln(10^10 A_s) = 3.047
    "m_nu": 0.06,  # eV, sum of neutrino masses
    "w0": -1.0,
    "wa": 0.0,
}


def _ccl_to_camb(cosmo):
    """Convert CCL cosmology object to CAMB parameter format.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object

    Returns
    -------
    dict
        CAMB parameters dictionary with As properly set
    """

    h = cosmo["h"]
    camb_params = {
        "H0": h * 100,
        "ombh2": cosmo["Omega_b"] * h**2,
        "omch2": cosmo["Omega_c"] * h**2,
        "ns": cosmo["n_s"],
    }

    # Handle normalization: prefer As, but convert sigma8 to As if needed
    As_val = cosmo.__getitem__("A_s")
    sigma8_val = cosmo.__getitem__("sigma8")

    if As_val is not None and not np.isnan(As_val):
        # Use As directly
        camb_params["As"] = As_val
    elif sigma8_val is not None:
        # Convert sigma8 to As using iterative CAMB calculation
        # see https://cosmocoffee.info/viewtopic.php?t=475
        As_fiducial = 2e-9  # Standard fiducial value

        # Step 1: Calculate current sigma8 with fiducial As
        temp_params = camb_params.copy()
        temp_params["As"] = As_fiducial

        pars = camb.set_params(**temp_params)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        sigma8_current = results.get_sigma8_0()

        # Step 2: Scale As to match target sigma8
        # As scales as sigma8^2
        As_scaled = As_fiducial * (sigma8_val / sigma8_current) ** 2
        camb_params["As"] = As_scaled

        # Step 3: Verify the result
        temp_params["As"] = As_scaled
        pars = camb.set_params(**temp_params)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        sigma8_final = results.get_sigma8_0()

        # Check accuracy (warn if >1% difference)
        relative_error = abs(sigma8_final - sigma8_val) / sigma8_val
        if relative_error > 0.01:
            print(
                f"Warning: CAMB sigma8 conversion accuracy: target={sigma8_val:.4f}, "
                f"achieved={sigma8_final:.4f}, error={relative_error:.1%}"
            )
    else:
        # No normalization specified, use CAMB default
        pass

    # Add dark energy parameters if they exist
    for camb_key, cosmo_key in [("w", "w0"), ("wa", "wa")]:
        if hasattr(cosmo._params, cosmo_key):
            camb_params[camb_key] = cosmo[cosmo_key]

    return camb_params


def _camb_to_ccl(camb_params):
    """Convert CAMB parameter format to CCL parameter dictionary.

    Parameters
    ----------
    camb_params : dict
        CAMB parameters dictionary with required keys:
        H0, ombh2, omch2, ns, and either As or sigma8

    Returns
    -------
    dict
        CCL parameters dictionary
    """
    h = camb_params["H0"] / 100.0
    ccl_params = {
        "Omega_c": camb_params["omch2"] / h**2,
        "Omega_b": camb_params["ombh2"] / h**2,
        "h": h,
        "n_s": camb_params["ns"],
        **{
            k: camb_params[v]
            for k, v in [("w0", "w"), ("wa", "wa")]
            if v in camb_params
        },
    }

    # CCL accepts either A_s or sigma8 directly
    if "As" in camb_params:
        ccl_params["A_s"] = camb_params["As"]
    elif "sigma8" in camb_params:
        ccl_params["sigma8"] = camb_params["sigma8"]
    else:
        raise ValueError("Must provide either 'As' or 'sigma8' in camb_params")

    return ccl_params


def _cosmocov_to_ccl(cosmocov_params):
    """Convert CosmoCov parameter format to CCL parameter dictionary.

    Parameters
    ----------
    cosmocov_params : dict
        CosmoCov parameters dictionary with required keys:
        Omega_m, omb, h0, sigma_8, n_spec

    Returns
    -------
    dict
        CCL parameters dictionary
    """
    required_params = ["Omega_m", "omb", "h0", "n_spec", "sigma_8"]
    missing_params = [p for p in required_params if p not in cosmocov_params]
    if missing_params:
        raise KeyError(f"Missing required cosmological parameters: {missing_params}")

    ccl_params = {
        "Omega_c": cosmocov_params["Omega_m"] - cosmocov_params["omb"],
        "Omega_b": cosmocov_params["omb"],
        "h": cosmocov_params["h0"],
        "sigma8": cosmocov_params["sigma_8"],
        "n_s": cosmocov_params["n_spec"],
        **{k: cosmocov_params[k] for k in ["w0", "wa"] if k in cosmocov_params},
    }

    return ccl_params


def get_cosmo(
    Omega_b=None,
    Omega_m=None,
    h=None,
    sig8=None,
    ns=None,
    w0=None,
    wa=None,
    mnu=None,
    transfer_function="boltzmann_camb",
    matter_power_spectrum="halofit",
    cosmocov_params=None,
    camb_params=None,
    extra_params=None,
):
    """Get CCL cosmology object with user-specified parameters.

    Defaults to astropy Planck18 cosmology (Table 2: TT,TE,EE+lowE+lensing+BAO).
    Can also use CosmoCov or CAMB parameter formats.

    Parameters
    ----------
    Omega_m : float, default=None
        Matter density parameter (defaults to Planck18: 0.30966)
    Omega_b : float, default=None
        Baryon density parameter (defaults to Planck18: 0.04897)
    h : float, default=None
        Reduced Hubble constant (defaults to Planck18: 0.6766)
    sig8 : float, default=None
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck18: 0.8102)
    ns : float, default=None
        Scalar spectral index (defaults to Planck18: 0.9665)
    w0 : float, default=None
        Dark energy equation of state parameter (defaults to -1.0)
    wa : float, default=None
        Dark energy equation of state evolution parameter (defaults to 0.0)
    mnu : float, default=None
        Total neutrino mass in eV (defaults to 0.06 eV)
    transfer_function : str, default='boltzmann_camb'
        Transfer function to use
    matter_power_spectrum : str, default='halofit'
        Matter power spectrum to use
    cosmocov_params : dict, optional
        Parameters in CosmoCov format (Omega_m, omb, h0, sigma_8, n_spec)
        If provided, entries override above parameters. Mutually exclusive with
        camb_params.
    camb_params : dict, optional
        Parameters in CAMB format (H0, ombh2, omch2, ns, sigma8)
        If provided, entries override above parameters. Mutually exclusive with
        cosmocov_params.
    extra_params : dict, optional
        Additional parameters to pass to CCL (e.g., for CAMB non-linear settings)

    Returns
    -------
    Cosmology
        pyccl cosmology object
    """
    # Check for parameter format conflicts
    if cosmocov_params is not None and camb_params is not None:
        raise ValueError(
            "Cannot provide both cosmocov_params and camb_params. Choose one format."
        )

    # Convert parameters to CCL format
    if cosmocov_params is not None:
        print("Using CosmoCov parameters to create CCL cosmology.")
        ccl_params = _cosmocov_to_ccl(cosmocov_params)
    elif camb_params is not None:
        print("Using CAMB parameters to create CCL cosmology.")
        ccl_params = _camb_to_ccl(camb_params)
    else:
        ccl_params = {}

    # Planck 2018 defaults from astropy (see PLANCK18 dict at module level)
    planck_defaults = {
        "Omega_m": PLANCK18["Omega_m"],
        "Omega_b": PLANCK18["Omega_b"],
        "h": PLANCK18["h"],
        "sig8": PLANCK18["sigma_8"],
        "ns": PLANCK18["n_s"],
        "w0": PLANCK18["w0"],
        "wa": PLANCK18["wa"],
        "mnu": PLANCK18["m_nu"],
    }

    mnu = ccl_params.get("mnu", mnu or planck_defaults["mnu"])
    h = ccl_params.get("h", h or planck_defaults["h"])
    pars = camb.CAMBparams()

    pars.set_cosmology(
        mnu=mnu,
        H0=h * 100,
    )

    Omega_nu = pars.omeganu

    combined_params = {
        "Omega_c": ccl_params.get(
            "Omega_c",
            (Omega_m or planck_defaults["Omega_m"])
            - (Omega_b or planck_defaults["Omega_b"])
            - Omega_nu,
        ),
        "Omega_b": ccl_params.get("Omega_b", Omega_b or planck_defaults["Omega_b"]),
        "h": h,
        "sigma8": ccl_params.get("sigma8", sig8 or planck_defaults["sig8"]),
        "n_s": ccl_params.get("n_s", ns or planck_defaults["ns"]),
        "w0": ccl_params.get("w0", w0 or planck_defaults["w0"]),
        "wa": ccl_params.get("wa", wa or planck_defaults["wa"]),
        "m_nu": mnu,
        "extra_parameters": extra_params,
    }

    return ccl.Cosmology(
        **combined_params,
        transfer_function=transfer_function,
        matter_power_spectrum=matter_power_spectrum,
    )


def get_theo_c_ell(
    ell,
    z,
    nz,
    backend="ccl",
    cosmo=None,
    Omega_b=None,
    Omega_m=None,
    h=None,
    sig8=None,
    ns=None,
    w0=None,
    wa=None,
):
    """Calculate theoretical angular power spectrum C_ell for weak lensing.

    Parameters
    ----------
    ell : array
        Multipole moments (e.g., np.arange(2, 2000))
    z : array
        Redshifts for n(z) distribution
    nz : array
        n(z) redshift distribution. If nz.shape[1] > 1, assumes multiple tomographic bins.
    backend : str, default="ccl"
        Backend to use: "ccl" or "camb"
    cosmo : ccl.Cosmology, optional
        CCL cosmology object. If None, will create using individual parameters.
    Omega_b : float, optional
        Baryon density parameter (defaults to Planck 2018)
    Omega_m : float, optional
        Matter density parameter (defaults to Planck 2018)
    h : float, optional
        Reduced Hubble constant (defaults to Planck 2018)
    sig8 : float, optional
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck 2018)
    ns : float, optional
        Scalar spectral index (defaults to Planck 2018)
    w0 : float, optional
        Dark energy equation of state parameter (defaults to -1.0)
    wa : float, optional
        Dark energy equation of state evolution parameter (defaults to 0.0)

    Returns
    -------
    cl : dict
        Angular power spectrum in dictionnary indexed by the tomographic bin keys.
    """
    if cosmo is None:
        cosmo = get_cosmo(
            Omega_b=Omega_b,
            Omega_m=Omega_m,
            h=h,
            sig8=sig8,
            ns=ns,
            w0=w0,
            wa=wa,
        )

    n_tomo_bins = nz.shape[1] if len(nz.shape) > 1 else 1
    tomo_bin_pairs = list(itertools.combinations_with_replacement(range(1, n_tomo_bins + 1), 2))
    cl = {}

    if backend == "ccl":
        tracers = {}

        # Create lensing tracer
        for bin_key in range(1, n_tomo_bins + 1):
            tracers[f"W{bin_key}"]= ccl.WeakLensingTracer(cosmo, dndz=(z, nz[:, bin_key]))

        for bin_key1, bin_key2 in tomo_bin_pairs:
            cl[f"W{bin_key1}xW{bin_key2}"] = ccl.angular_cl(
                cosmo, tracers[f"W{bin_key1}"], tracers[f"W{bin_key2}"], ell
            )

    elif backend == "camb":
        # Convert CCL cosmology to CAMB parameters
        import camb

        camb_kwargs = _ccl_to_camb(cosmo)

        # Set up CAMB parameters
        pars = camb.set_params(
            **camb_kwargs,
            WantTransfer=True,
            NonLinear=camb.model.NonLinear_both,
        )

        # Adjust for neutrino contribution
        if "mnu" in camb_kwargs and camb_kwargs["mnu"] > 0:
            omch2_adj = camb_kwargs["omch2"] - pars.omeganu * (pars.H0 / 100) ** 2
            pars.set_cosmology(omch2=omch2_adj)

        # Set up lensing source window
        pars.min_l = ell.min()
        pars.set_for_lmax(ell.max())
        if len(nz.shape) == 1:
            pars.SourceWindows = [
                camb.sources.SplinedSourceWindow(z=z, W=nz, source_type="lensing")
            ]
        else:
            pars.SourceWindows = [
                camb.sources.SplinedSourceWindow(z=z, W=nz[:, i], source_type="lensing")
                for i in range(nz.shape[1])
            ]

        # Calculate power spectrum
        results = camb.get_results(pars)
        theory_cls = results.get_source_cls_dict(lmax=ell.max(), raw_cl=True)

        # Interpolate to match input ell array
        # CAMB returns C_ell for ell = 0, 1, 2, ..., lmax
        ell_camb = np.arange(len(theory_cls["W1xW1"]))
        for bin_key1, bin_key2 in tomo_bin_pairs:
            cl_full = theory_cls[f"W{bin_key1}xW{bin_key2}"]
            cl[f"W{bin_key1}xW{bin_key2}"] = np.interp(ell, ell_camb, cl_full)

    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'ccl' or 'camb'")

    return cl


def c_ell_to_xi(cosmo, theta, ell, cl):
    """Convert angular power spectrum to correlation functions using CCL.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object (used for correlation function calculation)
    theta : array
        Angular separations in arcminutes
    ell : array
        Multipole moments
    cl : array
        Angular power spectrum

    Returns
    -------
    xip, xim : arrays
        xi+ and xi- correlation functions
    """
    theta_deg = theta / 60.0  # arcmin to degrees

    xip = ccl.correlation(
        cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG+", method="Bessel"
    )
    xim = ccl.correlation(
        cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG-", method="Bessel"
    )

    return xip, xim


def get_theo_xi(
    theta,
    z,
    nz,
    Omega_m=None,
    h=None,
    Omega_b=None,
    sig8=None,
    ns=None,
    ell_min=10,
    ell_max=20000,
    n_ell=500,
    backend="ccl",
    cosmo=None,
    **cosmo_kwargs,
):
    """Calculate theoretical xi+/xi- using individual parameters.

    Parameters
    ----------
    theta : array
        Angular separations in arcminutes
    z : array
        Redshift array
    nz : array
        n(z) redshift distribution. If nz.shape[1] > 1, assumes multiple tomographic bins.
    Omega_m : float, default=None
        Matter density parameter (defaults to Planck 2018)
    h : float, default=None
        Reduced Hubble constant (defaults to Planck 2018)
    Omega_b : float, default=None
        Baryon density parameter (defaults to Planck 2018)
    sig8 : float, default=None
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck 2018)
    ns : float, default=None
        Scalar spectral index (defaults to Planck 2018)
    ell_min : int, default=0
        Minimum ell for power spectrum calculation
    ell_max : int, default=20000
        Maximum ell for power spectrum calculation
    n_ell : int, default=500
        Number of ell bins
    backend : str, default="ccl"
        Backend to use: "ccl" or "camb"
    **cosmo_kwargs
        Additional arguments passed to backend

    Returns
    -------
    dict
        Theoretical xi+ and xi- correlation functions per tomographic bin combinations
    """
    # Create ell array for C_ell calculation
    ell = np.geomspace(ell_min, ell_max, n_ell)

    # Use provided cosmology or create from parameters
    if cosmo is None:
        cosmo = get_cosmo(
            Omega_m=Omega_m, Omega_b=Omega_b, h=h, sig8=sig8, ns=ns, **cosmo_kwargs
        )

    # Calculate C_ell
    cl = get_theo_c_ell(ell, z, nz, backend=backend, cosmo=cosmo)

    # Convert to xi
    return {k: c_ell_to_xi(cosmo, theta, ell, v) for k, v in cl.items()}
