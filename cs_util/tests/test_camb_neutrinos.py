"""Regression tests for the CCL-to-CAMB cosmology bridge."""

import camb
import numpy as np
import numpy.testing as npt
import pyccl as ccl
import pytest

from cs_util import cosmo


@pytest.mark.parametrize(
    "masses, split",
    [
        (0.0, "normal"),
        (0.06, "normal"),
        (0.3, "inverted"),
        (0.3, "equal"),
        ([0.01, 0.03, 0.08], "list"),
    ],
)
def test_ccl_to_camb_species(masses, split):
    """Preserve resolved masses, temperatures and CDM, not CAMB defaults."""
    c = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.96,
        A_s=2.1e-9,
        m_nu=masses,
        mass_split=split,
        Neff=3.2,
        T_CMB=2.73,
        Omega_k=0.01,
        w0=-0.9,
        wa=0.1,
    )
    kwargs = cosmo._ccl_to_camb(c)
    p = camb.set_params(**kwargs)
    assert kwargs["mnu"] == pytest.approx(np.sum(c["m_nu"]))
    assert p.num_nu_massive == c["N_nu_mass"]
    assert p.omch2 == pytest.approx(c["Omega_c"] * c["h"] ** 2)
    assert p.ombh2 == pytest.approx(c["Omega_b"] * c["h"] ** 2)
    assert p.omnuh2 == pytest.approx(c["Omega_nu_mass"] * c["h"] ** 2)
    assert p.num_nu_massless == pytest.approx(c["N_nu_rel"])
    assert p.N_eff == pytest.approx(c["Neff"])
    assert p.TCMB == c["T_CMB"]
    assert p.omk == c["Omega_k"]
    assert p.DarkEnergy.w == c["w0"]
    assert p.DarkEnergy.wa == c["wa"]
    if kwargs["mnu"]:
        npt.assert_allclose(p.nu_mass_fractions, np.array(c["m_nu"]) / kwargs["mnu"])


def test_ccl_to_camb_scalar_mass():
    """Also accept a scalar mass representation."""
    c = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.96,
        A_s=2.1e-9,
        m_nu=[0.3],
        mass_split="list",
    )
    keys = (
        "h",
        "Omega_b",
        "Omega_c",
        "n_s",
        "Neff",
        "T_CMB",
        "Omega_k",
        "w0",
        "wa",
        "T_ncdm",
        "Omega_nu_mass",
        "N_nu_rel",
        "A_s",
        "sigma8",
    )
    params = {key: c[key] for key in keys}
    params["m_nu"] = 0.3
    kwargs = cosmo._ccl_to_camb(params)
    assert kwargs["mnu"] == pytest.approx(0.3)
    assert kwargs["num_massive_neutrinos"] == 1


def test_ccl_to_camb_sigma8_background():
    """Normalize on the requested neutrino and dark-energy background."""
    c = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.96,
        sigma8=0.8,
        m_nu=0.3,
        w0=-0.9,
        wa=0.1,
    )
    p = camb.set_params(**cosmo._ccl_to_camb(c))
    p.set_matter_power(redshifts=[0], kmax=2)
    assert camb.get_results(p).get_sigma8_0() == pytest.approx(0.8, rel=1e-3)


def test_camb_neutrino_response():
    """The mass response agrees even though the nonlinear models differ."""
    ell = np.array([100, 500, 1000])
    z = np.linspace(0.01, 3, 300)
    nz = np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2)
    spectra = {}
    for mass in (0.06, 0.3):
        c = cosmo.get_cosmo(mnu=mass)
        spectra[mass] = {
            backend: cosmo.get_theo_c_ell(ell, z, nz, backend=backend, cosmo=c)["W1xW1"]
            for backend in ("camb", "ccl")
        }
    response_camb = spectra[0.3]["camb"] / spectra[0.06]["camb"]
    response_ccl = spectra[0.3]["ccl"] / spectra[0.06]["ccl"]
    assert abs(response_camb[-1] - 1) > 0.005
    npt.assert_allclose(response_camb, response_ccl, rtol=0.02)
