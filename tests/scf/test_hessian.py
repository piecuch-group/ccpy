"""Numerical Hessians and harmonic frequencies (ccpy.gradients.hessian)."""
import numpy as np
import pytest

from ccpy import Molecule, RHF
from ccpy.gradients.hessian import (
    AU_TO_WAVENUMBER,
    ISOTOPE_MASSES,
    fd_hessian,
    harmonic_analysis,
    isotope_masses,
)

WATER = "O 0.0 0.0 0.1196; H 0.0 0.7563 -0.4785; H 0.0 -0.7563 -0.4785"


def _rhf_energy(mol):
    return RHF(mol, verbose=0, e_tol=1.0e-13, d_tol=1.0e-11).kernel()


def _rhf_gradient(mol):
    mf = RHF(mol, verbose=0, e_tol=1.0e-13, d_tol=1.0e-11)
    mf.kernel()
    return mf.gradient()


def test_against_pyscf_analytic_rhf_hessian():
    gto = pytest.importorskip("pyscf.gto")
    from pyscf import scf
    from pyscf.data.elements import COMMON_ISOTOPE_MASSES
    from pyscf.hessian import thermo

    assert np.allclose(ISOTOPE_MASSES, COMMON_ISOTOPE_MASSES[1:37])
    pmol = gto.M(atom=WATER, basis="6-31g", verbose=0)
    pmf = scf.RHF(pmol).run(conv_tol=1.0e-13)
    h4 = pmf.Hessian().kernel()
    h_ref = h4.transpose(0, 2, 1, 3).reshape(9, 9)
    geometry = "; ".join(f"{pmol.atom_symbol(i)} {x:.12f} {y:.12f} {z:.12f}"
                         for i, (x, y, z) in enumerate(pmol.atom_coords()))
    mol = Molecule(geometry, "6-31g", unit="bohr", cholesky_tei=True, cholesky_tol=1.0e-12, verbose=False)

    # the harmonic analysis alone, on PySCF's own Hessian, with the same masses
    ref = thermo.harmonic_analysis(pmol, h4, mass=isotope_masses(mol))
    ours = harmonic_analysis(h_ref, mol, verbose=False)
    assert np.allclose(ours["frequencies"], ref["freq_wavenumber"].real, atol=1.0e-3)

    # the finite-difference Hessian from native-SCF energies
    H = fd_hessian(mol, energy=_rhf_energy, verbose=False)
    assert np.abs(H - h_ref).max() < 1.0e-5
    freqs = harmonic_analysis(H, mol, verbose=False)["frequencies"]
    assert np.allclose(freqs, ref["freq_wavenumber"].real, atol=0.05)


def test_gradient_and_energy_hessians_agree():
    mol = Molecule(WATER, "6-31g", auxiliary_basis="def2-universal-jkfit", verbose=False)
    H_e = fd_hessian(mol, energy=_rhf_energy, verbose=False)
    H_g = fd_hessian(mol, gradient=_rhf_gradient, verbose=False)
    assert np.abs(H_e - H_g).max() < 1.0e-5
    f_e = harmonic_analysis(H_e, mol, verbose=False)["frequencies"]
    f_g = harmonic_analysis(H_g, mol, verbose=False)["frequencies"]
    assert np.allclose(f_e, f_g, atol=0.05)


def test_diatomic_frequency_matches_curvature():
    """One vibration of HF, against sqrt(k/mu) from a 1D fit of E(R) (the HF 6-31G minimum)."""
    make = lambda R: Molecule(f"H 0 0 0; F 0 0 {R}", "6-31g", unit="bohr", cholesky_tei=True,
                              cholesky_tol=1.0e-12, verbose=False)
    R0 = 1.75
    Rs = R0 + np.linspace(-0.05, 0.05, 9)
    p = np.polyfit(Rs - R0, [_rhf_energy(make(R)) for R in Rs], 6)
    roots = np.roots(np.polyder(p)).real
    x = roots[np.argmin(np.abs(roots))]
    m = isotope_masses(make(R0))
    w_fit = np.sqrt(np.polyval(np.polyder(p, 2), x) / (m[0] * m[1] / m.sum())) * AU_TO_WAVENUMBER
    mol = make(R0 + x)
    res = harmonic_analysis(fd_hessian(mol, energy=_rhf_energy, verbose=False), mol, verbose=False)
    assert len(res["frequencies"]) == 1          # 5 rigid-body modes for a linear molecule
    assert abs(res["frequencies"][0] - w_fit) < 0.05
