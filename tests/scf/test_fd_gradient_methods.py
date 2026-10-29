"""Finite-difference nuclear gradients (ccpy.gradients.fd_gradient) across ccpy methods.

Every method is run from scratch (native SCF + Cholesky integrals, C1) at each displaced
geometry. A method passes when its 4-point FD gradient is
  * smooth: the gradients with steps of 1e-3 and 2e-3 bohr agree to TOL_STEP, and
  * invariant: the net force and net torque vanish to TOL_INVARIANCE.
The CCSD FD gradient is also compared with the analytic one.

Tests marked ``short`` take seconds; the unmarked ones (EOMCCSDt / EOM-CC(t;3), DEA-EOM(4p-2h),
DIP-EOM(4h-2p)) take 1-3 minutes each. Run ``pytest -m short`` to skip them.

Adaptive CC(P;Q) must use the same P spaces at every displaced geometry: the reference-
geometry run records them (AdaptDriver.pspaces) and every displaced run is given them with
pspaces=. Re-selecting at each geometry swaps triples near the cutoff between displacements
and makes the energy discontinuous.

Known failure (strict xfail):
  * CR-CC(2,3)_D: the energy changes under rotations among degenerate orbitals (the
    D-variant denominators use diagonal Hbar elements), so it is not a well-defined
    function of geometry when orbitals are degenerate (e.g. pi orbitals of a linear molecule).
"""
import contextlib
import io

import numpy as np
import pytest

from ccpy import AdaptDriver, AdaptEOMDriverSS, Driver, Molecule, RHF, ROHF, get_active_triples_pspace
from ccpy.gradients import fd_gradient

TOL_STEP = 1.0e-06
TOL_INVARIANCE = 1.0e-06

H2O = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"
NH2 = "N 0.0 0.05 0.1173; H 0.0 0.8572 -0.5692; H 0.1 -0.8572 -0.5692"
CH2 = "C 0.0 0.0 0.0; H 0.0 1.644403 -1.32213; H 0.05 -1.644403 -1.37213"  # bohr
HF = "F 0.0 0.0 0.0; H 0.0 0.0 1.7328795"  # bohr, linear: degenerate pi orbitals


def _molecule(geometry, charge=0, multiplicity=1, unit="angstrom", cartesian=False):
    return Molecule(geometry, "6-31g", unit=unit, charge=charge, multiplicity=multiplicity,
                    cartesian=cartesian, cholesky_tei=True, cholesky_tol=1.0e-12)


MOLECULES = {
    "h2o": lambda: _molecule(H2O),
    "nh2": lambda: _molecule(NH2, multiplicity=2),
    "nh2+": lambda: _molecule(NH2, charge=1),
    "ch2++": lambda: _molecule(CH2, charge=2, unit="bohr", cartesian=True),
    "ch2--": lambda: _molecule(CH2, charge=-2, unit="bohr", cartesian=True),
    "hf": lambda: _molecule(HF, unit="bohr"),
}


def _driver(mol, nfrozen, scf=RHF):
    mf = scf(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=nfrozen)
    driver.options.update({"energy_convergence": 1.0e-12, "amp_convergence": 1.0e-10,
                           "maximum_iterations": 300})
    return driver


def _ecc(driver):
    return driver.system.reference_energy + driver.correlation_energy


# ---------------------------------------------------------------- energy callables
def ccsd(mol):
    d = _driver(mol, 1)
    d.run_cc(method="ccsd")
    return _ecc(d)


def crcc23(variant, scf=RHF):
    def energy(mol):
        d = _driver(mol, 1, scf)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        d.run_ccp3(method="crcc23")
        return _ecc(d) + d.deltap3[0][variant]
    return energy


def cct3(variant, scf=RHF):
    def energy(mol):
        d = _driver(mol, 1, scf)
        d.system.set_active_space(nact_occupied=1, nact_unoccupied=2)
        t3 = get_active_triples_pspace(d.system, target_irrep="A")
        d.run_ccp(method="ccsdt_p", t3_excitations=t3)
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        d.run_ccp3(method="cct3")
        return _ecc(d) + d.deltap3[0][variant]
    return energy


def eomccsd(state, variant=None):
    def energy(mol):
        d = _driver(mol, 1)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        d.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=4, roots_per_irrep={"A": 3}, multiplicity=1)
        d.run_eomcc(method="eomccsd", state_index=[state])
        e = _ecc(d) + d.vertical_excitation_energy[state]
        if variant:
            d.run_lefteomcc(method="left_ccsd", state_index=[state])
            d.run_ccp3(method="crcc23", state_index=[state])
            e += d.deltap3[state][variant]
        return e
    return energy


def eomcct3(state, variant):
    def energy(mol):
        d = _driver(mol, 1)
        d.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
        t3 = get_active_triples_pspace(d.system, target_irrep="A")
        r3 = get_active_triples_pspace(d.system, target_irrep="A")
        d.run_ccp(method="ccsdt_p", t3_excitations=t3)
        d.run_hbar(method="ccsdt_p", t3_excitations=t3)
        d.run_leftccp(method="left_ccsdt_p", t3_excitations=t3)
        d.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=4, roots_per_irrep={"A": 3}, multiplicity=1)
        d.run_eomccp(method="eomccsdt_p", state_index=state, t3_excitations=t3, r3_excitations=r3)
        d.run_lefteomccp(method="left_ccsdt_p", state_index=state, t3_excitations=t3, r3_excitations=r3)
        d.run_ccp3(method="cct3", state_index=[state], t3_excitations=t3, r3_excitations=r3)
        return _ecc(d) + d.vertical_excitation_energy[state] + d.deltap3[state][variant]
    return energy


def ipeom(method):
    def energy(mol):
        d = _driver(mol, 1)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=4, roots_per_irrep={"A": 3})
        d.run_ipeomcc(method=method, state_index=[0])
        return _ecc(d) + d.vertical_excitation_energy[0]
    return energy


def eaeom(method):
    def energy(mol):
        d = _driver(mol, 1)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=6, roots_per_irrep={"A": 3})
        d.run_eaeomcc(method=method, state_index=[0])
        return _ecc(d) + d.vertical_excitation_energy[0]
    return energy


def deaeom(method):
    def energy(mol):
        d = _driver(mol, 0)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_guess(method="deacis", multiplicity=-1, nact_unoccupied=5, roots_per_irrep={"A": 4}, use_symmetry=False)
        d.run_deaeomcc(method=method, state_index=[0])
        return _ecc(d) + d.vertical_excitation_energy[0]
    return energy


def dipeom(method):
    def energy(mol):
        d = _driver(mol, 0)
        d.options["RHF_symmetry"] = False
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_guess(method="dipcis", multiplicity=-1, nact_occupied=d.system.noccupied_alpha,
                    roots_per_irrep={"A": 4}, use_symmetry=False)
        d.run_dipeomcc(method=method, state_index=[0])
        return _ecc(d) + d.vertical_excitation_energy[0]
    return energy


def adaptive(mol, pspaces=None, percentages=(0.0, 1.0, 2.0)):
    d = _driver(mol, 1)
    ad = AdaptDriver(d, percentage=list(percentages), pspaces=pspaces)
    ad.options["energy_tolerance"] = 1.0e-14  # run every macroiteration
    ad.run()
    return ad


def adaptive_eom(mol, pspaces=None, percentages=(0.0, 1.0, 2.0)):
    d = _driver(mol, 1)
    ad = AdaptEOMDriverSS(d, state_index=1, roots_per_irrep={"A": 4}, multiplicity=1,
                          percentage=list(percentages), pspaces=pspaces)
    ad.options["energy_tolerance"] = 1.0e-14
    ad.run()
    return ad


# ---------------------------------------------------------------- helpers
def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _check_fd(energy, mol):
    """Return the h=1e-3 FD gradient after checking smoothness and invariance."""
    g1 = _quiet(fd_gradient, energy, mol, step=1.0e-3, verbose=False)
    g2 = _quiet(fd_gradient, energy, mol, step=2.0e-3, verbose=False)
    X0 = np.array([atom[1] for atom in mol.atoms])
    assert np.abs(g1 - g2).max() < TOL_STEP, f"FD gradient depends on the step: {np.abs(g1 - g2).max():.2e}"
    assert np.linalg.norm(g1.sum(axis=0)) < TOL_INVARIANCE, "nonzero net force"
    assert np.linalg.norm(np.cross(X0, g1).sum(axis=0)) < TOL_INVARIANCE, "nonzero net torque"
    return g1


short = pytest.mark.short

CASES = [
    pytest.param("crcc23_A", crcc23("A"), "h2o", marks=short),
    pytest.param("crcc23_D", crcc23("D"), "h2o", marks=short),
    pytest.param("rohf_crcc23_A", crcc23("A", ROHF), "nh2", marks=short),
    pytest.param("cct3_A", cct3("A"), "h2o", marks=short),
    pytest.param("cct3_D", cct3("D"), "h2o", marks=short),
    pytest.param("rohf_cct3_A", cct3("A", ROHF), "nh2", marks=short),
    pytest.param("eomccsd", eomccsd(1), "h2o", marks=short),
    pytest.param("creomcc23_A", eomccsd(1, "A"), "h2o", marks=short),
    pytest.param("creomcc23_D", eomccsd(1, "D"), "h2o", marks=short),
    pytest.param("eomcct3_A", eomcct3(1, "A"), "h2o"),
    pytest.param("ipeom2", ipeom("ipeom2"), "h2o", marks=short),
    pytest.param("ipeom3", ipeom("ipeom3"), "h2o", marks=short),
    pytest.param("eaeom2", eaeom("eaeom2"), "nh2+", marks=short),
    pytest.param("eaeom3", eaeom("eaeom3"), "nh2+", marks=short),
    pytest.param("deaeom3", deaeom("deaeom3"), "ch2++", marks=short),
    pytest.param("deaeom4", deaeom("deaeom4"), "ch2++"),
    pytest.param("dipeom3", dipeom("dipeom3"), "ch2--", marks=short),
    pytest.param("dipeom4", dipeom("dipeom4"), "ch2--"),
    # linear molecule, degenerate pi orbitals: fine for variant A and for EOM roots
    pytest.param("linear_crcc23_A", crcc23("A"), "hf", marks=short),
    pytest.param("linear_eomccsd_pi", eomccsd(1), "hf", marks=short),
]


@pytest.mark.parametrize("name, energy, molecule", CASES, ids=[c.values[0] for c in CASES])
def test_fd_gradient_smooth_and_invariant(name, energy, molecule):
    _check_fd(energy, MOLECULES[molecule]())


@short
def test_fd_gradient_ccsd_matches_analytic():
    mol = MOLECULES["h2o"]()
    g_fd = _check_fd(ccsd, mol)

    def analytic(m):
        d = _driver(m, 1)
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        return d.run_gradient()

    assert np.allclose(g_fd, _quiet(analytic, mol), atol=1.0e-08)


@short
def test_fd_gradient_adaptive_ccpq_fixed_pspace():
    mol = MOLECULES["h2o"]()
    ref = _quiet(adaptive, mol)
    last = len(ref.pspaces) - 1

    def energy(m):
        return adaptive(m, pspaces=ref.pspaces).ccpq_energy[last]

    # the fixed-P-space run reproduces the adaptive run at the reference geometry
    fixed = _quiet(adaptive, mol, pspaces=ref.pspaces)
    assert np.allclose(fixed.ccp_energy, ref.ccp_energy, atol=1.0e-11)
    assert np.allclose(fixed.ccpq_energy, ref.ccpq_energy, atol=1.0e-11)
    _check_fd(energy, mol)


def test_fd_gradient_adaptive_eom_ccpq_fixed_pspace():
    mol = MOLECULES["h2o"]()
    ref = _quiet(adaptive_eom, mol)
    last = len(ref.pspaces) - 1

    def energy(m):
        return adaptive_eom(m, pspaces=ref.pspaces).eomccpq_energy[last]

    fixed = _quiet(adaptive_eom, mol, pspaces=ref.pspaces)
    assert np.allclose(fixed.eomccpq_energy, ref.eomccpq_energy, atol=1.0e-10)
    _check_fd(energy, mol)


@short
@pytest.mark.parametrize("variant", [
    "A",
    pytest.param("D", marks=pytest.mark.xfail(strict=True, reason=(
        "CR-CC(2,3)_D (and CC(t;3)_D, CR-EOMCC(2,3)_D) uses diagonal Hbar elements in its "
        "denominators, which are not invariant to rotations among degenerate orbitals; for a "
        "linear molecule the SCF returns an arbitrary orientation of each pi pair at every "
        "geometry, so E_D jumps by up to ~5e-6 Eh between displacements along the axis."))),
])
def test_crcc23_degenerate_orbital_rotation_invariance(variant):
    # Rotate the occupied pi pair of HF (relative to the pi* pair) at a fixed geometry. A
    # well-defined E(geometry), which FD needs, must not change.
    mol = MOLECULES["hf"]()

    def energy(angle):
        mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        mf.kernel()
        eps, C = mf.mo_energy, mf.C[0]
        p = next(p for p in range(1, len(eps)) if abs(eps[p + 1] - eps[p]) < 1.0e-8)  # first degenerate pair
        c, s = np.cos(angle), np.sin(angle)
        C[:, [p, p + 1]] = C[:, [p, p + 1]] @ np.array([[c, -s], [s, c]])
        d = Driver.from_scf(mf, nfrozen=1)
        d.options.update({"energy_convergence": 1.0e-12, "amp_convergence": 1.0e-10})
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        d.run_ccp3(method="crcc23")
        return _ecc(d) + d.deltap3[0][variant]

    assert abs(_quiet(energy, 0.0) - _quiet(energy, 0.4)) < 1.0e-9
