"""P-space methods of ccpy.methods (CCSDT(P), CC(t;3), CC(P;3), adaptive CC(P;Q), EOMCC(P),
IP-EOMCC(P)) against the legacy Driver."""
import copy

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf, fresh_driver, total

from ccpy import (CCP3, CCSDT_P, CCT3, EOMCCP3, EOMCCSDT_P, IPCCP3, IPEOM3_P, AdaptDriver, AdaptiveCCPQ, Molecule,
                  RHF, Reference, get_active_3h2p_pspace, get_active_triples_pspace)


@pytest.fixture(scope="module")
def f2():
    mol = Molecule("F 0.0 0.0 -1.8; F 0.0 0.0 1.8", "6-31g", unit="bohr", auxiliary_basis=AUX, verbose=False)
    return converged_scf(RHF, mol)


@pytest.fixture(scope="module")
def f2_t3(f2):
    ref = Reference(f2, nfrozen=1)
    ref.system.set_active_space(nact_occupied=3, nact_unoccupied=1)
    return get_active_triples_pspace(ref.system)


@pytest.fixture(scope="module")
def f2_ccsdt_p(f2, f2_t3):
    cc = CCSDT_P(f2, copy.deepcopy(f2_t3), nfrozen=1, **OPTS)
    cc.energy()
    return cc


def test_ccsdt_p_cct3_ccp3_2ba(f2, f2_t3, f2_ccsdt_p):
    driver = fresh_driver(f2, use_cholesky=True)
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=1)
    t3 = copy.deepcopy(f2_t3)
    driver.run_ccp("ccsdt_p_chol", t3_excitations=t3)
    driver.run_hbar("ccsd_chol")
    driver.run_leftcc("left_ccsd_chol")
    driver.run_ccp3("cct3_chol")
    cct3 = dict(driver.deltap3[0])
    driver.run_ccp3("ccp3_chol", state_index=0, t3_excitations=t3)
    cc = f2_ccsdt_p
    assert np.isclose(cc.energy(), total(driver), atol=1.0e-9)
    for variant in ("A", "D"):
        assert np.isclose(CCT3(cc, nacto=3, nactu=1, variant=variant).energy(), total(driver) + cct3[variant],
                          atol=1.0e-9)
    assert np.isclose(CCP3(cc).energy(), total(driver) + driver.deltap3[0]["D"], atol=1.0e-9)


def test_full_ccp3(f2, f2_t3, f2_ccsdt_p):
    # the CC(P) amplitudes come from the DF solver, HBar and left-CC(P) from the legacy code,
    # whose excitation orderings differ (see LeftCCSDT_P._align)
    driver = fresh_driver(f2)
    t3 = copy.deepcopy(f2_t3)
    driver.run_ccp("ccsdt_p", t3_excitations=t3)
    driver.run_hbar("ccsdt_p", t3_excitations=t3)
    driver.run_leftccp("left_ccsdt_p", t3_excitations=t3)
    driver.run_ccp3("ccp3", state_index=0, two_body_approx=False, t3_excitations=t3)
    full = CCP3(f2_ccsdt_p, two_body_approx=False)
    assert full.requires_df_port
    assert np.isclose(full.energy(), total(driver) + driver.deltap3[0]["D"], atol=1.0e-9)


def test_set_pspace_restart(f2, f2_t3):
    cc = CCSDT_P(f2, copy.deepcopy(f2_t3), nfrozen=1, **OPTS)
    e = cc.energy()
    cc.set_pspace(copy.deepcopy(f2_t3))
    assert not cc.converged
    assert np.isclose(cc.energy(), e, atol=1.0e-9)


@pytest.mark.parametrize("rhf_symmetry", [True, False])
def test_adaptive_ccpq(f2, rhf_symmetry):
    percentage = [0.0, 1.0, 2.0, 5.0]
    adapt = AdaptiveCCPQ(f2, nfrozen=1, percentage=percentage, energy_tolerance=1.0e-10, RHF_symmetry=rhf_symmetry,
                         **OPTS)
    e = adapt.energy()
    driver = fresh_driver(f2, use_cholesky=True)
    driver.options["RHF_symmetry"] = rhf_symmetry
    legacy = AdaptDriver(driver, percentage=percentage)
    legacy.options["energy_tolerance"] = 1.0e-10
    legacy.run()
    assert len(adapt.ccpq_energies) == len(percentage)
    assert np.allclose(adapt.ccp_energies, legacy.ccp_energy, atol=1.0e-9)
    assert np.allclose(adapt.ccpq_energies, legacy.ccpq_energy, atol=1.0e-7)
    sizes = [{s: p[s].shape[0] for s in p} for p in adapt.pspaces]
    assert sizes == [{s: p["t3_excitations"][s].shape[0] for s in p["t3_excitations"]} for p in legacy.pspaces]
    assert e == adapt.ccpq_energies[-1]
    if rhf_symmetry:
        # a copy at the same geometry reuses the recorded P spaces instead of selecting new ones
        again = adapt.at_geometry(f2.mol)
        assert np.isclose(again.energy(), e, atol=1.0e-8)
        assert np.allclose(again.ccp_energies, adapt.ccp_energies, atol=1.0e-8)


def test_eomccsdt_p_and_eomccp3(water_rhf):
    driver = fresh_driver(water_rhf)
    driver.system.set_active_space(2, 2)
    t3 = get_active_triples_pspace(driver.system)
    r3 = get_active_triples_pspace(driver.system)
    t3_new, r3_new = copy.deepcopy(t3), copy.deepcopy(r3)
    driver.run_ccp("ccsdt_p", t3_excitations=t3)
    driver.run_hbar("ccsdt_p", t3_excitations=t3)
    driver.run_guess("cisd", multiplicity=1, nact_occupied=2, nact_unoccupied=2, roots_per_irrep={"A": 2})
    for i in (1, 2):
        driver.run_eomccp("eomccsdt_p", state_index=i, t3_excitations=t3, r3_excitations=r3)
        driver.run_lefteomccp("left_ccsdt_p", state_index=i, t3_excitations=t3, r3_excitations=r3)
        driver.run_ccp3("ccp3", state_index=i, two_body_approx=False, t3_excitations=t3, r3_excitations=r3)
    eom = EOMCCSDT_P(water_rhf, r3_new, t3_excitations=t3_new, nfrozen=1, **OPTS)
    eom.initial_guess("cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 2})
    reference = np.array([total(driver, i) for i in (1, 2)])
    assert np.allclose(eom.energy(), reference, atol=1.0e-9)
    corrected = EOMCCP3(eom.left).energy()
    assert np.allclose(corrected, reference + [driver.deltap3[i]["D"] for i in (1, 2)], atol=1.0e-9)


def test_ipeom3_p_and_ipccp3(hydroxide_rhf):
    driver = fresh_driver(hydroxide_rhf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_guess("ipcisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=3, roots_per_irrep={"A": 2})
    driver.system.set_active_space(2, 2)
    r3 = get_active_3h2p_pspace(driver.system, num_active=1)
    r3_new = copy.deepcopy(r3)
    for i in (0, 1):
        driver.run_ipeomccp("ipeom3_p", state_index=i, r3_excitations=r3)
        driver.run_leftipeomccp("left_ipeom3_p", state_index=i, r3_excitations=r3)
        driver.run_ipccp3("ipccp3", state_index=i, r3_excitations=r3)
    ip = IPEOM3_P(hydroxide_rhf, r3_new, nfrozen=1, **OPTS)
    ip.initial_guess("ipcisd", multiplicity=-1, nacto=3, nactu=3, roots_per_irrep={"A": 2})
    reference = np.array([total(driver, i) for i in (0, 1)])
    assert np.allclose(ip.energy(), reference, atol=1.0e-9)
    corrected = IPCCP3(ip.left).energy()
    assert np.allclose(corrected, reference + [driver.deltap3[i]["D"] for i in (0, 1)], atol=1.0e-9)
