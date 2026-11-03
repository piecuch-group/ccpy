"""Methods that still use four-index equations (requires_df_port) against the legacy Driver
on the same SCF solution; the integrals are materialized from the same DF vectors."""
import numpy as np
import pytest
from conftest import AUX, OPTS, WATER, converged_scf, fresh_driver, total

from ccpy import (ACCSD, CC2, CC4, CCD, CCSDT, CCSDTQ, CCSDpT, CRCC24, CRCC34, CREACC23, CRIPCC23, DEAEOM3,
                  DIPEOM3, DIPEOMCCSDTA_star, EAEOM2, EOMCC3, IPEOM2, IPEOMCCSDTA_star, LRCCSD, LeftCCSDT,
                  LeftEOMCCSDT, EOMCCSDT, Molecule, RHF, ROHF, SFEOMCCSD)
from ccpy.methods.hbar import HBarCCSDTA


@pytest.mark.parametrize("name,cls,args", [("ccd", CCD, ()), ("cc2", CC2, ()), ("ccsdtq", CCSDTQ, ()),
                                           ("cc4", CC4, ()), ("accsd", ACCSD, ([1.0, 1.0, 0.0, 0.0, 1.0],))])
def test_ground_state(water_rhf, name, cls, args):
    driver = fresh_driver(water_rhf)
    driver.run_cc(name, acparray=args[0]) if args else driver.run_cc(name)
    method = cls(water_rhf, *args, nfrozen=1, **OPTS)
    assert method.requires_df_port
    assert np.isclose(method.energy(), total(driver), atol=1.0e-9)


def test_ground_state_corrections(water_rhf):
    driver = fresh_driver(water_rhf)
    driver.run_cc("ccsd")
    driver.run_ccp3("ccsd(t)")
    assert np.isclose(CCSDpT(water_rhf, nfrozen=1, **OPTS).energy(), total(driver) + driver.deltap3[0]["A"],
                      atol=1.0e-9)
    driver = fresh_driver(water_rhf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_leftcc("left_ccsd")
    driver.run_ccp4("crcc24")
    assert np.isclose(CRCC24(water_rhf, nfrozen=1, **OPTS).energy(), total(driver) + driver.deltap4[0]["D"],
                      atol=1.0e-9)
    driver = fresh_driver(water_rhf)
    driver.run_cc("ccsdt")
    driver.run_hbar("ccsdt")
    driver.run_leftcc("left_ccsdt")
    driver.run_ccp4("crcc34")
    assert np.isclose(CRCC34(water_rhf, nfrozen=1, **OPTS).energy(), total(driver) + driver.deltap4[0]["A"],
                      atol=1.0e-9)


def test_eomccsdt_with_left_states():
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 0.0 2.1160902167", "6-31g", unit="bohr", multiplicity=2,
                   auxiliary_basis=AUX, verbose=False)
    mf = converged_scf(ROHF, mol)
    driver = fresh_driver(mf)
    driver.run_cc("ccsdt")
    driver.run_hbar("ccsdt")
    driver.run_guess("cis", multiplicity=2, roots_per_irrep={"A": 3})
    driver.run_eomcc("eomccsdt", state_index=[1, 2, 3])
    driver.run_leftcc("left_ccsdt", state_index=[0, 1, 2, 3])
    eom = EOMCCSDT(mf, nfrozen=1, maximum_iterations=100, **OPTS)
    eom.initial_guess("cis", multiplicity=2, roots_per_irrep={"A": 3})
    assert isinstance(eom.cc, CCSDT)
    assert np.allclose(eom.energy(), [total(driver, i) for i in (1, 2, 3)], atol=1.0e-9)
    left = LeftCCSDT(eom.cc).solve()
    assert np.allclose(left.L.flatten(), driver.L[0].flatten(), atol=1.0e-7)
    left_eom = LeftEOMCCSDT(eom, solver="jacobi").solve()
    for i in range(3):
        assert np.allclose(left_eom.L[i].flatten(), driver.L[i + 1].flatten(), atol=1.0e-7)


def test_eomcc3(water_rhf):
    driver = fresh_driver(water_rhf)
    driver.run_cc("cc3")
    driver.run_hbar("cc3")
    driver.run_guess("cisd", multiplicity=1, nact_occupied=2, nact_unoccupied=2, roots_per_irrep={"A": 3})
    driver.run_eomcc("eomcc3", state_index=[1, 2, 3])
    eom = EOMCC3(water_rhf, nfrozen=1, **OPTS)
    eom.initial_guess("cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 3})
    assert np.allclose(eom.energy(), [total(driver, i) for i in (1, 2, 3)], atol=1.0e-9)


def _ip_ea_reference(mf, hbar, guess, eom, left, correction, roots):
    driver = fresh_driver(mf)
    driver.run_cc("ccsd")
    driver.run_hbar(hbar)
    driver.run_guess(guess, multiplicity=-1, nact_occupied=3, nact_unoccupied=3, roots_per_irrep={"A": roots})
    index = list(range(roots))
    getattr(driver, f"run_{eom[0]}")(eom[1], state_index=index)
    getattr(driver, f"run_{left[0]}")(left[1], state_index=index)
    getattr(driver, f"run_{correction[0]}")(correction[1], state_index=index)
    return driver


@pytest.mark.parametrize("cls,correction,hbar,variant", [
    (IPEOM2, CRIPCC23, None, "D"), (IPEOM2, IPEOMCCSDTA_star, HBarCCSDTA, "A"), (EAEOM2, CREACC23, None, "D")])
def test_ip_ea_eomcc_and_corrections(hydroxide_rhf, cls, correction, hbar, variant):
    kind = "ip" if cls is IPEOM2 else "ea"
    driver = _ip_ea_reference(hydroxide_rhf, "ccsdta" if hbar else "ccsd", f"{kind}cisd",
                              (f"{kind}eomcc", f"{kind}eom2"), (f"left{kind}eomcc", f"left_{kind}eom2"),
                              (f"{kind}ccp3", correction.__name__.lower()), roots=2)
    eom = cls(hydroxide_rhf, hbar_class=hbar, nfrozen=1, **OPTS)
    eom.initial_guess(f"{kind}cisd", multiplicity=-1, nacto=3, nactu=3, roots_per_irrep={"A": 2})
    reference = np.array([total(driver, i) for i in (0, 1)])
    assert np.allclose(eom.energy(), reference, atol=1.0e-9)
    corrected = correction(eom).energy()
    assert np.allclose(corrected, reference + [driver.deltap3[i][variant] for i in (0, 1)], atol=1.0e-9)


def test_dip_and_dea_eomcc(water_rhf):
    driver = fresh_driver(water_rhf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsdta")
    nocc = driver.system.noccupied_alpha
    driver.run_guess("dipcis", multiplicity=-1, nact_occupied=nocc, roots_per_irrep={"A": 2}, use_symmetry=False)
    driver.run_dipeomcc("dipeom3", state_index=[0, 1])
    driver.run_dipccp4("dipeomccsdta_star", state_index=[0, 1])
    dip = DIPEOM3(water_rhf, hbar_class=HBarCCSDTA, nfrozen=1, **OPTS)
    dip.initial_guess("dipcis", multiplicity=-1, nacto=nocc, roots_per_irrep={"A": 2}, use_symmetry=False)
    reference = np.array([total(driver, i) for i in (0, 1)])
    assert np.allclose(dip.energy(), reference, atol=1.0e-9)
    star = DIPEOMCCSDTA_star(dip).energy()
    assert np.allclose(star, reference + [driver.deltap4[i]["A"] for i in (0, 1)], atol=1.0e-9)

    dication = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, charge=2, verbose=False))
    driver = fresh_driver(dication)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_guess("deacis", multiplicity=-1, nact_unoccupied=4, roots_per_irrep={"A": 2}, use_symmetry=False)
    driver.run_deaeomcc("deaeom3", state_index=[0, 1])
    dea = DEAEOM3(dication, nfrozen=1, **OPTS)
    dea.initial_guess("deacis", multiplicity=-1, nactu=4, roots_per_irrep={"A": 2}, use_symmetry=False)
    assert np.allclose(dea.energy(), [total(driver, i) for i in (0, 1)], atol=1.0e-9)


def test_sfeomccsd():
    mol = Molecule("C 0 0 0; H 0 1.0 0.6; H 0 -1.0 0.6", "6-31g", auxiliary_basis=AUX, multiplicity=3, verbose=False)
    mf = converged_scf(ROHF, mol)
    driver = fresh_driver(mf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_guess("sfcis", multiplicity=1, roots_per_irrep={"A": 2}, nact_occupied=2, nact_unoccupied=2)
    driver.run_sfeomcc("sfeomccsd", state_index=[0, 1])
    sf = SFEOMCCSD(mf, nfrozen=1, **OPTS)
    sf.initial_guess("sfcis", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 2})
    assert np.allclose(sf.energy(), [total(driver, i) for i in (0, 1)], atol=1.0e-9)


def test_lrccsd_dipole(water_rhf):
    from ccpy.interfaces.scf_tools import get_multipole_integral

    driver = fresh_driver(water_rhf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    mu, _ = get_multipole_integral(1, water_rhf, driver.system)
    driver.run_lrcc(method="lrccsd", prop=mu[2])
    assert np.isclose(LRCCSD(water_rhf, mu[2], nfrozen=1, **OPTS).property(), driver.correlation_property,
                      atol=1.0e-9)

