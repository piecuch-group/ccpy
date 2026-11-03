"""DF/Cholesky methods of ccpy.methods against reference energies and the legacy Driver."""
from pathlib import Path

import numpy as np
import pytest
from conftest import OPTS, converged_scf, fresh_driver, total

from ccpy import CCSD, CREOMCC23, CRCC23, EOMCCSD, MP2, Molecule, RHF

DATA = str(Path(__file__).parents[1].absolute() / "data")


@pytest.fixture(scope="module")
def stretched_water_cc():
    """CCSD of H2O at R(OH) = 3.6 bohr (tests/h2o/test_creom23_chol_h2o.py)."""
    geometry = """O  -0.2015391293  -0.1082616784  0.0000000000
                  H   3.3968787793  -0.0015445063  0.0000000000
                  H  -0.1983073653   1.7197354648  0.0000000000"""
    mol = Molecule(geometry, DATA + "/h2o/paldusTZ.gbs", unit="bohr", symmetry=True, cholesky_tei=True,
                   cholesky_tol=1.0e-12, verbose=False)
    return CCSD(converged_scf(RHF, mol), nfrozen=1, **OPTS)


def test_ccsd_and_crcc23(stretched_water_cc):
    cc = stretched_water_cc
    assert np.isclose(cc.energy(), -76.0299191191, atol=1.0e-08)
    cr = CRCC23(cc)
    assert np.isclose(cr.energy(), -76.0385721524, atol=1.0e-08)
    assert set(cr.deltas) == {"A", "B", "C", "D"}
    # the correction shares the left-CCSD of the CC object
    assert cr.left is cc.left


def test_eomccsd_and_creomcc23(stretched_water_cc):
    eom = EOMCCSD(stretched_water_cc)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=3, nactu=2, roots_per_irrep={"A'": 3})
    assert np.allclose(eom.energy(), [-75.7051577894, -75.8177593641, -75.6371348770], atol=1.0e-08)
    assert np.isclose(eom.energy(root=1), -75.8177593641, atol=1.0e-08)
    creom = CREOMCC23(eom)
    full = [-75.7354406458, -75.8365241661, -75.6555306790]                         # CR-EOMCC(2,3)
    assert np.allclose(creom.full_energy(), full, atol=1.0e-08)
    assert np.isclose(creom.full_energy(root=1), full[1], atol=1.0e-08)
    # energy() is delta-CR-EOMCC(2,3): the full correction without r0 times the ground-state moment
    ddeltas = np.array([d["D"] for d in creom.ddeltas])
    assert np.allclose(creom.energy(), eom.energy() + ddeltas, atol=1.0e-12)
    assert np.allclose(creom.raw_energy(), eom.raw_energy() + ddeltas, atol=1.0e-12)
    assert not np.allclose(creom.energy(), full, atol=1.0e-06)


@pytest.mark.parametrize("solver", ["standard", "multiroot"])
def test_a2_singlets(stretched_water_cc, solver):
    singlet = EOMCCSD(stretched_water_cc, davidson_solver=solver)
    singlet.initial_guess(method="cisd", multiplicity=1, nacto=3, nactu=2, roots_per_irrep={'A"': 2})
    assert np.allclose(singlet.energy(), [-75.9741899000, -75.7311853758], atol=1.0e-08)
    assert np.allclose(CREOMCC23(singlet).full_energy(), [-75.9892626216, -75.7317203543], atol=1.0e-08)


@pytest.mark.parametrize("scf", ["oh_rohf", "oh_uhf"])
def test_open_shell_ccsd_and_crcc23(scf, request):
    mf = request.getfixturevalue(scf)
    cr = CRCC23(mf, nfrozen=1, **OPTS)
    driver = fresh_driver(mf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_leftcc("left_ccsd")
    driver.run_ccp3("crcc23")
    assert np.isclose(cr.cc.energy(), total(driver), atol=1.0e-09)
    assert np.isclose(cr.energy(), total(driver) + driver.deltap3[0]["D"], atol=1.0e-09)
    assert not cr.cc.options["RHF_symmetry"]


def test_uhf_eomccsd(oh_uhf):
    eom = EOMCCSD(oh_uhf, nfrozen=1, **OPTS)
    eom.initial_guess(method="cis", multiplicity=2, roots_per_irrep={"A": 3})
    driver = fresh_driver(oh_uhf)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_guess("cis", multiplicity=2, roots_per_irrep={"A": 3})
    driver.run_eomcc("eomccsd", state_index=[1, 2, 3])
    assert np.allclose(eom.energy(), [total(driver, i) for i in (1, 2, 3)], atol=1.0e-08)


def test_mp2(water_rhf):
    driver = fresh_driver(water_rhf)
    driver.run_mbpt("mp2")
    assert np.isclose(MP2(water_rhf, nfrozen=1).energy(), total(driver), atol=1.0e-10)
