"""Native (PySCF-free) density-fitted SCF built on the libint2 integrals ported from forte2."""
from pathlib import Path

import numpy as np
import pytest

from ccpy import Driver
from ccpy.integrals import Molecule, integrals
from ccpy.scf import RHF, ROHF, UHF

TEST_DATA_DIR = Path(__file__).parents[1].absolute() / "data"

H2O = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
CH2 = "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6"


@pytest.mark.parametrize("symmetry", [False, True])
def test_rhf_h2o(symmetry):
    # reference: PySCF scf.RHF(mol).density_fit(auxbasis="cc-pvtz-jkfit")
    mol = Molecule(H2O, "cc-pvtz", auxiliary_basis="cc-pvtz-jkfit", symmetry=symmetry)
    mf = RHF(mol, verbose=0)
    assert np.isclose(mf.kernel(), -76.0571212298, atol=1.0e-08)
    if symmetry:
        assert mol.point_group == "C2V"
        assert mf.determinant_symmetry == "a1"


@pytest.mark.parametrize("method, energy", [(ROHF, -38.8985653414), (UHF, -38.9025169104)])
def test_open_shell_ch2(method, energy):
    # reference: PySCF ROHF/UHF .density_fit(auxbasis="def2-universal-jkfit")
    mol = Molecule(CH2, "cc-pvdz", auxiliary_basis="def2-universal-jkfit", multiplicity=3, symmetry=True)
    mf = method(mol, verbose=0)
    assert np.isclose(mf.kernel(), energy, atol=1.0e-08)
    assert mf.determinant_symmetry == "b1"


def test_ccsd_from_native_rhf():
    # reference: PySCF cc.CCSD(frozen=1) on a cc-pvtz-ri density-fitted RHF
    mol = Molecule(H2O, "cc-pvdz", auxiliary_basis="cc-pvtz-rifit", symmetry=True)
    mf = RHF(mol, verbose=0)
    mf.kernel()
    for use_cholesky, method in ((False, "ccsd"), (True, "ccsd_chol")):
        driver = Driver.from_scf(mf, nfrozen=1, use_cholesky=use_cholesky)
        driver.run_cc(method=method)
        assert np.isclose(driver.system.reference_energy, mf.e_tot, atol=1.0e-10)
        assert np.isclose(driver.correlation_energy, -0.2112163207, atol=1.0e-07)


@pytest.mark.parametrize("method, ecorr", [(ROHF, -0.1217147627), (UHF, -0.1178095089)])
def test_ccsd_from_native_open_shell(method, ecorr):
    # reference: PySCF cc.UCCSD(frozen=1) on the same density-fitted reference
    mol = Molecule(CH2, "cc-pvdz", auxiliary_basis="cc-pvtz-rifit", multiplicity=3, symmetry=True)
    mf = method(mol, verbose=0)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    driver.run_cc(method="ccsd")
    assert np.isclose(driver.correlation_energy, ecorr, atol=1.0e-07)


def test_creom23_h2o_cholesky_gbs():
    """Same calculation as tests/h2o/test_creom23_chol_h2o.py, from a native SCF with a
    tight Cholesky factorization of the AO integrals and the basis read from a .gbs file."""
    water = """O  -0.2015391293       -0.1082616784        0.0000000000
               H   3.3968787793       -0.0015445063        0.0000000000
               H  -0.1983073653        1.7197354648        0.0000000000"""
    mol = Molecule(water, str(TEST_DATA_DIR / "h2o" / "paldusTZ.gbs"), unit="bohr",
                   symmetry=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    assert mol.point_group == "CS"

    driver = Driver.from_scf(mf, nfrozen=1, use_cholesky=True)
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    driver.run_guess(method="cisd_chol", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A'": 3}, multiplicity=1)
    driver.run_eomcc(method="eomccsd_chol", state_index=[1, 2, 3])
    driver.run_lefteomcc(method="left_ccsd_chol", state_index=[1, 2, 3])
    driver.run_ccp3(method="crcc23_chol", state_index=[0, 1, 2, 3])

    E0 = driver.system.reference_energy + driver.correlation_energy
    assert np.isclose(E0, -76.0299191191, atol=1.0e-07)
    assert np.isclose(E0 + driver.deltap3[0]["D"], -76.0385721524, atol=1.0e-07)
    assert np.isclose(E0 + driver.vertical_excitation_energy[1], -75.7051577894, atol=1.0e-07)
    assert np.isclose(E0 + driver.vertical_excitation_energy[2], -75.8177593641, atol=1.0e-07)
    assert np.isclose(E0 + driver.vertical_excitation_energy[3], -75.6371348770, atol=1.0e-07)
    assert np.isclose(E0 + driver.vertical_excitation_energy[1] + driver.deltap3[1]["D"], -75.7354406458, atol=1.0e-07)


@pytest.mark.parametrize("method, geometry, multiplicity", [
    (RHF, "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692", 1),
    (UHF, "C 0 0 0.1; H 0 0.98 -0.6; H 0.05 -0.98 -0.62", 3),
])
def test_df_gradient_vs_finite_difference(method, geometry, multiplicity):
    """Exercises all libint2 derivative integrals (overlap, kinetic, nuclear, 2c and 3c Coulomb)."""
    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="def2-universal-jkfit", multiplicity=multiplicity)
    mf = method(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    grad = mf.gradient()
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1.0e-08)  # translational invariance

    X = np.array([a[1] for a in mol.atoms])
    h = 1.0e-4
    for atom, x in ((0, 2), (2, 1)):
        energies = []
        for step in (-h, h):
            Y = X.copy()
            Y[atom, x] += step
            energies.append(method(mol.with_geometry(Y), verbose=0, e_tol=1.0e-12, d_tol=1.0e-10).kernel())
        assert np.isclose(grad[atom, x], (energies[1] - energies[0]) / (2 * h), atol=1.0e-06)


def test_integrals_against_pyscf():
    gto = pytest.importorskip("pyscf.gto")
    mol = Molecule(H2O, "cc-pvtz", auxiliary_basis="cc-pvtz-jkfit")
    pmol = gto.M(atom=H2O, basis="cc-pvtz")
    # libint2 orders real solid harmonics m = -l..l, so p functions are (y, z, x)
    perm, offset = [], 0
    for ib in range(pmol.nbas):
        l = pmol.bas_angular(ib)
        for _ in range(pmol.bas_nctr(ib)):
            idx = list(range(offset, offset + 2 * l + 1))
            perm += [idx[1], idx[2], idx[0]] if l == 1 else idx
            offset += 2 * l + 1
    p = np.ix_(perm, perm)
    assert np.allclose(integrals.overlap(mol), pmol.intor("int1e_ovlp")[p], atol=1.0e-09)
    assert np.allclose(integrals.hcore(mol), (pmol.intor("int1e_kin") + pmol.intor("int1e_nuc"))[p], atol=1.0e-09)
    dip = pmol.intor("int1e_r")
    assert np.allclose(np.array(integrals.emultipole1(mol)[1:]), np.array([d[p] for d in dip]), atol=1.0e-09)
    assert np.isclose(integrals.nuclear_repulsion(mol), pmol.energy_nuc(), atol=1.0e-09)
