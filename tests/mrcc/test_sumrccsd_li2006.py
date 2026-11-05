"""GMS-SU-CCSD for N2 and C2 against Li and Paldus, J. Chem. Phys. 124, 034112 (2006):
cc-pVTZ (spherical), RHF orbitals, 1s cores frozen, D2h symmetry.

N2 (R = 2.068 bohr): Table III nR-SU-CCSD vertical excitation energies (eV, two decimals)
relative to the SR-CCSD ground state of footnote a, -109.355346 Eh.
C2 (R = 1.2425 Angstrom): Table I FCI energies and nR-SU-CCSD deviations from them (mEh, two
decimals), i.e. total energies to +-5e-6 Eh.

The B1u/Au spaces of N2 are incomplete (internal doubles from the C conditions)."""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf, cc, ao2mo
from pyscf.data.nist import HARTREE2EV

from ccpy.models.system import System
from ccpy.mrcc.sumrccsd import kernel


def setup(atom, unit):
    mol = gto.M(atom=atom, unit=unit, basis="cc-pvtz", symmetry="D2h", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    n = C.shape[1]
    h = C.T @ mf.get_hcore() @ C
    e2 = ao2mo.restore(1, ao2mo.kernel(mol, C), n).transpose(0, 2, 1, 3)
    return mol, mf, h, e2, System(mol.nelectron, n, 1, 2, nuclear_repulsion=mol.energy_nuc())


def det(occ, ra=(), aa=(), rb=(), ab=()):
    """Determinant from occ by replacing alpha orbitals ra -> aa and beta orbitals rb -> ab."""
    a = [x for x in occ if x not in ra] + list(aa)
    b = [x for x in occ if x not in rb] + list(ab)
    return np.array(sorted(a) + sorted(b))


def pair(occ, hole, particle):
    """The alpha and the beta determinant of a single excitation."""
    return [det(occ, [hole], [particle]), det(occ, rb=[hole], ab=[particle])]


@pytest.fixture(scope="module")
def n2():
    mol, mf, h, e2, system = setup("N 0 0 0; N 0 0 2.068", "Bohr")
    irreps = [mol.irrep_name[list(mol.irrep_id).index(s)] for s in mf.get_orbsym()[:9]]
    assert irreps[2:] == ["Ag", "B1u", "Ag", "B2u", "B3u", "B2g", "B3g"]
    return mf, h, e2, system


@pytest.fixture(scope="module")
def c2():
    mol, mf, h, e2, system = setup("C 0 0 0; C 0 0 1.2425", "Angstrom")
    irreps = [mol.irrep_name[list(mol.irrep_id).index(s)] for s in mf.get_orbsym()[:9]]
    assert irreps[2:] == ["Ag", "B1u", "B2u", "B3u", "Ag", "B2g", "B3g"]
    return h, e2, system


def test_n2_ground_state_ccsd(n2):
    mf, h, e2, system = n2
    mycc = cc.CCSD(mf, frozen=2)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-12, 1e-10
    assert abs(mycc.run().e_tot - -109.355346) < 1e-6
    e = kernel(h, e2, system, [det([1, 2, 3, 4, 5])], convergence=1e-10)[2]
    assert abs(e[0].real - -109.355346) < 1e-6


# N2 correlated orbitals: 2 = 2sigma_u, 3 = 3sigma_g, 4/5 = 1pi_u (y/x), 6/7 = 1pi_g (xz/yz)
N2_OCC, N2_S2U, N2_S3G, N2_PIY, N2_PIX, N2_PGX, N2_PGY = [1, 2, 3, 4, 5], 2, 3, 4, 5, 6, 7
N2_CASES = {
    "2R B2g": (pair(N2_OCC, N2_S3G, N2_PGX), [8.24, 9.56]),
    "4R Au": (pair(N2_OCC, N2_PIX, N2_PGY) + pair(N2_OCC, N2_PIY, N2_PGX), [8.76, 9.73, 9.81, 10.34]),
    "2R B2u": (pair(N2_OCC, N2_S2U, N2_PGY), [11.63, 14.05]),
}


@pytest.mark.parametrize("name", list(N2_CASES))
def test_n2_table3(n2, name):
    _, h, e2, system = n2
    determinants, vees = N2_CASES[name]
    roots = np.sort(kernel(h, e2, system, determinants, convergence=1e-10)[2].real)
    assert np.allclose((roots - -109.355346) * HARTREE2EV, vees, atol=0.006), roots


# C2 correlated orbitals: 2 = 2sigma_u, 3/4 = 1pi_u (y/x), 5 = 3sigma_g
C2_OCC, C2_S2U, C2_PIY, C2_PIX, C2_S3G = [1, 2, 3, 4], 2, 3, 4, 5
C2_CASES = {  # model space, FCI energy, deviation (mEh) of the state
    "2R B1g 1Dg": ([det(C2_OCC, [C2_PIX], [C2_S3G], [C2_PIY], [C2_S3G]),
                    det(C2_OCC, [C2_PIY], [C2_S3G], [C2_PIX], [C2_S3G])], -75.704829, 11.20),
    "2R B2u 1Pu": (pair(C2_OCC, C2_PIY, C2_S3G), -75.738152, 18.96),
    "2R B1u 1Su+": (pair(C2_OCC, C2_S2U, C2_S3G), -75.584162, 45.41),
}


@pytest.mark.parametrize("name", list(C2_CASES))
def test_c2_table1(c2, name):
    h, e2, system = c2
    determinants, fci, deviation = C2_CASES[name]
    roots = kernel(h, e2, system, determinants, convergence=1e-10)[2].real
    assert np.abs(roots - (fci + 1e-3 * deviation)).min() < 1e-5, roots
