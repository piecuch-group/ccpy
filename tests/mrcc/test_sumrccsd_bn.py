"""GMS-SU-CCSD for BN against Table I of Li and Paldus, J. Chem. Phys. 134, 214118 (2011):
R = 1.281 Angstrom, Dunning-Hay (9s5p)/[3s2p] basis (GAMESS "DH", Basis Set Exchange
"SV (Dunning-Hay)"), RHF orbitals, B and N 1s frozen. The paper lists FCI energies and the
nR-SU-CCSD deviations from them (mEh); both are rounded, so agreement is checked to 0.02 mEh.
The 2R spaces are incomplete, with internal doubles fixed by the C conditions."""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf, ao2mo

from ccpy.models.system import System
from ccpy.mrcc.sumrccsd import kernel

SV_DUNNING_HAY = """
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (10s,5p) -> [3s,2p]
B    S
   2788.0                    0.001288
    419.00                   0.009835
     96.470                  0.047648
     28.070                  0.160069
      9.3760                 0.362894
      3.4060                 0.433582
      1.3060                 0.140082
B    S
      3.4060                -0.179330
      0.3245                 1.062594
B    S
      0.1022                 1.000000
B    P
     11.340                  0.017988
      2.4360                 0.110343
      0.6836                 0.383072
      0.2134                 0.647895
B    P
      0.07011                1.000000
#BASIS SET: (10s,5p) -> [3s,2p]
N    S
   5909.0                    0.001190
    887.50                   0.009099
    204.70                   0.044145
     59.840                  0.150464
     20.000                  0.356741
      7.1930                 0.446533
      2.6860                 0.145603
N    S
      7.1930                -0.160405
      0.7000                 1.058215
N    S
      0.2133                 1.000000
N    P
     26.790                  0.018254
      5.9560                 0.116461
      1.7070                 0.390178
      0.5314                 0.637102
N    P
      0.1654                 1.000000
END
"""


@pytest.fixture(scope="module")
def bn():
    basis = {atom: gto.parse(SV_DUNNING_HAY, atom) for atom in ("B", "N")}
    mol = gto.M(atom="B 0 0 0; N 0 0 1.281", basis=basis, symmetry="C2v", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    n = C.shape[1]
    h = C.T @ mf.get_hcore() @ C
    e2 = ao2mo.restore(1, ao2mo.kernel(mol, C), n).transpose(0, 2, 1, 3)
    # MOs 5, 6 are the pi pair (b1, b2) and 7 is a1, as labelled in the paper
    assert [int(s) for s in mf.get_orbsym()[:7]] == [0, 0, 0, 0, 2, 3, 0]
    irreps = [{0: "A1", 1: "A2", 2: "B1", 3: "B2"}[int(s)] for s in mf.get_orbsym()]
    return h, e2, System(12, n, 1, 2, nuclear_repulsion=mol.energy_nuc()), irreps


def mo(*numbers):
    """Paper MO numbers -> correlated orbitals (1-based; two cores frozen)."""
    return [k - 2 for k in numbers]


OCC = mo(3, 4, 5, 6)


def det(remove_alpha=(), add_alpha=(), remove_beta=(), add_beta=()):
    alpha = [x for x in OCC if x not in mo(*remove_alpha)] + mo(*add_alpha)
    beta = [x for x in OCC if x not in mo(*remove_beta)] + mo(*add_beta)
    return np.array(sorted(alpha) + sorted(beta))


# model space, and {state: (FCI energy, nR-SU-CCSD deviation in mEh)} from Table I
CASES = {
    "SR": ([det()], {"1A1": (-79.11465, 21.98)}),
    "2R B1": ([det((5,), (7,)), det(remove_beta=(5,), add_beta=(7,))],
              {"3B1": (-79.10132, 9.50), "1B1": (-79.07678, 7.82)}),
    "2R A2": ([det((5,), (7,), (6,), (7,)), det((6,), (7,), (5,), (7,))],
              {"3A2": (-79.01776, 3.62), "1A2": (-78.96175, 4.62)}),
    "5R A1": ([det(), det((4,), (7,)), det(remove_beta=(4,), add_beta=(7,)), det((5,), (7,), (5,), (7,)),
               det((6,), (7,), (6,), (7,))],
              {"1A1": (-79.11465, 17.10), "3A1": (-79.07184, 11.15), "1A1 4->7": (-78.99166, 4.72),
               "1A1 5^2,6^2->7^2": (-78.96175, 5.59)}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_li_paldus_2011_table1(bn, name):
    h, e2, system, _ = bn
    determinants, states = CASES[name]
    roots = np.sort(kernel(h, e2, system, determinants, convergence=1e-10)[2].real)
    for state, (fci, deviation) in states.items():
        target = fci + 1e-3 * deviation
        assert np.abs(roots - target).min() < 2e-5, (name, state, roots, target)


def test_symmetry_blocks(bn):
    """With orbital symmetries, pairs of references of different symmetry are skipped (Heff = 0,
    no coupling): a B1 + A2 model space gives the roots of the separate 2R spaces."""
    from ccpy.mrcc.sumrcc import get_model_space
    h, e2, system, irreps = bn
    labelled = System(12, h.shape[0], 1, 2, point_group="C2V", orbital_symmetries=irreps,
                      nuclear_repulsion=system.nuclear_repulsion)
    dets = CASES["2R B1"][0] + CASES["2R A2"][0]
    assert get_model_space(labelled, dets).symmetry == [2, 2, 1, 1]
    separate = np.sort(np.concatenate([kernel(h, e2, system, CASES[k][0], convergence=1e-10)[2].real
                                       for k in ("2R B1", "2R A2")]))
    for sys_ in (system, labelled):
        roots = np.sort(kernel(h, e2, sys_, dets, convergence=1e-10)[2].real)
        assert np.allclose(roots, separate, atol=1e-8)
