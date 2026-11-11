"""Adaptive CC(P;Q) with target="ccsdtq": the CC(P;3,4) correction with selection against CR-CC(2,4)
and CC(P;4), and CCSDTQ(P) on a P space of all triples and quadruples against CCSDTQ, for the
symmetrically stretched H2O molecule (R(OH) = 2Re) with the 6-31G basis and a frozen core."""
import copy

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import CCSDTQ, AdaptiveCCPQ, Molecule, RHF, Reference, get_active_triples_quadruples_pspace
from ccpy.methods import CCSDTQ_P, HBarCCSD, HBarCCSDLegacy, LeftCCSD
from ccpy.moments.ccp3_chol import calc_ccp3_2ba_with_selection
from ccpy.moments.ccp34 import calc_ccp34_2ba_with_selection
from ccpy.moments.ccp4 import calc_ccp4
from ccpy.moments.crcc24 import calc_crcc24

EMPTY3 = {s: np.ones((1, 6), dtype=np.int32) for s in ("aaa", "aab", "abb", "bbb")}
EMPTY4 = {s: np.ones((1, 8), dtype=np.int32) for s in ("aaaa", "aaab", "aabb", "abbb", "bbbb")}


@pytest.fixture(scope="module")
def water():
    mol = Molecule("O 0.0 0.0 -0.0180; H 0.0 3.030526 -2.117796; H 0.0 -3.030526 -2.117796", "6-31g",
                   unit="bohr", auxiliary_basis=AUX, verbose=False)
    return converged_scf(RHF, mol)


def full_excitations(system, spincases):
    """All excitations [a, b, ..., i, j, ...] (1-based, ascending within each spin block)."""
    from itertools import combinations, product

    nv = {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta}
    no = {"a": system.noccupied_alpha, "b": system.noccupied_beta}

    def block(n, spincase):
        return [sum(x, ()) for x in product(*(combinations(range(1, n[s] + 1), spincase.count(s)) for s in "ab"))]

    return {s: np.asarray([v + o for v in block(nv, s) for o in block(no, s)]) for s in spincases}


@pytest.mark.parametrize("rhf_symmetry", [True, False])
def test_ccp34_correction(water, rhf_symmetry):
    ref = Reference(water, nfrozen=1)
    ref.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
    t3, t4 = get_active_triples_quadruples_pspace(ref.system)
    cc = CCSDTQ_P(water, copy.deepcopy(t3), copy.deepcopy(t4), nfrozen=1, RHF_symmetry=rhf_symmetry, **OPTS)
    cc.energy()
    left = LeftCCSD(cc, RHF_symmetry=rhf_symmetry, **OPTS).ensure_solved()
    hbar, hbar4 = cc.get_hbar(HBarCCSD), cc.get_hbar(HBarCCSDLegacy)
    fock4 = cc.reference.legacy_hamiltonian(copy_=False)
    args = (cc.correlation_energy, hbar.H, cc.reference.H)

    def delta4(t3, t4, num_add=5, buffer_factor=2):
        e3 = calc_ccp3_2ba_with_selection(hbar.T, left.L, t3, *args, cc.system, num_add, use_RHF=rhf_symmetry)[0]
        e34, s3, s4 = calc_ccp34_2ba_with_selection(hbar.T, left.L, t3, t4, *args, hbar4.H, cc.system, num_add,
                                                    use_RHF=rhf_symmetry, buffer_factor=buffer_factor)
        return e34 - e3, s3, s4

    # without quadruples in the P space, the quadruples correction is CR-CC(2,4)
    d24 = calc_crcc24(hbar4.T, left.L, cc.correlation_energy, hbar4.H, fock4, cc.system, rhf_symmetry)[1]["D"]
    assert np.isclose(delta4(EMPTY3, EMPTY4)[0], d24, atol=1.0e-10)
    # closed shell: the P-space contributions of spin-flipped spincases agree (all variants of CC(P;4))
    if not rhf_symmetry:
        def ccp4(*spincases):
            p4 = {**EMPTY4, **{s: cc.t4_excitations[s] for s in spincases}}
            d = calc_ccp4(hbar4.T, left.L, cc.correlation_energy, hbar4.H, fock4, cc.system, p4, False)[1]
            return np.array([d[v] for v in "ABCD"])
        assert np.allclose(ccp4("aaaa"), ccp4("bbbb"), atol=1.0e-12)
        assert np.allclose(ccp4("aaab"), ccp4("abbb"), atol=1.0e-12)
    # the P-space contributions of the spincases add up
    full = delta4(EMPTY3, EMPTY4)[0]
    parts = sum(full - delta4(EMPTY3, {**EMPTY4, s: cc.t4_excitations[s]})[0] for s in EMPTY4)
    assert np.isclose(full - delta4(EMPTY3, cc.t4_excitations)[0], parts, atol=1.0e-12)

    # the selected excitations do not depend on the buffer size, and no quadruple is in the P space
    num_add = 40
    selections = [delta4(cc.t3_excitations, cc.t4_excitations, num_add, b)[1:] for b in (1, 2, 1000)]
    selections = [(sorted(map(tuple, s3)), sorted(map(tuple, s4))) for s3, s4 in selections]
    assert selections[0] == selections[1] == selections[2]
    triples, quadruples = selections[0]
    assert len(triples) + len(quadruples) == num_add and len(quadruples) > 0
    pspace = {(s, tuple(r)) for s in cc.t4_excitations for r in cc.t4_excitations[s]}
    spincase = {4: "aaaa", 3: "aaab", 2: "aabb", 1: "abbb", 0: "bbbb"}
    assert not any((spincase[sum(x % 2 for x in q) // 2], tuple((x + 1) // 2 for x in q)) in pspace
                   for q in quadruples)


@pytest.mark.parametrize("rhf_symmetry", [True, False])
def test_adaptive_ccpq_ccsdtq(water, rhf_symmetry):
    adapt = AdaptiveCCPQ(water, target="ccsdtq", nfrozen=1, percentage=[0.0, 1.0, 2.0], energy_tolerance=1.0e-10,
                         RHF_symmetry=rhf_symmetry, **OPTS)
    adapt.energy()
    sizes = [{s: 0 if np.all(p[s][0] == 1) else p[s].shape[0] for s in p} for p in adapt.pspaces]
    assert sizes[0] == {s: 0 for s in sizes[0]}
    assert sum(sizes[1].values()) < sum(sizes[2].values())
    assert any(sizes[2][s] for s in ("aaaa", "aaab", "aabb", "abbb", "bbbb"))
    # the CC(P;Q) energy is closer to CCSDTQ than the CC(P) energy
    e_ccsdtq = CCSDTQ(water, nfrozen=1, RHF_symmetry=rhf_symmetry, **OPTS).energy()
    assert abs(adapt.ccpq_energies[-1] - e_ccsdtq) < abs(adapt.ccp_energies[-1] - e_ccsdtq)

    # with all triples and quadruples in the P space, CC(P) is CCSDTQ and the correction vanishes
    system = Reference(water, nfrozen=1).system
    full = {**full_excitations(system, ("aaa", "aab", "abb", "bbb")),
            **full_excitations(system, ("aaaa", "aaab", "aabb", "abbb", "bbbb"))}
    exact = AdaptiveCCPQ(water, target="ccsdtq", nfrozen=1, pspaces=[full], RHF_symmetry=rhf_symmetry, **OPTS)
    assert np.isclose(exact.energy(), e_ccsdtq, atol=1.0e-8)
    assert np.isclose(exact.ccp_energies[0], e_ccsdtq, atol=1.0e-8)
