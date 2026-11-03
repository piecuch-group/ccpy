"""Amplitude-dependent Jacobi denominators (ccpy.cc.denominators): the diagonals match
finite-difference derivatives of the CC residuals, the CC(P) per-excitation values match the
full arrays, and the converged energies do not depend on the denominators."""
import copy
import itertools

import numpy as np
import pytest
from conftest import OPTS

import ccpy.cc.ccsd_chol as ccsd_chol
import ccpy.cc.ccsdt as ccsdt
from ccpy import CC3, CCSD, CCSDT, CCSDT_P, Reference, get_active_triples_pspace
from ccpy.cc.denominators import Denominators

KINDS = ["epstein-nesbet", "quasilinear", "newton"]
KERNEL_BLOCKS = {"update_t1a": "a", "update_t1b": "b", "update_t2a": "aa", "update_t2b": "ab", "update_t2c": "bb",
                 "update_t3a_v2": "aaa", "update_t3b_v2": "aab", "update_t3c_v2": "abb", "update_t3d_v2": "bbb"}


def _residual_loops(real, D):
    """cc_loops2 whose kernels return the residual R = step * D_MP and leave t unchanged."""
    class Loops:
        pass
    for name, spins in KERNEL_BLOCKS.items():
        def kernel(t, *args, f=getattr(real, name), spins=spins):
            return t, f(t.copy(), *args)[1] * D.mp(spins)
        setattr(Loops, name, staticmethod(kernel))
    return Loops


def _partners(spins, idx):
    """Index tuples of the amplitudes equal to +-t[idx], with their signs."""
    n = len(spins)
    groups = [[p for p in range(n) if spins[p] == s] for s in "ab"]

    def permutations():
        out = [({}, 1)]
        for g in groups:
            out = [({**m, **dict(zip(g, perm))}, sign * round(np.linalg.det(np.eye(len(g))[[g.index(x) for x in perm]])))
                   for m, sign in out for perm in itertools.permutations(g)]
        return out

    return {tuple(idx[v[p]] for p in range(n)) + tuple(idx[n + o[q]] for q in range(n)): sv * so
            for v, sv in permutations() for o, so in permutations()}


def _is_amplitude(spins, idx):
    n = len(spins)
    return not any(spins[p] == spins[q] and (idx[p] == idx[q] or idx[n + p] == idx[n + q])
                   for p in range(n) for q in range(p + 1, n))


@pytest.mark.parametrize("cls, module, blocks", [
    (CCSD, ccsd_chol, ["a", "b", "aa", "ab", "bb"]),
    (CCSDT, ccsdt, ["a", "b", "aa", "ab", "bb", "aaa", "aab", "abb", "bbb"]),
])
def test_diagonals_match_finite_differences(oh_uhf, cls, module, blocks, monkeypatch):
    cc = cls(oh_uhf)
    H = cc.hamiltonian
    rng = np.random.default_rng(3)
    T = cc._new_operator()
    for spins in ("aa", "ab", "bb"):
        x = 0.1 * rng.standard_normal(getattr(T, spins).shape)
        setattr(T, spins, sum(sign * np.transpose(x, p) for p, sign in _partners(spins, (0, 1, 2, 3)).items()))
    Z = cc._new_operator()  # T1 = T3 = 0 isolates the T2*T_K terms; T2 = 0 leaves the bare terms
    D, D0 = Denominators(T, H, "newton"), Denominators(Z, H, "newton")
    monkeypatch.setattr(module, "cc_loops2", _residual_loops(module.cc_loops2, D))

    def derivative(Tx, spins, idx, h=1.0e-4):
        values = []
        for step in (h, -h):
            Tp = copy.deepcopy(Tx)
            for p, sign in _partners(spins, idx).items():
                getattr(Tp, spins)[p] += sign * step
            dT = cc._new_operator()
            module.update(Tp, dT, H, cc._intermediates(), 0.0, False)
            values.append(getattr(dT, spins)[idx])
        return (values[0] - values[1]) / (2 * h)

    for spins in blocks:
        shape = getattr(T, spins).shape
        checked = 0
        while checked < 3:
            idx = tuple(int(rng.integers(0, n)) for n in shape)
            if not _is_amplitude(spins, idx):
                continue
            checked += 1
            linear = derivative(Z, spins, idx)  # -D_MP + Epstein-Nesbet terms
            en = D0.correction(spins)[idx]
            assert linear + D.mp(spins)[idx] == pytest.approx(en, abs=1.0e-9)
            assert derivative(T, spins, idx) - linear == pytest.approx(D.correction(spins)[idx] - en, abs=1.0e-8)


def test_pspace_values_match_full_arrays(oh_uhf):
    ref = Reference(oh_uhf)
    system = ref.system
    system.set_active_space(nact_occupied=system.noccupied_alpha, nact_unoccupied=system.nunoccupied_beta)
    excitations = get_active_triples_pspace(system)
    cc = CCSDT(oh_uhf)
    rng = np.random.default_rng(5)
    T = cc._new_operator()
    for spins in ("aa", "ab", "bb"):
        setattr(T, spins, rng.standard_normal(getattr(T, spins).shape))
    for kind in KINDS:
        D = Denominators(T, cc.hamiltonian, kind)
        for spins in ("aaa", "aab", "abb", "bbb"):
            rows = np.asarray(excitations[spins])
            full_idx = tuple(rows[:, k] - 1 for k in range(6))
            assert np.allclose(D.correction(spins, rows), D.correction(spins)[full_idx], atol=1.0e-12)
            assert np.allclose(D.mp(spins, rows), D.mp(spins)[full_idx], atol=1.0e-12)


@pytest.mark.parametrize("scf", ["water_rhf", "oh_uhf"])
@pytest.mark.parametrize("denominator", KINDS)
def test_ccsd_energy_independent_of_denominator(scf, denominator, request):
    mf = request.getfixturevalue(scf)
    e_mp = CCSD(mf, nfrozen=1, **OPTS).energy()
    assert CCSD(mf, nfrozen=1, denominator=denominator, **OPTS).energy() == pytest.approx(e_mp, abs=1.0e-8)


def test_ccsdt_energy_independent_of_denominator(oh_uhf):
    e_mp = CCSDT(oh_uhf, nfrozen=1, **OPTS).energy()
    assert CCSDT(oh_uhf, nfrozen=1, denominator="newton", **OPTS).energy() == pytest.approx(e_mp, abs=1.0e-8)


@pytest.mark.parametrize("scf", ["water_rhf", "oh_uhf"])
def test_ccsdt_p_energy_independent_of_denominator(scf, request):
    mf = request.getfixturevalue(scf)
    ref = Reference(mf, nfrozen=1)
    ref.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
    t3 = get_active_triples_pspace(ref.system)
    e_mp = CCSDT_P(mf, copy.deepcopy(t3), nfrozen=1, **OPTS).energy()
    e = CCSDT_P(mf, copy.deepcopy(t3), nfrozen=1, denominator="newton", **OPTS).energy()
    assert e == pytest.approx(e_mp, abs=1.0e-8)


def test_unsupported_denominator_is_rejected(water_rhf):
    with pytest.raises(ValueError, match="denominator"):
        CC3(water_rhf, denominator="newton")
    with pytest.raises(ValueError, match="denominator"):
        CCSD(water_rhf, denominator="shifted")
