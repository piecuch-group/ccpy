"""Rank-local Gauss-Seidel updates versus legacy per-spin equations."""
import copy
import itertools
from types import SimpleNamespace as NS

import numpy as np
import pytest
from ccpy.cc import ccsdt_p_chol
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator


def make_case(rhf=False, empty=()):
    rng = np.random.default_rng(741)
    oa, ob, n = 3, 3 if rhf else 4, 8
    va, vb = n - oa, n - ob
    system = NS(norbitals=n, noccupied_alpha=oa, noccupied_beta=ob,
                nunoccupied_alpha=va, nunoccupied_beta=vb)
    mats = {s: rng.normal(scale=.01, size=(n,) * (2 * len(s))) for s in ('a', 'b', 'aa', 'ab', 'bb')}
    chol = NS()
    for spin, no in [('a', oa), ('b', ob)]:
        mats[spin] = np.diag(np.r_[np.arange(-no, 0), np.arange(1, n-no+1)])
        full = rng.normal(scale=.03, size=(11, n, n))  # nonsymmetric factors
        slices = {'o': slice(0, no), 'v': slice(no, n)}
        setattr(chol, spin, NS(**{p: full[:, slices[p[0]], slices[p[1]]] for p in ('oo','ov','vo','vv')}))
    exc = {}
    for spin in ('aaa', 'aab', 'abb', 'bbb'):
        dims_v = [va if x == 'a' else vb for x in spin]
        dims_o = [oa if x == 'a' else ob for x in spin]
        def tuples(dims):
            return [x for x in itertools.product(*(range(1, d+1) for d in dims))
                    if all(x[i] < x[j] for i in range(3) for j in range(i+1, 3) if spin[i] == spin[j])]
        rows = np.array([v+o for v in tuples(dims_v) for o in tuples(dims_o)], dtype=np.int32)
        rows = rows[rng.choice(len(rows), max(1, len(rows)//3), replace=False)]
        exc[spin] = np.asfortranarray(rows)
    if rhf:
        mats['b'] = mats['a'].copy(); mats['bb'] = mats['aa'].copy(); chol.b = copy.deepcopy(chol.a)
        exc['bbb'] = exc['aaa'].copy(); exc['abb'] = np.asfortranarray(exc['aab'][:, [2,0,1,5,3,4]])
    for spin in empty:
        exc[spin] = np.ones((1, 6), dtype=np.int32, order='F')
    sizes = [0 if s in empty else len(exc[s]) for s in ('aaa','aab','abb','bbb')]
    T = ClusterOperator(system, 3, p_orders=[3], pspace_sizes=[sizes])
    for spin in T.spin_cases:
        a = rng.normal(scale=.001, size=getattr(T, spin).shape)
        if spin in ('aa','bb'):
            a = a - a.swapaxes(0,1); a = a - a.swapaxes(2,3)
        if spin in empty: a.fill(0.)
        setattr(T, spin, np.asfortranarray(a))
    if rhf:
        T.b=T.a.copy(); T.bb=T.aa.copy(); T.bbb=T.aaa.copy(); T.abb=T.aab.copy()
    H = Integral(system, 2, mats, chol=chol)
    return system, T, H, exc


def fresh_x(system):
    X = Integral.from_empty(system, 2, use_none=True)
    X.chol = NS(a=NS(), b=NS())
    return X


def ordered(values, exc, spin):
    if len(spin) < 3: return getattr(values, spin)
    order = np.lexsort(exc[spin].T[::-1])
    return getattr(values, spin)[order]


def legacy_rank_update(monkeypatch, T, dT, H, X, shift, rhf, exc, denominator,
                       freeze_rank=True):
    """Evaluate legacy full equations while freezing amplitudes within each rank."""
    pending_t2, pending_t3 = {}, {}

    def defer_t2(original, spin):
        def update(T, dT, X, H, shift, t3_excitations, residual_only=False):
            assert residual_only
            old = getattr(T, spin)
            # Native updates may modify their input; isolate only this spin.
            setattr(T, spin, old.copy(order='F'))
            T, dT = original(T, dT, X, H, shift, t3_excitations,
                             residual_only=False)
            pending_t2[spin] = (getattr(T, spin).copy(), getattr(dT, spin).copy())
            if freeze_rank:
                setattr(T, spin, old)
            return T, dT
        return update

    def commit_t2(T, dT, X, H, shift, t3_excitations, flag_RHF):
        assert set(pending_t2) == ({'aa', 'ab'} if flag_RHF else {'aa', 'ab', 'bb'})
        for spin, (amplitude, step) in pending_t2.items():
            setattr(T, spin, amplitude)
            setattr(dT, spin, step)
        return T, dT

    def defer_t3(original, spin):
        def update(T, dT, X, H, shift, t3_excitations, residual_only=False):
            assert residual_only
            T, dT, t3_excitations = original(
                T, dT, X, H, shift, t3_excitations, residual_only=False)
            step = getattr(dT, spin).copy()
            pending_t3[spin] = step
            # Sorting changes both the P list and its amplitudes. Subtract in
            # output order rather than restoring the unsorted input array.
            if freeze_rank:
                setattr(T, spin, np.asfortranarray(getattr(T, spin) - step))
            return T, dT, t3_excitations
        return update

    def restore_t3_residuals(T, dT, X, H, t3_excitations, flag_RHF):
        for spin, step in pending_t3.items():
            rows = t3_excitations[spin] - 1
            mp = sum(np.diag(getattr(H, s).oo)[rows[:, p + 3]]
                     - np.diag(getattr(H, s).vv)[rows[:, p]]
                     for p, s in enumerate(spin))
            # Production still divides by MP, then applies non-MP corrections.
            setattr(dT, spin, step * (mp - shift) if freeze_rank else np.zeros_like(step))

    with monkeypatch.context() as patch:
        for suffix, spin in zip('abc', ('aa', 'ab', 'bb')):
            name = f'update_t2{suffix}'
            patch.setattr(ccsdt_p_chol, name, defer_t2(getattr(ccsdt_p_chol, name), spin))
        for suffix, spin in zip('abcd', ('aaa', 'aab', 'abb', 'bbb')):
            name = f'update_t3{suffix}'
            patch.setattr(ccsdt_p_chol, name, defer_t3(getattr(ccsdt_p_chol, name), spin))
        patch.setattr(ccsdt_p_chol, 'add_vvvv_t2', lambda T, dT, X, rhf: None)
        patch.setattr(ccsdt_p_chol, 'update_t2_amplitudes', commit_t2)
        patch.setattr(ccsdt_p_chol, 'add_vvvv_t3', restore_t3_residuals)
        T, dT = ccsdt_p_chol.update(T, dT, H, X, shift, rhf, exc,
                                   denominator=denominator)
    if not freeze_rank:
        assert denominator == 'mp'
        for spin, step in pending_t3.items():
            setattr(dT, spin, step)
        if rhf:
            dT.abb, dT.bbb = dT.aab.copy(), dT.aaa.copy()
    return T, dT


@pytest.mark.parametrize('rhf,empty', [(True,()), (False,()), (False,('aaa',)),
    (False,('aab',)), (False,('abb',)), (False,('bbb',)),
    (True,('aaa','bbb')), (True,('aab','abb')), (False,('aaa','aab','abb','bbb'))])
@pytest.mark.parametrize('denominator', ['mp', 'quasilinear'])
def test_rank_local_full_updates(monkeypatch, rhf, empty, denominator):
    system, T, H, exc = make_case(rhf, empty)
    states = [(copy.deepcopy(T), copy.deepcopy(exc), fresh_x(system)) for _ in range(2)]
    add_t2, add_t3 = ccsdt_p_chol.add_vvvv_t2, ccsdt_p_chol.add_vvvv_t3
    # Two updates catch accidental reuse, stale connectivity and sorted-order mistakes.
    for iteration in range(2):
        outputs = []
        amp, space, X = states[0]
        amp, step = legacy_rank_update(monkeypatch, amp, copy.deepcopy(T),
            copy.deepcopy(H), X, .1, rhf, space, denominator)
        outputs.append((amp, step, space))

        amp, space, X = states[1]
        old, old_space = copy.deepcopy(amp), copy.deepcopy(space)
        seen = []

        def check_t2(T, dT, X, flag_RHF):
            seen.append('t2')
            for spin in ('a', 'b'):
                np.testing.assert_allclose(getattr(T, spin),
                    getattr(old, spin) + getattr(dT, spin), atol=2e-12, rtol=2e-11,
                    err_msg=spin)
                assert np.any(getattr(dT, spin))
            for spin in ('aa', 'ab', 'bb'):
                np.testing.assert_array_equal(getattr(T, spin), getattr(old, spin))
            return add_t2(T, dT, X, flag_RHF)

        def check_t3(T, dT, X, H, t3_excitations, flag_RHF):
            seen.append('t3')
            for spin in ('aa', 'ab', 'bb'):
                actual = getattr(T, spin)
                expected = getattr(old, spin) + getattr(dT, spin)
                if spin != 'ab':
                    # Native updates explicitly restore same-spin symmetry.
                    # Synthetic non-MP corrections need not preserve it.
                    b, a = np.tril_indices(actual.shape[0], -1)
                    j, i = np.tril_indices(actual.shape[2], -1)
                    actual = actual[b[:, None], a[:, None], j, i]
                    expected = expected[b[:, None], a[:, None], j, i]
                np.testing.assert_allclose(actual, expected, atol=2e-12,
                                           rtol=2e-11, err_msg=spin)
                assert np.any(getattr(dT, spin))
            for spin in ('aaa', 'aab', 'abb', 'bbb'):
                np.testing.assert_array_equal(ordered(T, t3_excitations, spin),
                                              ordered(old, old_space, spin))
            return add_t3(T, dT, X, H, t3_excitations, flag_RHF)

        with monkeypatch.context() as patch:
            patch.setattr(ccsdt_p_chol, 'add_vvvv_t2', check_t2)
            patch.setattr(ccsdt_p_chol, 'add_vvvv_t3', check_t3)
            amp, step = ccsdt_p_chol.update(amp, copy.deepcopy(T), copy.deepcopy(H), X, .1,
                rhf, space, denominator=denominator)
        assert seen == (['t2', 't3'] if len(empty) < 4 else ['t2'])
        outputs.append((amp, step, space))
        for spin in T.spin_cases:
            for which in (0, 1):
                a, b = [ordered(out[which], out[2], spin) for out in outputs]
                np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-11,
                    err_msg=f'{spin}, iteration {iteration}, operator {which}')
        states = [(out[0], out[2], state[2]) for out, state in zip(outputs, states)]


@pytest.mark.parametrize('geometry,multiplicity', [
    ('O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', 1),
    ('C 0 0 0; H 0 0 1.12', 2),
], ids=['water-singlet', 'CH-doublet'])
def test_rank_local_converged_physical_energy(monkeypatch, geometry, multiplicity):
    from ccpy import Molecule, RHF, UHF, CCSDT_P

    mol = Molecule(geometry, '6-31g', multiplicity=multiplicity, symmetry=False,
                   cholesky_tei=True, cholesky_tol=1e-10)
    mf = (RHF if multiplicity == 1 else UHF)(mol, verbose=0, e_tol=1e-12, d_tol=1e-10)
    empty = {s: np.ones((1, 6), dtype=np.int32) for s in ('aaa', 'aab', 'abb', 'bbb')}
    cc = CCSDT_P(mf, empty, nfrozen=0, maximum_iterations=120, denominator='mp',
                 energy_convergence=1e-10, amp_convergence=1e-8, amp_print_threshold=1.)
    rng = np.random.default_rng(181)
    exc = {}
    for spin in empty:
        vd = [getattr(cc.system, 'nunoccupied_' + ('alpha' if s == 'a' else 'beta')) for s in spin]
        od = [getattr(cc.system, 'noccupied_' + ('alpha' if s == 'a' else 'beta')) for s in spin]

        def valid(dims):
            return [x for x in itertools.product(*(range(1, d + 1) for d in dims))
                    if all(x[i] < x[j] for i in range(3) for j in range(i + 1, 3)
                           if spin[i] == spin[j])]

        space = np.array([v + o for v in valid(vd) for o in valid(od)], dtype=np.int32)
        exc[spin] = (space[rng.choice(len(space), min(len(space), 80), replace=False)]
                     if len(space) else empty[spin])
    if multiplicity == 1:
        exc['bbb'] = exc['aaa'].copy()
        exc['abb'] = exc['aab'][:, [2, 0, 1, 5, 3, 4]]

    results = []
    for legacy in (False, True):
        cc.set_pspace(copy.deepcopy(exc))
        cc.T = cc._new_operator()
        count = 0

        def step(*args, **kwargs):
            nonlocal count
            count += 1
            if legacy:
                return legacy_rank_update(monkeypatch, *args, **kwargs,
                                          denominator='mp', freeze_rank=False)
            return ccsdt_p_chol.update(*args, **kwargs, denominator='mp')

        cc.T, cc.correlation_energy, converged = cc._run(
            step, cc._new_operator(), cc._intermediates())
        assert converged, (legacy, count)
        assert 1 < count <= 120, (legacy, count)
        results.append((cc.correlation_energy, count))
    assert abs(results[0][0] - results[1][0]) < 1e-9, results
