import copy, itertools
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from ccpy.cc import ccsdt_p_chol
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator

def test_complete_updates_against_original_layout(monkeypatch):
    """Frozen full-update oracle from the pre-layout implementation, seed 792.

    All spin cases, empty and nonempty P, and nonzero input T3 are covered.
    The synthetic tensors intentionally avoid relying on physical symmetries
    beyond the RHF copies, making axis-order mistakes easier to detect.
    """
    reference = np.load(Path(__file__).with_name('data') / 'ccsdt_p_chol_layout_reference.npz')
    rng = np.random.default_rng(792)
    sys = NS(norbitals=8, noccupied_alpha=4, noccupied_beta=4, nunoccupied_alpha=4, nunoccupied_beta=4)
    for rhf, empty in itertools.product((True, False), (True, False)):
        mats = {s: rng.normal(scale=0.01, size=(8,) * (2 * len(s))) for s in ('a', 'b', 'aa', 'ab', 'bb')}
        for s in ('a', 'b'):
            mats[s] = np.diag(np.r_[np.arange(-4, 0), np.arange(1, 5)]) + 0.001 * mats[s]
        if rhf:
            mats['b'] = mats['a'].copy()
            mats['bb'] = mats['aa'].copy()
        chol = NS()
        for spin in ('a', 'b'):
            full = rng.normal(scale=0.03, size=(7, 8, 8))
            full += full.swapaxes(1, 2)
            setattr(chol, spin, NS(**{p: full[:, slice(0, 4) if p[0] == 'o' else slice(4, 8), slice(0, 4) if p[1] == 'o' else slice(4, 8)] for p in ('oo', 'ov', 'vo', 'vv')}))
        if rhf:
            chol.b = copy.deepcopy(chol.a)
        H = Integral(sys, 2, mats, chol=chol)
        exc = {s: np.asfortranarray([[1, 1, 1, 1, 1, 1] if empty else [1, 2, 3, 1, 2, 3] if s in ('aaa', 'bbb') else [1, 2, 1, 1, 2, 1] if s == 'aab' else [1, 1, 2, 1, 1, 2]], dtype=np.int32) for s in ('aaa', 'aab', 'abb', 'bbb')}
        T = ClusterOperator(sys, 3, p_orders=[3], pspace_sizes=[[1, 1, 1, 1]])
        dT = copy.deepcopy(T)
        for s in T.spin_cases:
            setattr(T, s, np.asfortranarray(rng.normal(scale=0.001, size=getattr(T, s).shape)))
        if rhf:
            T.b = T.a.copy()
            T.bb = T.aa.copy()
            T.abb = T.aab.copy()
            T.bbb = T.aaa.copy()
        X = Integral.from_empty(sys, 2, use_none=True)
        X.chol = NS(a=NS(), b=NS())
        # The stored oracle uses the original spin-by-spin update ordering.
        # Rank-local ordering is checked independently against frozen legacy
        # equations in test_shared_chol_ppl.
        from test_shared_chol_ppl import legacy_rank_update
        t, dt = legacy_rank_update(monkeypatch, copy.deepcopy(T), copy.deepcopy(dT),
            copy.deepcopy(H), X, 0.0, rhf, copy.deepcopy(exc), 'mp', freeze_rank=False)
        for which, operator in enumerate((t, dt)):
            for spin in T.spin_cases:
                np.testing.assert_allclose(getattr(operator, spin), reference[f'{rhf}_{empty}_{which}_{spin}'], rtol=2e-11, atol=2e-12)
        assert X.chol.a.vv.transpose(0, 2, 1).flags.f_contiguous
        if rhf:
            assert X.chol.a.vv is X.chol.b.vv
        if not empty:
            assert X.aa.vvov.flags.f_contiguous
            assert X.ab.vvov.flags.f_contiguous


def test_one_body_exchange_association():
    from ccpy.hbar.hbar_ccsd_chol import contract_vv_t1_exchange

    rng = np.random.default_rng(816)
    vv = rng.normal(size=(7, 5, 6))
    ov = rng.normal(size=(7, 3, 4))
    t1 = rng.normal(size=(6, 3))
    expected = -np.einsum('xaf,xne,fn->ae', vv, ov, t1, optimize=False)
    for pack in (np.ascontiguousarray, np.asfortranarray):
        actual = contract_vv_t1_exchange(pack(vv), pack(ov), pack(t1))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
