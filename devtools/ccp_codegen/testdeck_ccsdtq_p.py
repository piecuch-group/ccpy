"""CCSDTQ(P) testdeck: one dense ccsdtq.update vs one ccsdtq_p.update for H2O/6-31g.

T1-T4 come from a few dense CCSDTQ iterations (so all amplitudes are nontrivial). For a P space
(full, or a random patch of the triples and quadruples) the dense reference is ccsdtq.update with
T3/T4 outside P zeroed initially and after each spin case's update -- exactly what CC(P) sees.
Residuals and amplitudes are compared on T1, T2 and every P-space T3/T4 row.

usage: PATH=~/miniforge3/envs/ccpy/bin:$PATH OMP_NUM_THREADS=4 \\
       ~/miniforge3/envs/ccpy/bin/python devtools/ccp_codegen/testdeck_ccsdtq_p.py <dir with ccsdtq_p_loops*.so>
"""
import copy
import importlib.util
import itertools
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SPIN3 = ["aaa", "aab", "abb", "bbb"]
SPIN4 = ["aaaa", "aaab", "aabb", "abbb", "bbbb"]


def load_loops(d):
    so = next(Path(d).glob("ccsdtq_p_loops*.so"))
    spec = importlib.util.spec_from_file_location("ccsdtq_p_loops", so)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ccsdtq_p_loops


def canonical_rows(sp, nv, no):
    """All excitations (1-based) with ascending indices inside each spin block."""
    def part(n, spin):
        out = [()]
        for s in "ab":
            k = spin.count(s)
            if k:
                out = [o + c for o in out for c in itertools.combinations(range(1, n[s] + 1), k)]
        return out
    return np.array([v + o for v in part(nv, sp) for o in part(no, sp)], dtype=np.int32)


def perm_sign(p):
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def block_perms(sp):
    """Signed axis permutations of a dense T3/T4 within its same-spin blocks."""
    n = len(sp)
    groups = []
    for off in (0, n):
        for s in "ab":
            g = [off + p for p in range(n) if sp[p] == s]
            if g:
                groups.append(g)
    for perms in itertools.product(*[itertools.permutations(g) for g in groups]):
        axes = list(range(2 * n))
        sign = 1
        for g, p in zip(groups, perms):
            sign *= perm_sign([g.index(x) for x in p])
            for a, b in zip(g, p):
                axes[a] = b
        yield sign, axes


def mask_from_rows(sp, rows, shape):
    m = np.zeros(shape, dtype=bool)
    for r in rows:
        m[tuple(r - 1)] = True
    out = np.zeros(shape, dtype=bool)
    for _, axes in block_perms(sp):
        out |= m.transpose(axes)
    return out


def dense_update(T, dT, H, X, shift, masks):
    """ccsdtq.update with T3/T4 restricted to the P space after every spin case."""
    from ccpy.cc import ccsdtq
    from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
    from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
    from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates
    X = get_pre_ccs_intermediates(X, T, H, False)
    T, dT = ccsdtq.update_t1a(T, dT, H, X, shift)
    T, dT = ccsdtq.update_t1b(T, dT, H, X, shift)
    X = get_ccs_intermediates_opt(X, T, H, False)
    for f in (ccsdtq.update_t2a, ccsdtq.update_t2b, ccsdtq.update_t2c):
        T, dT = f(T, dT, X, H, shift)
    X = get_ccsd_intermediates(T, X, H, False)
    for sp, f in zip(SPIN3, (ccsdtq.update_t3a, ccsdtq.update_t3b, ccsdtq.update_t3c, ccsdtq.update_t3d)):
        T, dT = f(T, dT, X, H, shift)
        setattr(T, sp, getattr(T, sp) * masks[sp])
    X = add_VT3_intermediates(T, X, H, False)
    for sp, f in zip(SPIN4, (ccsdtq.update_t4a, ccsdtq.update_t4b, ccsdtq.update_t4c, ccsdtq.update_t4d,
                             ccsdtq.update_t4e)):
        T, dT = f(T, dT, X, H, shift)
        setattr(T, sp, getattr(T, sp) * masks[sp])
    return T, dT


def main():
    import ccpy.lib.core
    ccpy.lib.core.ccsdtq_p_loops = load_loops(sys.argv[1])   # standalone build, not yet in meson
    from pyscf import gto, scf
    from ccpy import Driver
    from ccpy.models.integrals import Integral
    from ccpy.cc import ccsdtq_p

    mol = gto.M(atom=[["O", (0.0, 0.0, -0.0180)], ["H", (0.0, 3.030526, -2.117796)],
                      ["H", (0.0, -3.030526, -2.117796)]], basis="6-31g", unit="Bohr", spin=0, symmetry="C2V")
    mf = scf.UHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["RHF_symmetry"] = False
    driver.options["maximum_iterations"] = 4
    driver.run_cc(method="ccsdtq")
    H, sysm = driver.hamiltonian, driver.system
    T0 = driver.T
    noa, nob, nua, nub = H.ab.oovv.shape
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    shift = 0.0
    rng = np.random.default_rng(2024)

    for label, frac in (("full P", 1.0), ("patchy P", 0.5)):
        rows = {}
        for sp in SPIN3 + SPIN4:
            r = canonical_rows(sp, nv, no)
            keep = rng.random(len(r)) < frac if frac < 1.0 else np.ones(len(r), dtype=bool)
            rows[sp] = r[keep]
        masks = {sp: mask_from_rows(sp, rows[sp], getattr(T0, sp).shape) for sp in rows}

        # dense reference
        Td = copy.deepcopy(T0)
        for sp in rows:
            setattr(Td, sp, getattr(Td, sp) * masks[sp])
        Tin = copy.deepcopy(Td)
        dTd = copy.deepcopy(Td)
        for sp in ["a", "b", "aa", "ab", "bb"] + SPIN3 + SPIN4:
            getattr(dTd, sp)[...] = 0.0
        X = Integral.from_empty(sysm, 2, data_type=H.a.oo.dtype, use_none=True)
        t0 = time.time()
        Td, dTd = dense_update(Td, dTd, H, X, shift, masks)
        td = time.time() - t0

        # CC(P)
        Tp = SimpleNamespace(**{sp: getattr(Tin, sp).copy() for sp in ["a", "b", "aa", "ab", "bb"]})
        dTp = SimpleNamespace(**{sp: np.zeros_like(getattr(Tin, sp)) for sp in ["a", "b", "aa", "ab", "bb"]})
        t3x, t4x = {}, {}
        for sp in SPIN3 + SPIN4:
            r = rows[sp]
            amps = np.array([getattr(Tin, sp)[tuple(x - 1)] for x in r]) if len(r) else np.zeros(1)
            if not len(r):
                r = np.ones((1, 2 * len(sp)), dtype=np.int32)
            setattr(Tp, sp, amps.copy())
            setattr(dTp, sp, np.zeros_like(amps))
            (t3x if len(sp) == 3 else t4x)[sp] = np.asfortranarray(r)
        Xp = Integral.from_empty(sysm, 2, data_type=H.a.oo.dtype, use_none=True)
        t0 = time.time()
        Tp, dTp = ccsdtq_p.update(Tp, dTp, H, Xp, shift, False, t3x, t4x)
        tp = time.time() - t0

        print(f"\n{label}: dense {td:.2f} s, CC(P) {tp:.2f} s")
        worst = 0.0
        for sp in ["a", "b", "aa", "ab", "bb"]:
            e1 = np.abs(getattr(Tp, sp) - getattr(Td, sp)).max()
            e2 = np.abs(getattr(dTp, sp) - getattr(dTd, sp)).max()
            worst = max(worst, e1, e2)
            print(f"   T{len(sp)}{sp:5s}  |dT| err {e2:.2e}   |T| err {e1:.2e}")
        for sp in SPIN3 + SPIN4:
            x = (t3x if len(sp) == 3 else t4x)[sp]
            if np.all(x == 1) and x.shape[0] == 1:
                continue
            ref_t = np.array([getattr(Td, sp)[tuple(r - 1)] for r in x])
            ref_r = np.array([getattr(dTd, sp)[tuple(r - 1)] for r in x])
            got = getattr(dTp, sp)
            if len(sp) == 4:   # cc_loops_t4 keeps the undivided residual; CC(P) keeps the increment
                fo = {"a": np.diag(H.a.oo), "b": np.diag(H.b.oo)}
                fv = {"a": np.diag(H.a.vv), "b": np.diag(H.b.vv)}
                den = np.array([sum(fo[s][r[4 + q] - 1] - fv[s][r[q] - 1] for q, s in enumerate(sp)) for r in x])
                got = got * (den - shift)
            e1 = np.abs(getattr(Tp, sp) - ref_t).max()
            e2 = np.abs(got - ref_r).max()
            worst = max(worst, e1, e2)
            print(f"   T{len(sp)}{sp:5s}  n={len(x):7d}  |dT| err {e2:.2e}   |T| err {e1:.2e}   max|T| {np.abs(ref_t).max():.2e}")
        print(f"   worst difference {worst:.2e}")


if __name__ == "__main__":
    main()
