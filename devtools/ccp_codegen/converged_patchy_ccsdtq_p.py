"""Converged patchy-P check: dense CCSDTQ restricted to a P space vs CCSDTQ(P).

The dense reference is ccsdtq.py with boolean P-space masks (pspace_aaa, ..., pspace_bbbb): after
every T3/T4 spin case, amplitudes and residuals outside P are zeroed, so only P-space T3/T4 are ever
updated. It saves no CPU, but converges to exactly the CC(P) fixed point for that P space.

usage: OMP_NUM_THREADS=4 ~/miniforge3/envs/ccpy/bin/python devtools/ccp_codegen/converged_patchy_ccsdtq_p.py \\
       [basis] [scale ...] [--frac 0.3]
       scale = R(OH)/Re for the symmetric stretch of H2O (Re = 1.84345 bohr), e.g. 1.0 1.5 2.0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).parent))
from testdeck_ccsdtq_p import SPIN3, SPIN4, canonical_rows, mask_from_rows  # noqa: E402

from ccpy import Driver  # noqa: E402
from ccpy.drivers.solvers import cc_jacobi  # noqa: E402
from ccpy.models.integrals import Integral  # noqa: E402
from ccpy.models.operators import ClusterOperator  # noqa: E402


def masked_update(masks):
    """ccsdtq.update with only P-space T3/T4 updated (UHF code path)."""
    from ccpy.cc import ccsdtq
    from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
    from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
    from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates

    from ccpy.lib.core import cc_loops_t4
    flat = {sp: np.asfortranarray(m).ravel(order="F").view(np.int8) for sp, m in masks.items()}

    def restrict(T, dT, sp):
        # Fortran zeroes T and dT outside P in place on flat column-major views (no copies)
        t, r = np.asfortranarray(getattr(T, sp)), np.asfortranarray(getattr(dT, sp))
        tv, rv = t.ravel(order="F"), r.ravel(order="F")
        out = cc_loops_t4.apply_pspace_mask(tv, rv, flat[sp])
        assert np.shares_memory(out[0], t) and np.shares_memory(out[1], r)
        setattr(T, sp, t)
        setattr(dT, sp, r)

    def update(T, dT, H, X, shift, flag_RHF):
        X = get_pre_ccs_intermediates(X, T, H, False)
        T, dT = ccsdtq.update_t1a(T, dT, H, X, shift)
        T, dT = ccsdtq.update_t1b(T, dT, H, X, shift)
        X = get_ccs_intermediates_opt(X, T, H, False)
        for f in (ccsdtq.update_t2a, ccsdtq.update_t2b, ccsdtq.update_t2c):
            T, dT = f(T, dT, X, H, shift)
        X = get_ccsd_intermediates(T, X, H, False)
        for sp, f in zip(SPIN3, (ccsdtq.update_t3a, ccsdtq.update_t3b, ccsdtq.update_t3c, ccsdtq.update_t3d)):
            T, dT = f(T, dT, X, H, shift)
            restrict(T, dT, sp)
        X = add_VT3_intermediates(T, X, H, False)
        for sp, f in zip(SPIN4, (ccsdtq.update_t4a, ccsdtq.update_t4b, ccsdtq.update_t4c,
                                 ccsdtq.update_t4d, ccsdtq.update_t4e)):
            T, dT = f(T, dT, X, H, shift)
            restrict(T, dT, sp)
        return T, dT

    return update


def make_driver(basis, scale):
    geometry = [["O", (0.0, 0.0, -0.0090 * scale)],
                ["H", (0.0, 1.515263 * scale, -1.058898 * scale)],
                ["H", (0.0, -1.515263 * scale, -1.058898 * scale)]]
    mol = gto.M(atom=geometry, basis=basis, charge=0, spin=0, symmetry="C2V", cart=False, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=1)
    driver.options["RHF_symmetry"] = False   # a random P space is not spin symmetric
    driver.options["energy_convergence"] = 1.0e-10
    driver.options["amp_convergence"] = 1.0e-08
    driver.options["maximum_iterations"] = 200
    return driver


def run(basis, scale, frac, seed):
    driver = make_driver(basis, scale)
    sysm = driver.system
    nv = {"a": sysm.nunoccupied_alpha, "b": sysm.nunoccupied_beta}
    no = {"a": sysm.noccupied_alpha, "b": sysm.noccupied_beta}
    rng = np.random.default_rng(seed)
    rows = {}
    for sp in SPIN3 + SPIN4:
        r = canonical_rows(sp, nv, no)
        rows[sp] = r[rng.random(len(r)) < frac]
        assert len(rows[sp]) > 0, sp

    # dense CCSDTQ restricted to P
    driver.set_operator_params("ccsdtq")
    driver.options["method"] = "CCSDTQ"
    T = ClusterOperator(sysm, order=4)
    dT = ClusterOperator(sysm, order=4)
    masks = {sp: mask_from_rows(sp, rows[sp], getattr(T, sp).shape) for sp in rows}
    X = Integral.from_empty(sysm, 2, data_type=driver.hamiltonian.a.oo.dtype, use_none=True)
    t0 = time.perf_counter()
    T, e_dense, _ = cc_jacobi(masked_update(masks), T, dT, driver.hamiltonian, X, driver.options)
    t_dense = time.perf_counter() - t0

    # CCSDTQ(P), same P space
    driver2 = make_driver(basis, scale)
    t0 = time.perf_counter()
    pexc = {sp: np.asfortranarray(rows[sp]) for sp in SPIN3 + SPIN4}   # sorted in place by CC(P)
    driver2.run_ccp(method="ccsdtq_p",
                    t3_excitations={sp: pexc[sp] for sp in SPIN3},
                    t4_excitations={sp: pexc[sp] for sp in SPIN4})
    t_ccp = time.perf_counter() - t0
    e_ccp = driver2.correlation_energy

    # amplitude check on the P space
    worst = 0.0
    for sp in SPIN3 + SPIN4:
        ref = np.array([getattr(T, sp)[tuple(r - 1)] for r in pexc[sp]])
        worst = max(worst, np.abs(getattr(driver2.T, sp) - ref).max())
    n3 = sum(len(rows[sp]) for sp in SPIN3)
    n4 = sum(len(rows[sp]) for sp in SPIN4)
    return dict(basis=basis, scale=scale, n3=n3, n4=n4, e_dense=e_dense, e_ccp=e_ccp,
                de=e_ccp - e_dense, damp=worst, t_dense=t_dense, t_ccp=t_ccp)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("basis", nargs="?", default="6-31g")
    p.add_argument("scales", nargs="*", type=float, default=[1.0, 1.5, 2.0])
    p.add_argument("--frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args()
    results = [run(a.basis, s, a.frac, a.seed) for s in a.scales]
    print("\n basis     R/Re   |P3|     |P4|      E(dense-P)          E(CCSDTQ(P))      diff      max|dT_P|  t_dense  t_ccp")
    for r in results:
        print(f" {r['basis']:8s} {r['scale']:4.2f} {r['n3']:7d} {r['n4']:9d}  {r['e_dense']:18.12f} {r['e_ccp']:18.12f}"
              f"  {r['de']:9.2e}  {r['damp']:9.2e}  {r['t_dense']:7.1f}  {r['t_ccp']:6.1f}")
        assert abs(r["de"]) < 1.0e-9 and r["damp"] < 1.0e-6, r
