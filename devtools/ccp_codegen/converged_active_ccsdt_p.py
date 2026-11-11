"""CCSDt check: dense CCSDT with a boolean T3 mask vs native CCSDT(P), same active-triples P space.

H2O cc-pVDZ (geometry of tests/h2o/test_cctq34_h2o.py::test_cctq34_h2o_ccpvdz), (2,2) active space, nfrozen=0.
Runs: full CCSDT, dense CCSDT masked to get_active_triples_pspace (A1), dense CCSDT masked to the active
criterion alone (no symmetry filter), and CCSDT(P) on the same P space.

usage: OMP_NUM_THREADS=4 ~/miniforge3/envs/ccpy/bin/python devtools/ccp_codegen/converged_active_ccsdt_p.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from testdeck_ccsdtq_p import SPIN3, mask_from_rows  # noqa: E402

from ccpy import Driver, Molecule, RHF, get_active_triples_pspace  # noqa: E402
from ccpy.drivers.solvers import cc_jacobi  # noqa: E402
from ccpy.models.integrals import Integral  # noqa: E402
from ccpy.models.operators import ClusterOperator  # noqa: E402

E_FCI = -76.072348


def make_driver():
    mol = Molecule('''O 0.0 0.0 -0.0135
                      H 0.0 2.2728945 -1.588347
                      H 0.0 -2.2728945 -1.588347''',
                   basis="cc-pvdz", unit="bohr", charge=0, multiplicity=1, symmetry=True,
                   cartesian=False, cholesky_tei=True, cholesky_tol=1.0e-14)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    driver.options["energy_convergence"] = 1.0e-10
    driver.options["amp_convergence"] = 1.0e-08
    driver.options["maximum_iterations"] = 200
    driver.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
    return driver


def masked_update(masks):
    """ccsdt.update (UHF path) with T3 and dT3 zeroed outside P after every spin case."""
    from ccpy.cc import ccsdt
    from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
    from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates

    def update(T, dT, H, X, shift, flag_RHF, **kw):
        X = get_pre_ccs_intermediates(X, T, H, False)
        T, dT = ccsdt.update_t1a(T, dT, H, X, shift)
        T, dT = ccsdt.update_t1b(T, dT, H, X, shift)
        X = get_ccs_intermediates_opt(X, T, H, False)
        for f in (ccsdt.update_t2a, ccsdt.update_t2b, ccsdt.update_t2c):
            T, dT = f(T, dT, X, H, shift)
        X = get_ccsd_intermediates(T, X, H, False)
        for sp, f in zip(SPIN3, (ccsdt.update_t3a, ccsdt.update_t3b, ccsdt.update_t3c, ccsdt.update_t3d)):
            T, dT = f(T, dT, X, H, shift)
            getattr(T, sp)[~masks[sp]] = 0.0
            getattr(dT, sp)[~masks[sp]] = 0.0
        return T, dT
    return update


def run_dense(driver, masks=None):
    driver.set_operator_params("ccsdt")
    driver.options["method"] = "CCSDT"
    T = ClusterOperator(driver.system, order=3)
    dT = ClusterOperator(driver.system, order=3)
    X = Integral.from_empty(driver.system, 2, data_type=driver.hamiltonian.a.oo.dtype, use_none=True)
    from ccpy.cc import ccsdt
    upd = ccsdt.update if masks is None else masked_update(masks)
    opts = dict(driver.options, RHF_symmetry=False)
    T, e, _ = cc_jacobi(upd, T, dT, driver.hamiltonian, X, opts)
    return T, e


def active_rows(sp, sysm):
    """Canonical triples with >=1 active occupied and >=1 active virtual, no symmetry filter (1-based)."""
    no = {"a": sysm.noccupied_alpha, "b": sysm.noccupied_beta}
    nv = {"a": sysm.nunoccupied_alpha, "b": sysm.nunoccupied_beta}
    nao, nav = sysm.num_act_occupied_alpha, sysm.num_act_unoccupied_alpha

    def part(n, s):
        k = sp.count(s)
        return list(itertools.combinations(range(n[s]), k)) if k else [()]
    rows = []
    for va in part(nv, "a"):
        for vb in part(nv, "b"):
            if sum(x < nav for x in va + vb) < 1:
                continue
            for oa in part(no, "a"):
                for ob in part(no, "b"):
                    if sum(i >= no["a"] - nao for i in oa + ob) < 1:
                        continue
                    rows.append([x + 1 for x in va + vb + oa + ob])
    return np.array(rows, dtype=np.int32)


if __name__ == "__main__":
    driver = make_driver()
    sysm = driver.system
    eref = sysm.reference_energy

    pex = get_active_triples_pspace(sysm, target_irrep="A1")
    pex = {sp: np.asarray(pex[sp], dtype=np.int32) for sp in SPIN3}
    shape = {sp: getattr(ClusterOperator(sysm, order=3), sp).shape for sp in SPIN3}
    mask_sym = {sp: mask_from_rows(sp, pex[sp], shape[sp]) for sp in SPIN3}
    rows_nosym = {sp: active_rows(sp, sysm) for sp in SPIN3}
    mask_nosym = {sp: mask_from_rows(sp, rows_nosym[sp], shape[sp]) for sp in SPIN3}

    _, e_full = run_dense(driver)
    T_sym, e_sym = run_dense(driver, mask_sym)
    T_nos, e_nos = run_dense(driver, mask_nosym)

    # amplitudes in the no-symmetry mask but outside the A1 P space must vanish by symmetry
    leak = max(np.abs(getattr(T_nos, sp)[mask_nosym[sp] & ~mask_sym[sp]]).max(initial=0.0) for sp in SPIN3)

    d2 = make_driver()
    d2.run_ccp(method="ccsdt_p", t3_excitations={sp: np.asfortranarray(pex[sp]) for sp in SPIN3})
    e_ccp = d2.correlation_energy
    # d2.run_ccp sorted pex in place to match d2.T; abb/bbb lists are rebuilt from aab/aaa under RHF symmetry
    damp = max(np.abs(getattr(d2.T, sp) - np.array([getattr(T_sym, sp)[tuple(r - 1)] for r in pex[sp]])).max()
               for sp in ("aaa", "aab"))

    print("\n |P| (A1) =", {sp: len(pex[sp]) for sp in SPIN3}, " |P| (no sym) =", {sp: len(rows_nosym[sp]) for sp in SPIN3})
    for name, e in [("CCSDT (full dense)", e_full), ("CCSDt dense mask A1", e_sym),
                    ("CCSDt dense mask nosym", e_nos), ("CCSDt CCSDT(P)", e_ccp)]:
        print(f" {name:24s} E = {eref + e:16.10f}   E - E(FCI) = {1000 * (eref + e - E_FCI):8.4f} mEh")
    print(f" CCSDT(P) - dense(A1) = {e_ccp - e_sym:.2e}   max|dT3_P| = {damp:.2e}   non-A1 leak = {leak:.2e}")
