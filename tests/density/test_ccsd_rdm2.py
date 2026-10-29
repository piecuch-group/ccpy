"""CCSD normal-ordered 2-RDM (ccpy.density.rdm2) against exact determinant-space densities.

The reference is evaluated by brute force, <0|(1 + Lambda) e^-T {r+ s+ q p} e^T|0>, with the
operators built as sparse matrices in the full N-electron space of a small system.
"""
import io
import itertools
import contextlib

import numpy as np
import pytest
import scipy.sparse as sps

from ccpy import Driver
from ccpy.density.rdm1 import calc_rdm1
from ccpy.density.rdm2 import calc_rdm2_normal_order, calc_rdm2_vvvv_slice
from ccpy.integrals import Molecule
from ccpy.interfaces.scf_tools import load_scf_integrals
from ccpy.scf import RHF

BLOCKS = ["".join(c) for c in itertools.product("ov", repeat=4)]


class _FockSpace:
    def __init__(self, nso, nel):
        self.nso = nso
        self.sectors = {}
        for k in (nel, nel - 1, nel - 2):
            dets = [sum(1 << i for i in c) for c in itertools.combinations(range(nso), k)]
            self.sectors[k] = {d: i for i, d in enumerate(dets)}

    def annihilate(self, p, nel):
        """sparse matrix of a_p from the nel- to the (nel-1)-electron sector"""
        src, dst = self.sectors[nel], self.sectors[nel - 1]
        rows, cols, vals = [], [], []
        for d, i in src.items():
            if d >> p & 1:
                sign = -1.0 if bin(d & ((1 << p) - 1)).count("1") % 2 else 1.0
                rows.append(dst[d ^ (1 << p)]); cols.append(i); vals.append(sign)
        return sps.csr_matrix((vals, (rows, cols)), shape=(len(dst), len(src)))

def spin_orbital_amplitudes(X, no, nv, n):
    """ccpy RHF/UHF blocks -> spin-orbital tensors x1[a,i], x2[a,b,i,j] (orbitals a<n alpha, >=n beta)"""
    N = 2 * n
    occ = [list(range(no)), list(range(n, n + no))]
    vir = [list(range(no, n)), list(range(n + no, 2 * n))]
    x1 = np.zeros((N, N)); x2 = np.zeros((N,) * 4)
    for s, blk in ((0, X.a), (1, X.b)):
        x1[np.ix_(vir[s], occ[s])] = blk
    def put(a, b, i, j, val):
        x2[np.ix_(a, b, i, j)] += val
    put(vir[0], vir[0], occ[0], occ[0], X.aa)
    put(vir[1], vir[1], occ[1], occ[1], X.bb)
    put(vir[0], vir[1], occ[0], occ[1], X.ab)
    put(vir[1], vir[0], occ[0], occ[1], -X.ab.transpose(1, 0, 2, 3))
    put(vir[0], vir[1], occ[1], occ[0], -X.ab.transpose(0, 1, 3, 2))
    put(vir[1], vir[0], occ[1], occ[0], X.ab.transpose(1, 0, 3, 2))
    return x1, x2

def exact_densities(T, L, no, n):
    """returns (gammaN[p,q] = <{q^+ p}>, R[p,q,r,s] = <{r^+ s^+ q p}>) over 2n spin orbitals"""
    N, nel = 2 * n, 2 * no
    fs = _FockSpace(N, nel)
    a1 = [fs.annihilate(p, nel) for p in range(N)]
    a2 = [fs.annihilate(p, nel - 1) for p in range(N)]
    dim = len(fs.sectors[nel])
    E = lambda p, q: a1[p].T @ a1[q]   # a_p^+ a_q
    t1, t2 = spin_orbital_amplitudes(T, no, nv := n - no, n)
    l1, l2 = spin_orbital_amplitudes(L, no, nv, n)
    Top = sps.csr_matrix((dim, dim)); Lop = sps.csr_matrix((dim, dim))
    for a, i in zip(*np.nonzero(t1)):
        Top = Top + t1[a, i] * E(a, i)
    for a, i in zip(*np.nonzero(l1)):
        Lop = Lop + l1[a, i] * E(i, a)
    for a, b, i, j in zip(*np.nonzero(t2)):
        # 1/4 t_ij^ab a^+ b^+ j i
        Top = Top + 0.25 * t2[a, b, i, j] * (a1[a].T @ a2[b].T @ a2[j] @ a1[i])
    for a, b, i, j in zip(*np.nonzero(l2)):
        # 1/4 l^ij_ab i^+ j^+ b a
        Lop = Lop + 0.25 * l2[a, b, i, j] * (a1[i].T @ a2[j].T @ a2[b] @ a1[a])
    ref = fs.sectors[nel][sum(1 << i for i in list(range(no)) + list(range(n, n + no)))]
    ket = np.zeros(dim); ket[ref] = 1.0
    bra = ket.copy()
    def expm_vec(op, v, sign):
        out, term = v.copy(), v.copy()
        for k in range(1, 12):
            term = sign * (op @ term) / k
            out = out + term
        return out
    ket = expm_vec(Top, ket, 1.0)                    # e^T |0>
    bra = bra + Lop.T @ bra                          # (1 + Lambda)^+ |0>
    bra = expm_vec(Top.T, bra, -1.0)                 # (e^-T)^+ (1+L)^+|0>
    # full densities
    g = np.array([[bra @ (E(r, p) @ ket) for r in range(N)] for p in range(N)])  # g[p,r] = <r^+ p>
    ket2 = np.array([[a2[q] @ (a1[p] @ ket) for q in range(N)] for p in range(N)])  # a_q a_p ket
    bra2 = np.array([[a2[s] @ (a1[r] @ bra) for r in range(N)] for s in range(N)])  # (r^+ s^+)^+ bra = a_s a_r bra
    # <bra| r^+ s^+ q p |ket> = (a_s a_r bra) . (a_q a_p ket)
    G = np.einsum("srx,pqx->pqrs", bra2, ket2)
    occ = np.zeros(N); occ[:no] = 1; occ[n:n + no] = 1
    gN = g - np.diag(occ)                            # gN[p,r] = <{r^+ p}>
    d = np.eye(N)
    R = G.copy()
    R -= np.einsum("q,sq,pr->pqrs", occ, d, gN)
    R += np.einsum("p,sp,qr->pqrs", occ, d, gN)
    R += np.einsum("q,rq,ps->pqrs", occ, d, gN)
    R -= np.einsum("p,rp,qs->pqrs", occ, d, gN)
    R -= np.einsum("q,p,sq,rp->pqrs", occ, occ, d, d) - np.einsum("q,p,sp,rq->pqrs", occ, occ, d, d)
    return gN, R


def _run_ccsd(geometry, basis, nfrozen=0, **kwargs):
    mol = Molecule(geometry, basis, **kwargs)
    mf = RHF(mol, verbose=0)
    mf.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        driver = Driver.from_scf(mf, nfrozen=nfrozen)
        driver.options["amp_convergence"] = 1.0e-10
        driver.run_cc(method="ccsd")
        driver.run_hbar(method="ccsd")
        driver.run_leftcc(method="left_ccsd")
    return mf, driver


@pytest.mark.parametrize("geometry", [
    "H 0 0 0; H 0 0.1 0.9; H 0 0 1.9; H 0.1 0 2.8; H 0 0 3.8; H 0 0.05 4.75",
    "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.05 -0.7572 -0.4692",
])
def test_rdm2_blocks_exact(geometry):
    _, driver = _run_ccsd(geometry, "sto-3g", auxiliary_basis="def2-universal-jkfit")
    T, L, system = driver.T, driver.L[0], driver.system
    no, n = system.noccupied_alpha, system.norbitals
    gN, R = exact_densities(T, L, no, n)
    A, B = slice(0, n), slice(n, 2 * n)
    sl = {"o": slice(0, no), "v": slice(no, n)}

    rdm1 = calc_rdm1(T, L, system)
    for b in ("oo", "ov", "vo", "vv"):
        ref = gN[A, A][sl[b[0]], sl[b[1]]] + (np.eye(no) if b == "oo" else 0.0)
        assert np.allclose(getattr(rdm1.a, b), ref, atol=1.0e-10)

    rdm2 = calc_rdm2_normal_order(T, L, system, flag_RHF=True)
    exact = {"aa": R[A, A, A, A], "ab": R[A, B, A, B]}
    for spin in ("aa", "ab"):
        for b in BLOCKS:
            arr = getattr(getattr(rdm2, spin), b, None)
            if arr is None:
                continue
            ref = exact[spin][tuple(sl[c] for c in b)]
            assert np.allclose(arr, ref, atol=1.0e-10), f"{spin}.{b}"
    # vvvv blocks, built one slice at a time
    nv = n - no
    for a in range(nv):
        vvvv_aa, vvvv_ab = calc_rdm2_vvvv_slice(T, L, a)
        assert np.allclose(vvvv_aa, exact["aa"][sl["v"], sl["v"], sl["v"], sl["v"]][a], atol=1.0e-10)
        assert np.allclose(vvvv_ab, exact["ab"][sl["v"], sl["v"], sl["v"], sl["v"]][a], atol=1.0e-10)


def test_rdm2_energy_trace():
    """E_corr = sum f gammaN + 1/4 <pq||rs> G_aa (x2 for bb) + <pq|rs> G_ab, with frozen core."""
    mf, driver = _run_ccsd(
        "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692", "cc-pvdz",
        nfrozen=1, auxiliary_basis="cc-pvtz-rifit",
    )
    T, L, system = driver.T, driver.L[0], driver.system
    no, n = system.noccupied_alpha, system.norbitals
    sl = {"o": slice(0, no), "v": slice(no, n)}
    with contextlib.redirect_stdout(io.StringIO()):
        _, H = load_scf_integrals(mf, nfrozen=1, sorted=False)

    rdm1 = calc_rdm1(T, L, system)
    g = np.zeros((n, n))
    for b in ("oo", "ov", "vo", "vv"):
        g[sl[b[0]], sl[b[1]]] = getattr(rdm1.a, b)
    gN = g.T - np.diag([1.0] * no + [0.0] * (n - no))

    rdm2 = calc_rdm2_normal_order(T, L, system, flag_RHF=True)
    Raa, Rab = np.zeros((n,) * 4), np.zeros((n,) * 4)
    for b in BLOCKS:
        idx = tuple(sl[c] for c in b)
        if getattr(rdm2.aa, b, None) is not None:
            arr = rdm2.aa.__dict__[b]
            Raa[idx] = arr
            Raa[(idx[1], idx[0], idx[2], idx[3])] = -arr.transpose(1, 0, 2, 3)
            Raa[(idx[0], idx[1], idx[3], idx[2])] = -arr.transpose(0, 1, 3, 2)
            Raa[(idx[1], idx[0], idx[3], idx[2])] = arr.transpose(1, 0, 3, 2)
        if rdm2.ab.__dict__[b] is not None:
            Rab[idx] = rdm2.ab.__dict__[b]
    for a in range(n - no):
        vvvv_aa, vvvv_ab = calc_rdm2_vvvv_slice(T, L, a)
        Raa[no + a, no:, no:, no:] = vvvv_aa
        Rab[no + a, no:, no:, no:] = vvvv_ab
    energy = (
        2.0 * np.sum(H.a * gN)
        + 2.0 * 0.25 * np.einsum("pqrs,rspq->", H.aa, Raa)
        + np.einsum("pqrs,rspq->", H.ab, Rab)
    )
    assert np.isclose(energy, driver.correlation_energy, atol=1.0e-10)
