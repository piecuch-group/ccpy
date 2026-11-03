"""Spin-orbital reference evaluator of the CR-CC(2,3) triples correction.

For every triple K = (i<j<k, a<b<c) it exposes the three factors of

    delta = sum_K M_K N_K / D_K,

    M_K = <K| Hbar |0>                       (moment of the CCSD equations)
    N_K = <0| (1 + Lambda) Hbar |K>          (left-CCSD projection)
    D_K = E_CCSD - <K| Hbar |K>              (denominators of variants A-D)

which ccpy's Fortran kernels (ccpy/lib/fortran/crcc_loops.f90) only return as sums. This
is the starting point of the analytic CR-CC(2,3) gradient: the reverse-mode seeds are
dδ/dM_K = N_K/D_K, dδ/dN_K = M_K/D_K and dδ/dD_K = -M_K N_K/D_K^2.

Everything is written with dense spin-orbital tensors, so the expressions read like the
equations and can be checked term by term; the cost is memory (spin-orbital arrays are
up to 64 times the spin-integrated ones), so this module is meant for small molecules.

Spin-orbital ordering: occupied = [alpha occupied, beta occupied], virtual = [alpha
virtual, beta virtual]. Integral layouts follow ccpy: v[p,q,r,s] = <pq||rs>,
t2[a,b,i,j], l2[a,b,i,j], h1[p,q].
"""

import numpy as np


# --------------------------------------------------------------------------------------
# spin-integrated ccpy blocks -> dense spin-orbital tensors
# --------------------------------------------------------------------------------------
def _full_block(block, spins, system):
    """Reassemble a SortedIntegral (e.g. H.ab) into one array over all correlated orbitals."""
    n = system.norbitals
    ranges = {
        "a": {"o": slice(0, system.noccupied_alpha), "v": slice(system.noccupied_alpha, n)},
        "b": {"o": slice(0, system.noccupied_beta), "v": slice(system.noccupied_beta, n)},
    }
    full = np.zeros((n,) * len(spins))
    for name in block.slices:
        idx = tuple(ranges[s][ov] for s, ov in zip(spins, name))
        full[idx] = getattr(block, name)
    return full


class SpinOrbitalSpace:
    """Index bookkeeping between ccpy's spin-integrated blocks and spin orbitals."""

    def __init__(self, system):
        n = system.norbitals
        noa, nob = system.noccupied_alpha, system.noccupied_beta
        # spin orbitals in "spin-blocked" order: alpha 0..n-1, beta n..2n-1
        occ = list(range(noa)) + [n + i for i in range(nob)]
        vir = list(range(noa, n)) + [n + a for a in range(nob, n)]
        self.order = np.array(occ + vir)
        self.nocc, self.nvir = len(occ), len(vir)
        self.n = n
        self.system = system
        self.o = slice(0, self.nocc)
        self.v = slice(self.nocc, self.nocc + self.nvir)

    def one_body(self, op):
        """Spin-orbital one-body array (occ+vir ordering) from an Integral-like op with .a/.b."""
        n = self.n
        h = np.zeros((2 * n, 2 * n))
        h[:n, :n] = _full_block(op.a, "aa", self.system)
        h[n:, n:] = _full_block(op.b, "bb", self.system)
        return h[np.ix_(self.order, self.order)]

    def two_body(self, op):
        """Antisymmetric spin-orbital two-body array from .aa/.ab/.bb (layout <pq||rs>)."""
        n = self.n
        v = np.zeros((2 * n,) * 4)
        a, b = slice(0, n), slice(n, 2 * n)
        v[a, a, a, a] = _full_block(op.aa, "aaaa", self.system)
        v[b, b, b, b] = _full_block(op.bb, "bbbb", self.system)
        ab = _full_block(op.ab, "abab", self.system)
        v[a, b, a, b] = ab
        v[b, a, b, a] = ab.transpose(1, 0, 3, 2)
        v[a, b, b, a] = -ab.transpose(0, 1, 3, 2)
        v[b, a, a, b] = -ab.transpose(1, 0, 2, 3)
        o = self.order
        return v[np.ix_(o, o, o, o)]

    def t1(self, op):
        """[a, i] spin-orbital singles from op.a / op.b (ccpy layout [vir, occ])."""
        s = self.system
        x = np.zeros((self.nvir, self.nocc))
        nua, noa = s.nunoccupied_alpha, s.noccupied_alpha
        x[:nua, :noa] = op.a
        x[nua:, noa:] = op.b
        return x

    def t2(self, op):
        """[a, b, i, j] antisymmetric spin-orbital doubles from op.aa / op.ab / op.bb."""
        s = self.system
        nua, noa = s.nunoccupied_alpha, s.noccupied_alpha
        A, B = slice(0, nua), slice(nua, self.nvir)
        I, J = slice(0, noa), slice(noa, self.nocc)
        x = np.zeros((self.nvir, self.nvir, self.nocc, self.nocc))
        x[A, A, I, I] = op.aa
        x[B, B, J, J] = op.bb
        x[A, B, I, J] = op.ab
        x[B, A, J, I] = op.ab.transpose(1, 0, 3, 2)
        x[A, B, J, I] = -op.ab.transpose(0, 1, 3, 2)
        x[B, A, I, J] = -op.ab.transpose(1, 0, 2, 3)
        return x


def spin_orbital_quantities(driver, state=0):
    """Spin-orbital T, Lambda, bare Fock, Hbar and bare <oo||vv> of a driver after run_cc,
    run_hbar and run_leftcc (ground state, CCSD)."""
    space = SpinOrbitalSpace(driver.system)
    o, v = space.o, space.v
    f = space.one_body(driver.fock)
    h1 = space.one_body(driver.hamiltonian)
    h2 = space.two_body(driver.hamiltonian)
    return {
        "space": space,
        "t1": space.t1(driver.T),
        "t2": space.t2(driver.T),
        "l1": space.t1(driver.L[state]),
        "l2": space.t2(driver.L[state]),
        "f_oo": np.diag(f[o, o]).copy(),
        "f_vv": np.diag(f[v, v]).copy(),
        "h_oo": h1[o, o],
        "h_vv": h1[v, v],
        "h_ov": h1[o, v],
        "h_vooo": h2[v, o, o, o],
        "h_vvov": h2[v, v, o, v],
        "h_vovv": h2[v, o, v, v],
        "h_ooov": h2[o, o, o, v],
        "h_voov": h2[v, o, o, v],
        "h_oooo": h2[o, o, o, o],
        "h_vvvv": h2[v, v, v, v],
        # Hbar(oovv) = <oo||vv>, the bare integrals
        "v_oovv": h2[o, o, v, v],
    }


# --------------------------------------------------------------------------------------
# M_K, N_K, D_K for one occupied triple
# --------------------------------------------------------------------------------------
def _antisymmetrize_abc(x):
    """A_abc x = x(abc) + x(bca) + x(cab) - x(acb) - x(bac) - x(cba)."""
    return (x + x.transpose(1, 2, 0) + x.transpose(2, 0, 1)
            - x.transpose(0, 2, 1) - x.transpose(1, 0, 2) - x.transpose(2, 1, 0))


def triples_intermediates(q):
    """Quantities shared by all occupied triples."""
    t2, h_ov = q["t2"], q["h_ov"]
    # H(vvov) + h(me) t(ab,im): the h(ov) T2 part of <K|Hbar|0> is carried here
    I_vvov = q["h_vvov"] + np.einsum("me,abim->abie", h_ov, t2, optimize=True)
    # diagonal of <ijkabc|(V_N T2)_C|ijkabc> (three-body part of the D denominator)
    d3_o = np.einsum("ijae,aeij->aij", q["v_oovv"], t2, optimize=True)        # d3_o[a,i,j]
    d3_v = -np.einsum("imab,abim->aib", q["v_oovv"], t2, optimize=True)       # d3_v[a,i,b]
    voov_diag = np.einsum("aiia->ai", q["h_voov"])                            # [a,i]
    oooo_diag = np.einsum("jiji->ij", q["h_oooo"])                            # [i,j]
    vvvv_diag = np.einsum("baba->ab", q["h_vvvv"])                            # [a,b]
    return {"I_vvov": I_vvov, "d3_o": d3_o, "d3_v": d3_v,
            "voov_diag": voov_diag, "oooo_diag": oooo_diag, "vvvv_diag": vvvv_diag}


def triple_factors(q, X, i, j, k):
    """M[a,b,c], N[a,b,c] and the denominators {variant: D[a,b,c]} of the occupied
    triple (i, j, k), as in crcc23A_opt of crcc_loops.f90 (written in spin orbitals)."""
    t2, l1, l2 = q["t2"], q["l1"], q["l2"]
    vooo, vovv, ooov = q["h_vooo"], q["h_vovv"], q["h_ooov"]
    I_vvov, v_oovv, h_ov = X["I_vvov"], q["v_oovv"], q["h_ov"]

    # ---- M_K = <ijkabc|Hbar|0> ----
    x = (-0.5 * np.einsum("am,bcm->abc", vooo[:, :, i, j], t2[:, :, :, k])
         + 0.5 * np.einsum("am,bcm->abc", vooo[:, :, k, j], t2[:, :, :, i])
         + 0.5 * np.einsum("am,bcm->abc", vooo[:, :, i, k], t2[:, :, :, j])
         + 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k])
         - 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k])
         - 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, k, :], t2[:, :, j, i]))
    M = _antisymmetrize_abc(x)

    # ---- N_K = <0|(L1 + L2) Hbar|ijkabc> ----
    y = (0.5 * np.einsum("eba,ec->abc", vovv[:, i, :, :], l2[:, :, j, k])
         - 0.5 * np.einsum("eba,ec->abc", vovv[:, j, :, :], l2[:, :, i, k])
         - 0.5 * np.einsum("eba,ec->abc", vovv[:, k, :, :], l2[:, :, j, i])
         - 0.5 * np.einsum("ma,bcm->abc", ooov[j, i, :, :], l2[:, :, :, k])
         + 0.5 * np.einsum("ma,bcm->abc", ooov[k, i, :, :], l2[:, :, :, j])
         + 0.5 * np.einsum("ma,bcm->abc", ooov[j, k, :, :], l2[:, :, :, i]))
    # disconnected L1 * V and L2 * h(ov) terms, A(k/ij)A(c/ab)
    z = (np.einsum("c,ab->abc", l1[:, k], v_oovv[i, j]) - np.einsum("c,ab->abc", l1[:, i], v_oovv[k, j])
         - np.einsum("c,ab->abc", l1[:, j], v_oovv[i, k])
         + np.einsum("c,ab->abc", h_ov[k], l2[:, :, i, j]) - np.einsum("c,ab->abc", h_ov[i], l2[:, :, k, j])
         - np.einsum("c,ab->abc", h_ov[j], l2[:, :, i, k]))
    z = z - z.transpose(2, 1, 0) - z.transpose(0, 2, 1)
    N = _antisymmetrize_abc(y) + z

    # ---- denominators ----
    n = np.newaxis
    occ = [i, j, k]
    fv, hv = q["f_vv"], np.diag(q["h_vv"])
    D = {}
    D["A"] = (sum(q["f_oo"][m] for m in occ)
              - fv[:, n, n] - fv[n, :, n] - fv[n, n, :])
    DB = (sum(q["h_oo"][m, m] for m in occ)
          - hv[:, n, n] - hv[n, :, n] - hv[n, n, :])
    D["B"] = DB
    vo = X["voov_diag"][:, occ].sum(axis=1)                 # sum over i,j,k of h(aiia)
    vv = X["vvvv_diag"]
    oo = X["oooo_diag"]
    DC = (DB - vo[:, n, n] - vo[n, :, n] - vo[n, n, :]
          - (oo[i, j] + oo[i, k] + oo[j, k])
          - vv[:, :, n] - vv[:, n, :] - vv[n, :, :])
    D["C"] = DC
    d3o = X["d3_o"][:, i, j] + X["d3_o"][:, i, k] + X["d3_o"][:, j, k]        # [a]
    d3v = X["d3_v"][:, i, :] + X["d3_v"][:, j, :] + X["d3_v"][:, k, :]        # [a,b]
    D["D"] = (DC + d3o[:, n, n] + d3o[n, :, n] + d3o[n, n, :]
              - d3v[:, :, n] - d3v[:, n, :] - d3v[n, :, :])
    return M, N, D


def crcc23_correction(driver, state=0):
    """delta(2,3) for variants A-D from the spin-orbital factors (equal to ccpy's
    driver.deltap3[0] after run_ccp3(method="crcc23"))."""
    q = spin_orbital_quantities(driver, state)
    X = triples_intermediates(q)
    nocc, nvir = q["space"].nocc, q["space"].nvir
    # unique virtual triples a < b < c
    mask = np.zeros((nvir,) * 3, dtype=bool)
    for x in range(nvir):
        for y in range(x + 1, nvir):
            mask[x, y, y + 1:] = True
    delta = dict.fromkeys("ABCD", 0.0)
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for k in range(j + 1, nocc):
                M, N, D = triple_factors(q, X, i, j, k)
                MN = (M * N)[mask]
                for variant in delta:
                    delta[variant] += np.sum(MN / D[variant][mask])
    return delta
