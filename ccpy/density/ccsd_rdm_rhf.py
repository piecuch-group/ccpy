"""Spin-traced (closed-shell) CCSD response densities from the ground-state T and Lambda
amplitudes.

The intermediates follow pyscf/cc/ccsd_rdm.py (Apache License 2.0, Copyright 2014-2020
The PySCF Developers), rewritten in-core for ccpy's operator containers.

Conventions (identical to PySCF):
    rdm1[p, q] = <q_a^+ p_a> + <q_b^+ p_b>
    rdm2[p, q, r, s] = sum_{sigma, tau} <p_sigma^+ r_tau^+ s_tau q_sigma>,
so that, with h and (pq|rs) in the MO basis,
    E = E_nuc + einsum('pq,qp', h, rdm1) + 0.5 * einsum('pqrs,pqrs', eri, rdm2).
These are the unrelaxed densities of the CCSD Lagrangian <0|(1 + Lambda) e^(-T) H e^T|0>,
so the energy above is the CCSD energy. Both include the frozen core.

No quantity with four virtual indices is ever formed. The two-particle density is returned
as a ``Density2`` holding the 15 occupied/virtual blocks with at most three virtual indices;
its vvvv block is generated, three virtual indices at a time, by ``CCSDVvvvSlices``.
"""

import itertools

import numpy as np

# all occupied/virtual block labels except vvvv
BLOCKS = ["".join(k) for k in itertools.product("ov", repeat=4) if "".join(k) != "vvvv"]
# the 8 index permutations that leave (pq|rs) unchanged for real orbitals
_SYMMETRY_PERMUTATIONS = [
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
]


def _einsum(*args):
    return np.einsum(*args, optimize=True)


def _pyscf_amplitudes(T, L):
    """ccpy RHF amplitudes -> PySCF layout t1[i,a], t2[i,j,a,b] (alpha-beta), l1, l2."""
    t1 = np.ascontiguousarray(T.a.T)
    t2 = np.ascontiguousarray(T.ab.transpose(2, 3, 0, 1))
    l1 = np.ascontiguousarray(L.a.T)
    l2 = np.ascontiguousarray(L.ab.transpose(2, 3, 0, 1))
    return t1, t2, l1, l2


def _gamma1_intermediates(t1, t2, l1, l2):
    doo = -np.einsum("ja,ia->ij", t1, l1)
    dvv = np.einsum("ia,ib->ab", t1, l1)
    xtv = np.einsum("ie,me->im", t1, l1)
    dvo = t1.T - np.einsum("im,ma->ai", xtv, t1)
    theta = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    doo -= _einsum("jkab,ikab->ij", theta, l2)
    dvv += _einsum("jica,jicb->ab", theta, l2)
    xt1 = _einsum("mnef,inef->mi", l2, theta)
    xt2 = _einsum("mnaf,mnef->ea", l2, theta)
    dvo += _einsum("imae,me->ai", theta, l1)
    dvo -= np.einsum("mi,ma->ai", xt1, t1)
    dvo -= np.einsum("ie,ae->ai", t1, xt2)
    dov = l1
    return doo, dov, dvo, dvv


def _gamma2_intermediates(t1, t2, l1, l2):
    """The PySCF intermediates except dvvvv. gvvvv = l2 . tau is never formed; its only
    other use, gvovv, is contracted as (l2 . t1) . tau."""
    pvOOv = _einsum("ikca,jkcb->aijb", l2, t2)
    moo = np.einsum("dljd->jl", pvOOv) * 2
    mvv = np.einsum("blld->db", pvOOv) * 2
    gooov = _einsum("kc,cija->jkia", t1, pvOOv)

    theta = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    pvoOV = -_einsum("ikca,jkbc->aijb", l2, t2)
    pvoOV += _einsum("ikac,jkbc->aijb", l2, theta)
    moo += np.einsum("dljd->jl", pvoOV)
    mvv += np.einsum("blld->db", pvoOV)
    gooov -= _einsum("jc,cika->jkia", t1, pvoOV)

    mia = 2.0 * _einsum("kc,ikac->ia", l1, t2) - _einsum("kc,ikca->ia", l1, t2)
    mab = _einsum("kc,kb->cb", l1, t1)
    mij = _einsum("kc,jc->jk", l1, t1) + 0.5 * moo

    tau = t2 + _einsum("ia,jb->ijab", t1, t1)
    goooo = 0.5 * _einsum("ijab,klab->ijkl", tau, l2)
    doooo = 2.0 * goooo.transpose(0, 2, 1, 3) - goooo.transpose(0, 3, 1, 2)

    gooov += _einsum("ji,ka->jkia", -0.5 * moo, t1)
    gooov += _einsum("la,jkil->jkia", 2.0 * t1, goooo)
    gooov -= _einsum("ib,jkba->jkia", l1, tau)
    gooov -= _einsum("jkba,ib->jkia", l2, t1)
    dooov = 2.0 * gooov.transpose(0, 2, 1, 3) - gooov.transpose(1, 2, 0, 3)

    goovv = _einsum("ia,jb->ijab", mia, t1)
    tmpoovv = _einsum("ijkl,klab->ijab", goooo, tau)
    tmpoovv -= _einsum("jk,ikab->ijab", mij, tau)
    tmpoovv -= _einsum("cb,ijac->ijab", mab, t2)
    tmpoovv -= _einsum("bd,ijad->ijab", 0.5 * mvv, tau)
    tmpoovv += 0.5 * tau
    tmpoovv += 0.5 * l2
    goovv += tmpoovv

    gOvvO = _einsum("kiac,jc,kb->iabj", l2, t1, t1)
    gOvvO += pvOOv.transpose(1, 0, 3, 2)
    govVO = _einsum("ia,jb->iabj", l1, t1)
    govVO -= _einsum("ikac,jc,kb->iabj", l2, t1, t1)
    govVO += pvoOV.transpose(1, 0, 3, 2)
    dovvo = 2.0 * govVO + gOvvO
    doovv = (-2.0 * gOvvO - govVO).transpose(3, 0, 1, 2)

    tau_half = tau - 0.5 * t2
    goovv += _einsum("dlib,jlda->ijab", pvOOv, tau_half)
    goovv -= _einsum("dlia,jldb->ijab", pvoOV, tau_half)
    goovv += _einsum("dlia,jlbd->ijab", pvoOV + 0.5 * pvOOv, t2)
    dovov = 2.0 * goovv.transpose(0, 2, 1, 3) - goovv.transpose(1, 2, 0, 3)

    # -t1[i,d] gvvvv[a,d,b,c] with gvvvv[a,d,b,c] = sum_jk l2[j,k,a,d] tau[j,k,b,c]
    l2t1 = _einsum("jkad,id->jkai", l2, t1)
    gvovv = -_einsum("jkai,jkbc->aibc", l2t1, tau)
    gvovv += _einsum("akic,kb->aibc", pvoOV, t1)
    gvovv -= _einsum("akib,kc->aibc", pvOOv, t1)
    gvovv += _einsum("ja,jibc->aibc", l1, t2)
    gvovv += _einsum("ja,jb,ic->aibc", l1, t1, t1)
    gvovv += _einsum("ba,ic->aibc", 0.5 * mvv, t1)
    gvovv += _einsum("ja,jibc->aibc", t1, l2)
    dovvv = 2.0 * gvovv.transpose(1, 3, 0, 2) - gvovv.transpose(1, 2, 0, 3)

    return dovov, doooo, doovv, dovvo, dovvv, dooov, tau


class Density2:
    """Spin-traced two-particle density in chemists' order (pairs with (pq|rs) through
    E = 1/2 sum (pq|rs) D[p,q,r,s]), stored as occupied/virtual blocks.

    ``blocks[key]`` holds the block ``key`` (e.g. "ovvv") for the 15 keys other than "vvvv".
    The vvvv block is never stored: ``vvvv`` is a list of callables, each returning one
    contribution's 8-fold-symmetrized slice D[a, :, :, :] (three virtual indices, virtual
    indices counted from the first virtual). Densities from different sources are combined
    with ``+``, which adds the blocks and concatenates the vvvv generators.

    Occupied indices run over the frozen core and the correlated occupied orbitals.
    """

    def __init__(self, blocks, nocc, nvir, vvvv=()):
        self.nocc = nocc
        self.nvir = nvir
        dims = {"o": nocc, "v": nvir}
        self.blocks = {
            key: (np.asarray(blocks[key]) if key in blocks else np.zeros(tuple(dims[c] for c in key)))
            for key in BLOCKS
        }
        self.vvvv = list(vvvv)

    def __add__(self, other):
        if (self.nocc, self.nvir) != (other.nocc, other.nvir):
            raise ValueError("Densities over different orbital spaces cannot be added.")
        blocks = {k: self.blocks[k] + other.blocks[k] for k in BLOCKS}
        return Density2(blocks, self.nocc, self.nvir, self.vvvv + other.vvvv)

    def scaled(self, factor):
        blocks = {k: factor * v for k, v in self.blocks.items()}
        return Density2(blocks, self.nocc, self.nvir, [lambda a, g=g: factor * g(a) for g in self.vvvv])

    def add_blocks(self, extra):
        """A new density with the (non-vvvv) blocks in the dict ``extra`` added."""
        blocks = {k: self.blocks[k] + extra.get(k, 0.0) for k in BLOCKS}
        return Density2(blocks, self.nocc, self.nvir, self.vvvv)

    def symmetrized(self):
        """The 8-fold (pq|rs) symmetrization of the blocks, which is all that real integrals
        see (the vvvv generators already return symmetrized slices)."""
        sym = {}
        for key in BLOCKS:
            acc = 0.0
            for perm in _SYMMETRY_PERMUTATIONS:
                src = [None] * 4
                for k in range(4):
                    src[perm[k]] = key[k]
                acc = acc + self.blocks["".join(src)].transpose(perm)
            sym[key] = acc / 8.0
        return Density2(sym, self.nocc, self.nvir, self.vvvv)

    def vvvv_slice(self, a):
        """The symmetrized D[a, :, :, :] of the vvvv block, summed over all contributions."""
        out = np.zeros((self.nvir,) * 3)
        for gen in self.vvvv:
            out += gen(a)
        return out

    def contract_vvvv(self, R_vv):
        """sum_cd D[a,b,c,d] R[x,c,d] for the vvvv block, slice by slice."""
        out = np.zeros(R_vv.shape[:1] + (self.nvir, self.nvir))
        if not self.vvvv:
            return out
        for a in range(self.nvir):
            out[:, a, :] = _einsum("bcd,xcd->xb", self.vvvv_slice(a), R_vv)
        return out

    def contract(self, R):
        """Y[x,p,q] = sum_rs D[p,q,r,s] R[x,r,s] over the full orbital space."""
        no = self.nocc
        sl = {"o": slice(0, no), "v": slice(no, no + self.nvir)}
        Y = np.zeros(R.shape)
        for key, blk in self.blocks.items():
            Y[:, sl[key[0]], sl[key[1]]] += _einsum("pqrs,xrs->xpq", blk, R[:, sl[key[2]], sl[key[3]]])
        Y[:, sl["v"], sl["v"]] += self.contract_vvvv(R[:, sl["v"], sl["v"]])
        return Y

    def to_ao(self, C):
        """Back-transform to the AO basis (n_ao^4; for the four-center gradient path)."""
        no = self.nocc
        Cb = {"o": C[:, :no], "v": C[:, no:]}
        Cv = Cb["v"]
        nao = C.shape[0]
        G = np.zeros((nao,) * 4)
        for key, blk in self.blocks.items():
            G += _einsum("mp,nq,pqrs,lr,ks->mnlk", Cb[key[0]], Cb[key[1]], blk, Cb[key[2]], Cb[key[3]])
        if self.vvvv:
            for a in range(self.nvir):
                G += _einsum("m,nq,qrs,lr,ks->mnlk", Cv[:, a], Cv, self.vvvv_slice(a), Cv, Cv)
        return G


class CCSDVvvvSlices:
    """Symmetrized vvvv slices of the CCSD density,
    D[a,b,c,d] = 2 g[b,d,a,c] - g[b,d,c,a] + 2 g[a,c,b,d] - g[a,c,d,b],
    g[w,x,y,z] = sum_ij l2[i,j,w,x] tau[i,j,y,z], built three virtual indices at a time."""

    _G_TERMS = ((2.0, (1, 3, 0, 2)), (-1.0, (1, 3, 2, 0)), (2.0, (0, 2, 1, 3)), (-1.0, (0, 2, 3, 1)))

    def __init__(self, l2, tau):
        self.l2 = l2
        self.tau = tau

    def _g_fixed(self, pos, a):
        """g with the index at position ``pos`` fixed to ``a``."""
        l2, tau = self.l2, self.tau
        if pos == 0:
            return _einsum("ijx,ijyz->xyz", l2[:, :, a, :], tau)
        if pos == 1:
            return _einsum("ijw,ijyz->wyz", l2[:, :, :, a], tau)
        if pos == 2:
            return _einsum("ijwx,ijz->wxz", l2, tau[:, :, a, :])
        return _einsum("ijwx,ijy->wxy", l2, tau[:, :, :, a])

    def raw_fixed(self, q, a):
        """Unsymmetrized D with position ``q`` fixed to ``a``; axes are the other positions."""
        out = 0.0
        for coeff, g_to_d in self._G_TERMS:
            p = g_to_d.index(q)
            sl = self._g_fixed(p, a)
            remaining_d = [g_to_d[k] for k in range(4) if k != p]
            out = out + coeff * sl.transpose(np.argsort(remaining_d))
        return out

    def __call__(self, a):
        out = 0.0
        for perm in _SYMMETRY_PERMUTATIONS:
            sl = self.raw_fixed(perm[0], a)
            remaining = [p for p in range(4) if p != perm[0]]
            out = out + sl.transpose([remaining.index(perm[k]) for k in (1, 2, 3)])
        return out / 8.0


def calc_ccsd_rdms_rhf(T, L, system):
    """Spin-traced CCSD response densities of a closed-shell (RHF) reference.

    Parameters
    ----------
    T, L : ClusterOperator
        Converged CCSD and left-CCSD (Lambda) amplitudes (ground state).
    system : System
        Must be closed shell; ``nfrozen`` core orbitals are added back with occupation 2.

    Returns
    -------
    rdm1 : ndarray, shape (nmo, nmo)
    rdm2 : Density2
        Two-particle density in blocks (see the module docstring); occupied indices run
        over the frozen core and the correlated occupied orbitals.
    """
    if system.noccupied_alpha != system.noccupied_beta:
        raise NotImplementedError("calc_ccsd_rdms_rhf requires a closed-shell reference.")
    t1, t2, l1, l2 = _pyscf_amplitudes(T, L)
    nocc, nvir = t1.shape
    nfz = system.nfrozen
    nO = nfz + nocc  # all occupied orbitals
    nmo = nO + nvir

    # ---- one-particle density, normal-ordered part, full space ----
    doo, dov, dvo, dvv = _gamma1_intermediates(t1, t2, l1, l2)
    act_o = slice(nfz, nO)
    v_full = slice(nO, nmo)
    d1 = np.zeros((nmo, nmo))
    d1[act_o, act_o] = doo + doo.T
    d1[act_o, v_full] = dov + dvo.T
    d1[v_full, act_o] = d1[act_o, v_full].T
    d1[v_full, v_full] = dvv + dvv.T

    # ---- two-particle density, d2[p,q,r,s] = < p^+ r^+ s q > (normal-ordered), active blocks ----
    dovov, doooo, doovv, dovvo, dovvv, dooov, tau = _gamma2_intermediates(t1, t2, l1, l2)
    act = {}
    act["ovov"] = dovov + dovov.transpose(2, 3, 0, 1)
    act["vovo"] = act["ovov"].transpose(1, 0, 3, 2)
    act["oovv"] = doovv + doovv.transpose(1, 0, 3, 2)
    act["vvoo"] = act["oovv"].transpose(2, 3, 0, 1)
    act["ovvo"] = dovvo + dovvo.transpose(3, 2, 1, 0)
    act["voov"] = act["ovvo"].transpose(1, 0, 3, 2)
    act["oooo"] = 2.0 * (doooo + doooo.transpose(1, 0, 3, 2))
    act["ovvv"] = dovvv
    act["vvov"] = dovvv.transpose(2, 3, 0, 1)
    act["vvvo"] = dovvv.transpose(3, 2, 1, 0)
    act["vovv"] = dovvv.transpose(1, 0, 3, 2)
    act["ooov"] = dooov
    act["ovoo"] = dooov.transpose(2, 3, 0, 1)
    act["oovo"] = dooov.transpose(1, 0, 3, 2)
    act["vooo"] = dooov.transpose(3, 2, 1, 0)

    # embed the correlated occupied indices into the full occupied space (frozen core first)
    dims = {"o": nO, "v": nvir}
    actsl = {"o": act_o, "v": slice(0, nvir)}
    d2 = {}
    for key in BLOCKS:
        blk = np.zeros(tuple(dims[c] for c in key))
        blk[tuple(actsl[c] for c in key)] = act[key]
        d2[key] = blk

    # ---- reference x correlation and reference x reference parts ----
    sl = {"o": slice(0, nO), "v": v_full}
    I = np.eye(nO)
    for key in BLOCKS:
        X, Y = key[2], key[3]
        if key[:2] == "oo":   # d2[i,i,:,:] += 2 d1
            d2[key] += 2.0 * np.einsum("ij,xy->ijxy", I, d1[sl[X], sl[Y]])
        if key[2:] == "oo":   # d2[:,:,i,i] += 2 d1
            d2[key] += 2.0 * np.einsum("xy,ij->xyij", d1[sl[key[0]], sl[key[1]]], I)
        if key[1:3] == "oo":  # d2[:,i,i,:] -= d1
            d2[key] -= np.einsum("xy,ij->xijy", d1[sl[key[0]], sl[key[3]]], I)
        if key[0] == "o" and key[3] == "o":  # d2[i,:,:,i] -= d1.T
            d2[key] -= np.einsum("yx,ij->ixyj", d1[sl[key[2]], sl[key[1]]], I)
    d2["oooo"] += 4.0 * np.einsum("ij,kl->ijkl", I, I)
    d2["oooo"] -= 2.0 * np.einsum("il,jk->ijkl", I, I)

    rdm1 = d1.copy()
    rdm1[np.arange(nO), np.arange(nO)] += 2.0

    # chemists' order: D[p,q,r,s] = d2[q,p,s,r]
    chem = {}
    for key in BLOCKS:
        src = key[1] + key[0] + key[3] + key[2]
        chem[key] = d2[src].transpose(1, 0, 3, 2)
    return rdm1, Density2(chem, nO, nvir, [CCSDVvvvSlices(l2, tau)])
