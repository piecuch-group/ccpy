"""Module containing function to calculate the CC correlation energy."""
import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.models.operators import ClusterOperator, FockOperator
from ccpy.lib.core import leftccsdt_p_loops

def get_ci_energy(C, H0):

    e1a = ccpy_einsum("me,em->", H0.a.ov, C.a)
    e1b = ccpy_einsum("me,em->", H0.b.ov, C.b)
    e2a = 0.25 * ccpy_einsum("mnef,efmn->", H0.aa.oovv, C.aa)
    e2b = ccpy_einsum("mnef,efmn->", H0.ab.oovv, C.ab)
    e2c = 0.25 * ccpy_einsum("mnef,efmn->", H0.bb.oovv, C.bb)

    Ecorr = e1a + e1b + e2a + e2b + e2c

    return Ecorr

def get_cc_energy(T, H0):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0>.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced integrals F_N and V_N that define the bare Hamiltonian H_N

    Returns
    -------
    Ecorr : float
        CC correlation energy
    """
    e1a = ccpy_einsum("me,em->", H0.a.ov, T.a)
    e1b = ccpy_einsum("me,em->", H0.b.ov, T.b)
    e2aa = 0.25 * ccpy_einsum("mnef,efmn->", H0.aa.oovv, T.aa)
    e2ab = ccpy_einsum("mnef,efmn->", H0.ab.oovv, T.ab)
    e2bb = 0.25 * ccpy_einsum("mnef,efmn->", H0.bb.oovv, T.bb)
    e1a1a = 0.5 * ccpy_einsum("mnef,fn,em->", H0.aa.oovv, T.a, T.a)
    e1b1b = 0.5 * ccpy_einsum("mnef,fn,em->", H0.bb.oovv, T.b, T.b)
    e1a1b = ccpy_einsum("mnef,em,fn->", H0.ab.oovv, T.a, T.b)

    Ecorr = e1a + e1b + e2aa + e2ab + e2bb + e1a1a + e1a1b + e1b1b
    return Ecorr

def get_cc_energy_unsorted(T, H0, occ_a, unocc_a, occ_b, unocc_b):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0> using unsorted integral operator H."""
    e1a = ccpy_einsum("me,em->", H0.a[occ_a, unocc_a], T.a)
    e1b = ccpy_einsum("me,em->", H0.b[occ_b, unocc_b], T.b)
    e2aa = 0.25 * ccpy_einsum("mnef,efmn->", H0.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa)
    e2ab = ccpy_einsum("mnef,efmn->", H0.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab)
    e2bb = 0.25 * ccpy_einsum("mnef,efmn->", H0.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb)
    e1a1a = 0.5 * ccpy_einsum("mnef,fn,em->", H0.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, T.a)
    e1b1b = 0.5 * ccpy_einsum("mnef,fn,em->", H0.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, T.b)
    e1a1b = ccpy_einsum("mnef,em,fn->", H0.ab[occ_a, occ_b, unocc_a, unocc_b], T.a, T.b)

    Ecorr = e1a + e1b + e2aa + e2ab + e2bb + e1a1a + e1a1b + e1b1b
    return Ecorr

def get_lcc_energy(L, LH):
    return np.sqrt( np.sum(LH.flatten()**2) ) / np.sqrt( np.sum(L.flatten()**2) )

def get_r0(R, H, omega):

    r0 = 0.0
    r0 += ccpy_einsum("me,em->", H.a.ov, R.a)
    r0 += ccpy_einsum("me,em->", H.b.ov, R.b)
    r0 += 0.25 * ccpy_einsum("mnef,efmn->", H.aa.oovv, R.aa)
    r0 += ccpy_einsum("mnef,efmn->", H.ab.oovv, R.ab)
    r0 += 0.25 * ccpy_einsum("mnef,efmn->", H.bb.oovv, R.bb)

    return r0 / omega

def get_rel(R, r0):
    r"""Compute the reduced excitation level (REL) metric, given by
    \sum_{n=0}^{N} n*<0|(R_{\mu,n})^+ R_{\mu,n})|0>/\sum_{n=0}^{N} <0|(R_{\mu,n})^+ R_{\mu,n})|0>.
    [See J. Chem. Phys. 122, 214107 (2005)]."""
    rel_0 = r0**2
    rel_1 = (
            ccpy_einsum("ai,ai->", R.a, R.a)
            + ccpy_einsum("ai,ai->", R.b, R.b)
    )
    rel_2 = (
            0.25 * ccpy_einsum("abij,abij->", R.aa, R.aa)
            + ccpy_einsum("abij,abij->", R.ab, R.ab)
            + 0.25 * ccpy_einsum("abij,abij->", R.bb, R.bb)
    )
    rel = (rel_1 + 2.0 * rel_2)/(rel_0 + rel_1 + rel_2)
    return rel

def get_rel_ea(R):
    """Compute the reduced excitation level (REL) metric for electron-attached states."""
    rel_1 = (
            ccpy_einsum("a,a->", R.a, R.a)
    )
    rel_2 = (
            0.5 * ccpy_einsum("abj,abj->", R.aa, R.aa)
            + ccpy_einsum("abj,abj->", R.ab, R.ab)
    )
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def get_rel_ip(R):
    """Compute the reduced excitation level (REL) metric for electron-ionized states."""
    rel_1 = (
            ccpy_einsum("i,i->", R.a, R.a)
    )
    rel_2 = (
            0.5 * ccpy_einsum("ibj,ibj->", R.aa, R.aa)
            + ccpy_einsum("ibj,ibj->", R.ab, R.ab)
    )
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def get_LR(R, L, l3_excitations=None, r3_excitations=None):

    LR = 0.0
    nua, noa = L.a.shape
    nub, nob = L.b.shape
    # explicitly enforce biorthonormality
    LR =  ccpy_einsum("em,em->", R.a, L.a)
    LR += ccpy_einsum("em,em->", R.b, L.b)
    LR += 0.25 * ccpy_einsum("efmn,efmn->", R.aa, L.aa)
    LR += ccpy_einsum("efmn,efmn->", R.ab, L.ab)
    LR += 0.25 * ccpy_einsum("efmn,efmn->", R.bb, L.bb)

    if L.order == 3 and R.order == 3:
        if l3_excitations is None and r3_excitations is None:
            LR += (1.0 / 36.0) * ccpy_einsum("efgmno,efgmno->", R.aaa, L.aaa)
            LR += (1.0 / 4.0) * ccpy_einsum("efgmno,efgmno->", R.aab, L.aab)
            LR += (1.0 / 4.0) * ccpy_einsum("efgmno,efgmno->", R.abb, L.abb)
            LR += (1.0 / 36.0) * ccpy_einsum("efgmno,efgmno->", R.bbb, L.bbb)
        else:
            # LR_P = leftccsdt_p_loops.lr(L.aaa, l3_excitations["aaa"],
            #                                               L.aab, l3_excitations["aab"],
            #                                               L.abb, l3_excitations["abb"],
            #                                               L.bbb, l3_excitations["bbb"],
            #                                               R.aaa, r3_excitations["aaa"],
            #                                               R.aab, r3_excitations["aab"],
            #                                               R.abb, r3_excitations["abb"],
            #                                               R.bbb, r3_excitations["bbb"],
            #                                               noa, nua, nob, nub)
            # This is allowed because L and R are always aligned through the left-EOMCC(P) iterations
            LR_aaa = np.dot(L.aaa.T, R.aaa)
            LR_aab = np.dot(L.aab.T, R.aab)
            LR_abb = np.dot(L.abb.T, R.abb)
            LR_bbb = np.dot(L.bbb.T, R.bbb)
            LR_P = LR_aaa + LR_aab + LR_abb + LR_bbb
            LR += LR_P
    return LR

def get_LR_ipeom(R, L, l3_excitations=None, r3_excitations=None):
    # these should be + signs! Ghost loop rule.
    LR = ccpy_einsum("m,m->", R.a, L.a)
    LR += 0.5 * ccpy_einsum("mfn,mfn->", R.aa, L.aa)
    LR += ccpy_einsum("mfn,mfn->", R.ab, L.ab)
    if L.order == 3 and R.order == 3:
        if l3_excitations is None and r3_excitations is None:
            LR += (1.0 / 12.0) * ccpy_einsum("ifgno,ifgno->", R.aaa, L.aaa)
            LR += (1.0 / 2.0) * ccpy_einsum("ifgno,ifgno->", R.aab, L.aab)
            LR += (1.0 / 4.0) * ccpy_einsum("ifgno,ifgno->", R.abb, L.abb)
        else:
            # This is allowed because L and R are always aligned through the left-IPEOMCC(P) iterations
            LR_aaa = np.dot(L.aaa.T, R.aaa)
            LR_aab = np.dot(L.aab.T, R.aab)
            LR_abb = np.dot(L.abb.T, R.abb)
            LR_P = LR_aaa + LR_aab + LR_abb
            LR += LR_P
    return LR

def get_LR_eaeom(R, L, l3_excitations=None, r3_excitations=None):
    LR = ccpy_einsum("e,e->", R.a, L.a)
    LR += 0.5 * ccpy_einsum("efn,efn->", R.aa, L.aa)
    LR += ccpy_einsum("efn,efn->", R.ab, L.ab)
    if L.order == 3 and R.order == 3:
        if l3_excitations is None and r3_excitations is None:
            LR += (1.0 / 12.0) * ccpy_einsum("efgno,efgno->", R.aaa, L.aaa)
            LR += (1.0 / 2.0) * ccpy_einsum("efgno,efgno->", R.aab, L.aab)
            LR += (1.0 / 4.0) * ccpy_einsum("efgno,efgno->", R.abb, L.abb)
        else:
            # This is allowed because L and R are always aligned through the left-EAEOMCC(P) iterations
            LR_aaa = np.dot(L.aaa.T, R.aaa)
            LR_aab = np.dot(L.aab.T, R.aab)
            LR_abb = np.dot(L.abb.T, R.abb)
            LR_P = LR_aaa + LR_aab + LR_abb
            LR += LR_P
    return LR
