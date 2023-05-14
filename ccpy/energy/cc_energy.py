"""Module containing function to calculate the CC correlation energy."""
import numpy as np
from ccpy.models.operators import ClusterOperator, FockOperator

def get_ci_energy(C, H0):

    e1a = np.einsum("me,em->", H0.a.ov, C.a, optimize=True)
    e1b = np.einsum("me,em->", H0.b.ov, C.b, optimize=True)
    e2a = 0.25 * np.einsum("mnef,efmn->", H0.aa.oovv, C.aa, optimize=True)
    e2b = np.einsum("mnef,efmn->", H0.ab.oovv, C.ab, optimize=True)
    e2c = 0.25 * np.einsum("mnef,efmn->", H0.bb.oovv, C.bb, optimize=True)

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
    e1a = np.einsum("me,em->", H0.a.ov, T.a, optimize=True)
    e1b = np.einsum("me,em->", H0.b.ov, T.b, optimize=True)
    e2aa = 0.25 * np.einsum("mnef,efmn->", H0.aa.oovv, T.aa, optimize=True)
    e2ab = np.einsum("mnef,efmn->", H0.ab.oovv, T.ab, optimize=True)
    e2bb = 0.25 * np.einsum("mnef,efmn->", H0.bb.oovv, T.bb, optimize=True)
    e1a1a = 0.5 * np.einsum("mnef,fn,em->", H0.aa.oovv, T.a, T.a, optimize=True)
    e1b1b = 0.5 * np.einsum("mnef,fn,em->", H0.bb.oovv, T.b, T.b, optimize=True)
    e1a1b = np.einsum("mnef,em,fn->", H0.ab.oovv, T.a, T.b, optimize=True)

    Ecorr = e1a + e1b + e2aa + e2ab + e2bb + e1a1a + e1a1b + e1b1b
    return Ecorr

def get_cc_energy_unsorted(T, H0, occ_a, unocc_a, occ_b, unocc_b):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0> using unsorted integral operator H.
    """
    e1a = np.einsum("me,em->", H0.a[occ_a, unocc_a], T.a, optimize=True)
    e1b = np.einsum("me,em->", H0.b[occ_b, unocc_b], T.b, optimize=True)
    e2aa = 0.25 * np.einsum("mnef,efmn->", H0.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
    e2ab = np.einsum("mnef,efmn->", H0.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    e2bb = 0.25 * np.einsum("mnef,efmn->", H0.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)
    e1a1a = 0.5 * np.einsum("mnef,fn,em->", H0.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, T.a, optimize=True)
    e1b1b = 0.5 * np.einsum("mnef,fn,em->", H0.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, T.b, optimize=True)
    e1a1b = np.einsum("mnef,em,fn->", H0.ab[occ_a, occ_b, unocc_a, unocc_b], T.a, T.b, optimize=True)

    Ecorr = e1a + e1b + e2aa + e2ab + e2bb + e1a1a + e1a1b + e1b1b
    return Ecorr

def get_lcc_energy(L, LH):
    return np.sqrt( np.sum(LH.flatten()**2) ) / np.sqrt( np.sum(L.flatten()**2) )

def get_r0(R, H, omega):

    r0 = 0.0
    r0 += np.einsum("me,em->", H.a.ov, R.a, optimize=True)
    r0 += np.einsum("me,em->", H.b.ov, R.b, optimize=True)
    r0 += 0.25 * np.einsum("mnef,efmn->", H.aa.oovv, R.aa, optimize=True)
    r0 += np.einsum("mnef,efmn->", H.ab.oovv, R.ab, optimize=True)
    r0 += 0.25 * np.einsum("mnef,efmn->", H.bb.oovv, R.bb, optimize=True)

    return r0 / omega

def get_rel(R, r0):
    """
       Compute the reduced excitation level (REL) metric, given by
       \sum_{n=0}^{N} n*<0|(R_{\mu,n})^+ R_{\mu,n})|0>/\sum_{n=0}^{N} <0|(R_{\mu,n})^+ R_{\mu,n})|0>.
       We assume only CCSD-level contributions (e.g., REL for EOMCCSDT, EOMCCSDt, etc. will not include
       effects of R3).
       [See J. Chem. Phys. 122, 214107 (2005)].
    """
    rel_0 = r0**2
    rel_1 = (
            np.einsum("ai,ai->", R.a, R.a, optimize=True)
            + np.einsum("ai,ai->", R.b, R.b, optimize=True)
    )
    rel_2 = (
            0.25 * np.einsum("abij,abij->", R.aa, R.aa, optimize=True)
            + np.einsum("abij,abij->", R.ab, R.ab, optimize=True)
            + 0.25 * np.einsum("abij,abij->", R.bb, R.bb, optimize=True)
    )
    rel = (rel_1 + 2.0 * rel_2)/(rel_0 + rel_1 + rel_2)
    return rel

def get_LR(R, L):

    # explicitly enforce biorthonormality
    if isinstance(L, ClusterOperator):
        LR =  np.einsum("em,em->", R.a, L.a, optimize=True)
        LR += np.einsum("em,em->", R.b, L.b, optimize=True)
        LR += 0.25 * np.einsum("efmn,efmn->", R.aa, L.aa, optimize=True)
        LR += np.einsum("efmn,efmn->", R.ab, L.ab, optimize=True)
        LR += 0.25 * np.einsum("efmn,efmn->", R.bb, L.bb, optimize=True)

        if L.order == 3 and R.order == 3:
            LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.aaa, L.aaa, optimize=True)
            LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.aab, L.aab, optimize=True)
            LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.abb, L.abb, optimize=True)
            LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.bbb, L.bbb, optimize=True)

    if isinstance(L, FockOperator):

        LR = -np.einsum("m,m->", R.a, L.a, optimize=True)
        LR -= np.einsum("m,m->", R.b, L.b, optimize=True)
        LR -= 0.5 * np.einsum("fnm,fnm->", R.aa, L.aa, optimize=True)
        LR -= np.einsum("fnm,fnm->", R.ab, L.ab, optimize=True)
        LR -= np.einsum("fnm,fnm->", R.ba, L.ba, optimize=True)
        LR -= 0.5 * np.einsum("fnm,fnm->", R.bb, L.bb, optimize=True)

    return LR
