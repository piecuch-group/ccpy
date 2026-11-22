'''
Coupled-Cluster Method with Singles, Doubles, and Arbitrary Subsets of Triples and Quadruples [CCSDTQ(P)]
[This version uses Cholesky decomposition of the two-electron integrals]

Every Hbar(vvvv) term of the T3 and T4 equations (the ladders, the h1(vv) terms that the 4-index
loops merge into them, and the T4 (vvvv T2 T2) terms) comes from one pass over W_ab(e,f) slices
per pair spin (ccpy.cc.chol_ppl.vvvv_t34), passed to the *_chol loops as their starting residual.
'''

import numpy as np
# Modules for type checking
from typing import Tuple, Dict
from ccpy.models.operators import ClusterOperator
from ccpy.models.integrals import Integral
#
from ccpy.hbar.hbar_ccsd_chol import get_ccsd_intermediates
from ccpy.lib.core import ccsdtq_p_loops
from ccpy.cc.ccsdt_p_chol import update_t1_and_t2_residual
from ccpy.cc.chol_ppl import vvvv_t34


def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           X: Integral,
           shift: float,
           flag_RHF: bool,
           t3_excitations: Dict[str, np.ndarray],
           t4_excitations: Dict[str, np.ndarray]) -> Tuple[ClusterOperator, ClusterOperator]:
    """One CCSDTQ(P) update with Cholesky integrals; see ccpy.cc.ccsdtq_p.update for the arguments."""
    # [1., 1., 1., 1., 1., 1.] (and its 8-column analog) is the "empty" state in the Fortran modules.
    do_t3 = {key: not np.array_equal(t3_excitations[key][0, :], np.ones(6)) for key in ("aaa", "aab", "abb", "bbb")}
    do_t4 = {key: not np.array_equal(t4_excitations[key][0, :], np.ones(8))
             for key in ("aaaa", "aaab", "aabb", "abbb", "bbbb")}

    # Last iteration's o*v^3 Hbar blocks are rebuilt below and not read before that.
    for block, names in ((X.aa, ("vovv", "vvov")), (X.ab, ("vovv", "ovvv", "vvov", "vvvo")),
                         (X.bb, ("vovv", "vvov"))):
        for name in names:
            setattr(block, name, None)

    # update T1 and T2
    T, dT = update_t1_and_t2_residual(T, dT, H, X, shift, flag_RHF, t3_excitations)
    T, dT = update_t2a(T, dT, X, H, shift, t3_excitations, t4_excitations)
    T, dT = update_t2b(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift, t3_excitations, t4_excitations)

    # CCSD intermediates
    X = get_ccsd_intermediates(T, X, H, flag_RHF)
    # Hbar(vvvv) and h1(vv) parts of the T3 and T4 residuals, in the current P-list order
    R = vvvv_t34(T, X, H, t3_excitations, t4_excitations, flag_RHF)

    # Transpose integrals appropriately
    X.a.vv = X.a.vv.T
    X.b.vv = X.b.vv.T
    #
    X.aa.vvov = X.aa.vvov.transpose(3, 0, 1, 2)
    X.ab.vvov = X.ab.vvov.transpose(3, 0, 1, 2)
    X.ab.vvvo = X.ab.vvvo.transpose(2, 0, 1, 3)
    X.bb.vvov = X.bb.vvov.transpose(3, 0, 1, 2)
    #
    X.aa.voov = X.aa.voov.transpose(1, 3, 0, 2)
    X.ab.voov = X.ab.voov.transpose(1, 3, 0, 2)
    X.ab.ovvo = X.ab.ovvo.transpose(0, 2, 1, 3)
    X.ab.vovo = X.ab.vovo.transpose(1, 2, 0, 3)
    X.ab.ovov = X.ab.ovov.transpose(0, 3, 1, 2)
    X.bb.voov = X.bb.voov.transpose(1, 3, 0, 2)
    X = _fortran_order(X, flag_RHF, False)

    # update T3
    if do_t3["aaa"]:
        T, dT, t3_excitations = update_t3a(T, dT, X, H, shift, t3_excitations, t4_excitations, R["aaa"])
    if do_t3["aab"]:
        T, dT, t3_excitations = update_t3b(T, dT, X, H, shift, t3_excitations, t4_excitations, R["aab"])
    if flag_RHF:
        T.abb = T.aab.copy()
        dT.abb = dT.aab.copy()
        t3_excitations["abb"] = t3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
        t3_excitations["bbb"] = t3_excitations["aaa"].copy(order="F")
    else:
        if do_t3["abb"]:
            T, dT, t3_excitations = update_t3c(T, dT, X, H, shift, t3_excitations, t4_excitations, R["abb"])
        if do_t3["bbb"]:
            T, dT, t3_excitations = update_t3d(T, dT, X, H, shift, t3_excitations, t4_excitations, R["bbb"])

    # Transpose integrals back
    X.a.vv = X.a.vv.T
    X.b.vv = X.b.vv.T
    #
    X.aa.vvov = X.aa.vvov.transpose(1, 2, 3, 0)
    X.ab.vvov = X.ab.vvov.transpose(1, 2, 3, 0)
    X.ab.vvvo = X.ab.vvvo.transpose(1, 2, 0, 3)
    X.bb.vvov = X.bb.vvov.transpose(1, 2, 3, 0)
    #
    X.aa.voov = X.aa.voov.transpose(2, 0, 3, 1)
    X.ab.voov = X.ab.voov.transpose(2, 0, 3, 1)
    X.ab.ovvo = X.ab.ovvo.transpose(0, 2, 1, 3)
    X.ab.vovo = X.ab.vovo.transpose(2, 0, 1, 3)
    X.ab.ovov = X.ab.ovov.transpose(0, 2, 3, 1)
    X.bb.voov = X.bb.voov.transpose(2, 0, 3, 1)
    X = _fortran_order(X, flag_RHF, True)

    # add VT3 intermediates to two-body part of HBar
    X = add_VT3_intermediates(T, X, H, t3_excitations)

    # update T4
    if do_t4["aaaa"]:
        T, dT, t4_excitations = update_t4a(T, dT, X, H, shift, t3_excitations, t4_excitations, R["aaaa"])
    if do_t4["aaab"]:
        T, dT, t4_excitations = update_t4b(T, dT, X, H, shift, t3_excitations, t4_excitations, R["aaab"])
    if do_t4["aabb"]:
        T, dT, t4_excitations = update_t4c(T, dT, X, H, shift, t3_excitations, t4_excitations, R["aabb"])
    if flag_RHF:
        T.abbb = T.aaab.copy()
        dT.abbb = dT.aaab.copy()
        t4_excitations["abbb"] = t4_excitations["aaab"][:, np.array([3, 0, 1, 2, 7, 4, 5, 6])]
        T.bbbb = T.aaaa.copy()
        dT.bbbb = dT.aaaa.copy()
        t4_excitations["bbbb"] = t4_excitations["aaaa"].copy(order="F")
    else:
        if do_t4["abbb"]:
            T, dT, t4_excitations = update_t4d(T, dT, X, H, shift, t3_excitations, t4_excitations, R["abbb"])
        if do_t4["bbbb"]:
            T, dT, t4_excitations = update_t4e(T, dT, X, H, shift, t3_excitations, t4_excitations, R["bbbb"])

    return T, dT


def _fortran_order(X, flag_RHF, bb_vvov_owned):
    """Store the 4-index Hbar blocks the native T3/T4 routines read in Fortran order, one at a
    time, so f2py does not copy every o*v^3 argument at every call (T3 and T4 use different
    layouts, hence once per phase). add_vt3_p updates bb.vvov in place, so under RHF it needs
    its own buffer (bb_vvov_owned)."""
    for block, names in ((X.aa, ("vvov", "voov", "oooo")),
                         (X.ab, ("vvov", "vvvo", "voov", "ovvo", "vovo", "ovov", "oooo"))):
        for name in names:
            setattr(block, name, np.asfortranarray(getattr(block, name)))
    if flag_RHF:
        X.bb.voov = X.aa.voov
        X.bb.oooo = X.aa.oooo
        X.bb.vvov = X.aa.vvov.copy(order="F") if bb_vvov_owned else X.aa.vvov
    else:
        for name in ("vvov", "voov", "oooo"):
            setattr(X.bb, name, np.asfortranarray(getattr(X.bb, name)))
    return X


def update_t2a(T, dT, X, H, shift, t3_excitations, t4_excitations):
    """Finish t2a from its raw residual: T3 and T4 terms, antisymmetry and the update."""
    T.aa, dT.aa = ccsdtq_p_loops.update_t2a(
        T.aa,
        dT.aa,
        t3_excitations["aaa"], t3_excitations["aab"],
        T.aaa, T.aab,
        X.a.ov, X.b.ov,
        X.aa.ooov, X.aa.vovv,
        X.ab.ooov, X.ab.vovv,
        H.a.oo, H.a.vv,
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        shift
    )
    return T, dT


def update_t2b(T, dT, X, H, shift, t3_excitations, t4_excitations):
    """Finish t2b from its raw residual: T3 and T4 terms and the update."""
    T.ab, dT.ab = ccsdtq_p_loops.update_t2b(
        T.ab,
        dT.ab,
        t3_excitations["aab"], t3_excitations["abb"],
        T.aab, T.abb,
        X.a.ov, X.b.ov,
        X.aa.ooov, X.aa.vovv,
        X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
        X.bb.ooov, X.bb.vovv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        T.abbb, t4_excitations["abbb"],
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        shift
    )
    return T, dT


def update_t2c(T, dT, X, H, shift, t3_excitations, t4_excitations):
    """Finish t2c from its raw residual: T3 and T4 terms, antisymmetry and the update."""
    T.bb, dT.bb = ccsdtq_p_loops.update_t2c(
        T.bb,
        dT.bb,
        t3_excitations["abb"], t3_excitations["bbb"],
        T.abb, T.bbb,
        X.a.ov, X.b.ov,
        X.ab.oovo, X.ab.ovvv,
        X.bb.ooov, X.bb.vovv,
        H.b.oo, H.b.vv,
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        H.bb.oovv, H.ab.oovv, H.aa.oovv,
        shift
    )
    return T, dT


def update_t3a(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)

    dT.aaa, T.aaa, t3_excitations["aaa"] = ccsdtq_p_loops.update_t3a_p_chol(
        resid,
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        H.a.ov, H.b.ov, H.aa.vovv, H.ab.vovv,
        H.aa.ooov, H.ab.ooov,
        shift
    )
    return T, dT, t3_excitations


def update_t3b(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)

    dT.aab, T.aab, t3_excitations["aab"] = ccsdtq_p_loops.update_t3b_p_chol(
        resid,
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov,
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo,
        H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        H.a.ov, H.b.ov, H.aa.ooov, H.ab.ooov,
        H.ab.oovo, H.bb.ooov, H.aa.vovv, H.ab.vovv,
        H.ab.ovvv, H.bb.vovv,
        shift
    )
    return T, dT, t3_excitations


def update_t3c(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.abb, T.abb, t3_excitations["abb"] = ccsdtq_p_loops.update_t3c_p_chol(
        resid,
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        H.b.ov, H.a.ov, H.bb.ooov, H.ab.oovo,
        H.ab.ooov, H.aa.ooov, H.bb.vovv, H.ab.ovvv,
        H.ab.vovv, H.aa.vovv,
        shift
    )
    return T, dT, t3_excitations


def update_t3d(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.bbb, T.bbb, t3_excitations["bbb"] = ccsdtq_p_loops.update_t3d_p_chol(
        resid,
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo, H.b.vv,
        H0.bb.oovv, H.bb.vvov, I2C_vooo,
        H.bb.oooo, H.bb.voov,
        H0.ab.oovv, H.ab.ovvo,
        H0.b.oo, H0.b.vv,
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        H.b.ov, H.a.ov, H.bb.vovv, H.ab.ovvv,
        H.bb.ooov, H.ab.oovo,
        shift
    )
    return T, dT, t3_excitations


def add_VT3_intermediates(T, H, H0, t3_excitations):
    """
    Add the (V_N T3)_C pieces to the vooo/vvov-type blocks of HBar that enter the T4 equations
    (cf. ccpy.hbar.hbar_ccsdt.add_VT3_intermediates).
    """
    H.aa.vooo, H.ab.vooo, H.ab.ovoo, H.aa.vvov,\
    H.ab.vvov, H.ab.vvvo, H.bb.vooo, H.bb.vvov = ccsdtq_p_loops.add_vt3_p(
        H.aa.vooo, H.ab.vooo, H.ab.ovoo, H.aa.vvov,
        H.ab.vvov, H.ab.vvvo, H.bb.vooo, H.bb.vvov,
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv
    )
    return H


def update_t4a(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t4a amplitudes by calculating the projection <ijklabcd|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    dT.aaaa, T.aaaa, t4_excitations["aaaa"] = ccsdtq_p_loops.update_t4a_p_chol(
        resid,
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.ab.ooov,
        H.ab.voov, H.ab.vovv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations


def update_t4b(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t4b amplitudes by calculating the projection <ijkl~abcd~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    dT.aaab, T.aaab, t4_excitations["aaab"] = ccsdtq_p_loops.update_t4b_p_chol(
        resid,
        T.aaab, t4_excitations["aaab"],
        T.aaaa, t4_excitations["aaaa"],
        T.aabb, t4_excitations["aabb"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.b.oo,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.bb.ooov, H.bb.voov, H.bb.vovv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations


def update_t4c(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t4c amplitudes by calculating the projection <ijk~l~abc~d~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    dT.aabb, T.aabb, t4_excitations["aabb"] = ccsdtq_p_loops.update_t4c_p_chol(
        resid,
        T.aabb, t4_excitations["aabb"],
        T.aaab, t4_excitations["aaab"],
        T.abbb, t4_excitations["abbb"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.aa, T.ab, T.bb,
        H.a.oo, H.b.oo,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.bb.oooo, H.bb.ooov, H.bb.vooo,
        H.bb.voov, H.bb.vovv, H.bb.vvov,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations


def update_t4d(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t4d amplitudes by calculating the projection <ij~k~l~ab~c~d~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    dT.abbb, T.abbb, t4_excitations["abbb"] = ccsdtq_p_loops.update_t4d_p_chol(
        resid,
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        T.bbbb, t4_excitations["bbbb"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.b.oo,
        H.aa.ooov, H.aa.voov, H.aa.vovv, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.bb.oooo, H.bb.ooov, H.bb.vooo,
        H.bb.voov, H.bb.vovv, H.bb.vvov,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations


def update_t4e(T, dT, H, H0, shift, t3_excitations, t4_excitations, resid):
    """
    Update t4e amplitudes by calculating the projection <i~j~k~l~a~b~c~d~|(H_N exp(T1+T2+T3+T4))_C|0>,
    starting from its Hbar(vvvv) part `resid`.
    """
    dT.bbbb, T.bbbb, t4_excitations["bbbb"] = ccsdtq_p_loops.update_t4e_p_chol(
        resid,
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo,
        H.ab.oovo, H.ab.ovvo, H.ab.ovvv, H.bb.oooo,
        H.bb.ooov, H.bb.vooo, H.bb.voov, H.bb.vovv,
        H.bb.vvov,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations
