import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import ipeom3_p_loops
from ccpy.lib.core import leftipeom3_p_loops
from ccpy.left.left_ipeom_intermediates import get_leftipeom3_p_intermediates

def update_l(L, omega, H, RHF_symmetry, system, l3_excitations):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = ipeom3_p_loops.update_r(
        L.a,
        L.aa,
        L.ab,
        L.aaa,
        l3_excitations["aaa"],
        L.aab,
        l3_excitations["aab"],
        L.abb,
        l3_excitations["abb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    return L

def LH_fun(dL, L, T, H, flag_RHF, system, t3_excitations, l3_excitations):

    # determine whether r3 updates should be done. Stupid compatibility with
    # empty sections r3_excitations
    do_l3 = {"aaa": True, "aab": True, "abb": True}
    if np.array_equal(l3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(l3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(l3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["abb"] = False

    # Get intermediates
    X = get_leftipeom3_p_intermediates(L, l3_excitations, T, do_l3, system)
    # update R1
    dL = build_LH_1A(dL, L, l3_excitations, H, X)
    # update R2
    dL = build_LH_2A(dL, L, l3_excitations, H, X)
    dL = build_LH_2B(dL, L, l3_excitations, H, X)
    # update R3
    if do_l3["aaa"]:
        dL, L, l3_excitations = build_LH_3A(dL, L, l3_excitations, H, X)
    if do_l3["aab"]:
        dL, L, l3_excitations = build_LH_3B(dL, L, l3_excitations, H, X)
    if do_l3["abb"]:
        dL, L, l3_excitations = build_LH_3C(dL, L, l3_excitations, H, X)
    return dL.flatten()

def build_LH_1A(dL, L, l3_excitations, H, X):
    dL.a = -1.0 * ccpy_einsum("m,im->i", L.a, H.a.oo)
    dL.a -= 0.5 * ccpy_einsum("mfn,finm->i", L.aa, H.aa.vooo)
    dL.a -= ccpy_einsum("mfn,ifmn->i", L.ab, H.ab.ovoo)
    dL.a += ccpy_einsum("ibaj,abj->i", H.ab.ovvo, X["ab"]["vvo"])
    dL.a += ccpy_einsum("bija,abj->i", H.aa.voov, X["aa"]["vvo"])
    dL.a += 0.5 * ccpy_einsum("ljk,iklj->i", X["aa"]["ooo"], H.aa.oooo)
    dL.a += ccpy_einsum("jcb,ibjc->i", X["ab"]["ovv"], H.ab.ovov)
    dL.a += ccpy_einsum("ljk,iklj->i", X["ab"]["ooo"], H.ab.oooo)
    return dL

def build_LH_2A(dL, L, l3_excitations, H, X):
    dL.aa = ccpy_einsum("i,jb->ibj", L.a, H.a.ov)
    dL.aa -= 0.5 * ccpy_einsum("m,ijmb->ibj", L.a, H.aa.ooov)
    dL.aa += 0.5 * ccpy_einsum("iej,eb->ibj", L.aa, H.a.vv)
    dL.aa -= ccpy_einsum("ibm,jm->ibj", L.aa, H.a.oo)
    dL.aa += 0.25 * ccpy_einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo)
    dL.aa += ccpy_einsum("iem,ejmb->ibj", L.aa, H.aa.voov)
    dL.aa += ccpy_einsum("iem,jebm->ibj", L.ab, H.ab.ovvo)
    dL.aa += 0.5 * ccpy_einsum("e,ijeb->ibj", X["a"]["v"], H.aa.oovv)
    dL.aa += ccpy_einsum("fej,eibf->ibj", X["aa"]["vvo"], H.aa.vovv)
    dL.aa -= 0.5 * ccpy_einsum("fbm,jimf->ibj", X["aa"]["vvo"], H.aa.ooov)
    dL.aa -= ccpy_einsum("imn,njmb->ibj", X["aa"]["ooo"], H.aa.ooov)
    dL.aa -= ccpy_einsum("imn,jnbm->ibj", X["ab"]["ooo"], H.ab.oovo)
    dL.aa -= ccpy_einsum("ife,jebf->ibj", X["ab"]["ovv"], H.ab.ovvv)
    dL.aa = leftipeom3_p_loops.build_lh_2a(
            dL.aa,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            H.aa.vooo, H.aa.vvov, H.ab.ovoo, H.ab.vvvo,
    )
    return dL

def build_LH_2B(dL, L, l3_excitations, H, X):
    dL.ab = ccpy_einsum("i,jb->ibj", L.a, H.b.ov)
    dL.ab -= ccpy_einsum("m,ijmb->ibj", L.a, H.ab.ooov)
    dL.ab -= ccpy_einsum("ibm,jm->ibj", L.ab, H.b.oo)
    dL.ab -= ccpy_einsum("mbj,im->ibj", L.ab, H.a.oo)
    dL.ab += ccpy_einsum("iej,eb->ibj", L.ab, H.b.vv)
    dL.ab += ccpy_einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo)
    dL.ab += ccpy_einsum("iem,ejmb->ibj", L.aa, H.ab.voov)
    dL.ab += ccpy_einsum("iem,ejmb->ibj", L.ab, H.bb.voov)
    dL.ab -= ccpy_einsum("mej,iemb->ibj", L.ab, H.ab.ovov)
    dL.ab += ccpy_einsum("e,ijeb->ibj", X["a"]["v"], H.ab.oovv)
    dL.ab += ccpy_einsum("fei,ejfb->ibj", X["aa"]["vvo"], H.ab.vovv)
    dL.ab -= ccpy_einsum("ife,ejfb->ibj", X["ab"]["ovv"], H.bb.vovv)
    dL.ab -= ccpy_einsum("ebm,ijem->ibj", X["ab"]["vvo"], H.ab.oovo)
    dL.ab += ccpy_einsum("fej,iefb->ibj", X["ab"]["vvo"], H.ab.ovvv)
    dL.ab -= ccpy_einsum("imn,njmb->ibj", X["aa"]["ooo"], H.ab.ooov)
    dL.ab -= ccpy_einsum("imn,njmb->ibj", X["ab"]["ooo"], H.bb.ooov)
    dL.ab += ccpy_einsum("njm,imnb->ibj", X["ab"]["ooo"], H.ab.ooov)
    dL.ab -= ccpy_einsum("meb,ijme->ibj", X["ab"]["ovv"], H.ab.ooov)
    dL.ab = leftipeom3_p_loops.build_lh_2b(
            dL.ab,
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.aa.vooo, 
            H.ab.vooo, H.ab.ovoo, H.ab.vvov,
            H.bb.vooo, H.bb.vvov,
    )
    return dL

def build_LH_3A(dL, L, l3_excitations, H, X):
    dL.aaa, L.aaa, l3_excitations["aaa"] = leftipeom3_p_loops.build_lh_3a(
            L.a, L.aa,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            H.a.ov, H.a.oo.transpose(1, 0), H.a.vv,
            H.aa.vvvv, H.aa.oooo.transpose(2, 3, 0, 1), H.aa.voov.transpose(3, 2, 1, 0), H.aa.ooov, H.aa.vovv, H.aa.oovv,
            H.ab.ovvo.transpose(2, 3, 0, 1),
            X["aa"]["vvo"], X["aa"]["ooo"],
    )
    return dL, L, l3_excitations

def build_LH_3B(dL, L, l3_excitations, H, X):
    dL.aab, L.aab, l3_excitations["aab"] = leftipeom3_p_loops.build_lh_3b(
            L.a, L.aa, L.ab,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.a.ov, H.b.ov, H.a.oo.transpose(1, 0), H.a.vv, H.b.oo.transpose(1, 0), H.b.vv,
            H.aa.oooo.transpose(2, 3, 0, 1), H.aa.voov.transpose(3, 2, 1, 0), H.aa.ooov, H.aa.oovv,
            H.ab.vvvv, H.ab.oooo.transpose(2, 3, 0, 1), H.ab.ovvo.transpose(2, 3, 0, 1), H.ab.vovo.transpose(2, 3, 0, 1), H.ab.ovov.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.oovv,
            H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
            H.bb.voov.transpose(3, 2, 1, 0),
            X["aa"]["vvo"], X["aa"]["ooo"],
            X["ab"]["vvo"], X["ab"]["ovv"], X["ab"]["ooo"],
    )
    return dL, L, l3_excitations

def build_LH_3C(dL, L, l3_excitations, H, X):
    dL.abb, L.abb, l3_excitations["abb"] = leftipeom3_p_loops.build_lh_3c(
            L.a, L.ab,
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.b.ov, H.a.oo.transpose(1, 0), H.b.oo.transpose(1, 0), H.b.vv,
            H.ab.oooo.transpose(2, 3, 0, 1), H.ab.ovov.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.ooov, H.ab.oovv,
            H.bb.vvvv, H.bb.oooo.transpose(2, 3, 0, 1), H.bb.voov.transpose(3, 2, 1, 0), H.bb.ooov, H.bb.vovv, H.bb.oovv,
            X["ab"]["vvo"], X["ab"]["ovv"], X["ab"]["ooo"],
    )
    return dL, L, l3_excitations


