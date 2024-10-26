import numpy as np
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
    dL.a = -1.0 * np.einsum("m,im->i", L.a, H.a.oo, optimize=True)
    dL.a -= 0.5 * np.einsum("mfn,finm->i", L.aa, H.aa.vooo, optimize=True)
    dL.a -= np.einsum("mfn,ifmn->i", L.ab, H.ab.ovoo, optimize=True)
    dL.a += np.einsum("ibaj,abj->i", H.ab.ovvo, X["ab"]["vvo"], optimize=True)
    dL.a += np.einsum("bija,abj->i", H.aa.voov, X["aa"]["vvo"], optimize=True)
    dL.a += 0.5 * np.einsum("ljk,iklj->i", X["aa"]["ooo"], H.aa.oooo, optimize=True)
    dL.a += np.einsum("jcb,ibjc->i", X["ab"]["ovv"], H.ab.ovov, optimize=True)
    dL.a += np.einsum("ljk,iklj->i", X["ab"]["ooo"], H.ab.oooo, optimize=True)
    return dL

def build_LH_2A(dL, L, l3_excitations, H, X):
    dL.aa = np.einsum("i,jb->ibj", L.a, H.a.ov, optimize=True)
    dL.aa -= 0.5 * np.einsum("m,ijmb->ibj", L.a, H.aa.ooov, optimize=True)
    dL.aa += 0.5 * np.einsum("iej,eb->ibj", L.aa, H.a.vv, optimize=True)
    dL.aa -= np.einsum("ibm,jm->ibj", L.aa, H.a.oo, optimize=True)
    dL.aa += 0.25 * np.einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo, optimize=True)
    dL.aa += np.einsum("iem,ejmb->ibj", L.aa, H.aa.voov, optimize=True)
    dL.aa += np.einsum("iem,jebm->ibj", L.ab, H.ab.ovvo, optimize=True)
    dL.aa += 0.5 * np.einsum("e,ijeb->ibj", X["a"]["v"], H.aa.oovv, optimize=True)
    dL.aa += np.einsum("fej,eibf->ibj", X["aa"]["vvo"], H.aa.vovv, optimize=True)
    dL.aa -= 0.5 * np.einsum("fbm,jimf->ibj", X["aa"]["vvo"], H.aa.ooov, optimize=True)
    dL.aa -= np.einsum("imn,njmb->ibj", X["aa"]["ooo"], H.aa.ooov, optimize=True)
    dL.aa -= np.einsum("imn,jnbm->ibj", X["ab"]["ooo"], H.ab.oovo, optimize=True)
    dL.aa -= np.einsum("ife,jebf->ibj", X["ab"]["ovv"], H.ab.ovvv, optimize=True)
    dL.aa = leftipeom3_p_loops.build_lh_2a(
            dL.aa,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            H.aa.vooo, H.aa.vvov, H.ab.ovoo, H.ab.vvvo,
    )
    return dL

def build_LH_2B(dL, L, l3_excitations, H, X):
    dL.ab = np.einsum("i,jb->ibj", L.a, H.b.ov, optimize=True)
    dL.ab -= np.einsum("m,ijmb->ibj", L.a, H.ab.ooov, optimize=True)
    dL.ab -= np.einsum("ibm,jm->ibj", L.ab, H.b.oo, optimize=True)
    dL.ab -= np.einsum("mbj,im->ibj", L.ab, H.a.oo, optimize=True)
    dL.ab += np.einsum("iej,eb->ibj", L.ab, H.b.vv, optimize=True)
    dL.ab += np.einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo, optimize=True)
    dL.ab += np.einsum("iem,ejmb->ibj", L.aa, H.ab.voov, optimize=True)
    dL.ab += np.einsum("iem,ejmb->ibj", L.ab, H.bb.voov, optimize=True)
    dL.ab -= np.einsum("mej,iemb->ibj", L.ab, H.ab.ovov, optimize=True)
    dL.ab += np.einsum("e,ijeb->ibj", X["a"]["v"], H.ab.oovv, optimize=True)
    dL.ab += np.einsum("fei,ejfb->ibj", X["aa"]["vvo"], H.ab.vovv, optimize=True)
    dL.ab -= np.einsum("ife,ejfb->ibj", X["ab"]["ovv"], H.bb.vovv, optimize=True)
    dL.ab -= np.einsum("ebm,ijem->ibj", X["ab"]["vvo"], H.ab.oovo, optimize=True)
    dL.ab += np.einsum("fej,iefb->ibj", X["ab"]["vvo"], H.ab.ovvv, optimize=True)
    dL.ab -= np.einsum("imn,njmb->ibj", X["aa"]["ooo"], H.ab.ooov, optimize=True)
    dL.ab -= np.einsum("imn,njmb->ibj", X["ab"]["ooo"], H.bb.ooov, optimize=True)
    dL.ab += np.einsum("njm,imnb->ibj", X["ab"]["ooo"], H.ab.ooov, optimize=True)
    dL.ab -= np.einsum("meb,ijme->ibj", X["ab"]["ovv"], H.ab.ooov, optimize=True)
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


