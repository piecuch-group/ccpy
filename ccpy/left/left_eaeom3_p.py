import numpy as np
from ccpy.lib.core import eaeom3_p_loops
from ccpy.lib.core import lefteaeom3_p_loops
from ccpy.left.left_eaeom_intermediates import get_lefteaeom3_p_intermediates

def update_l(L, omega, H, RHF_symmetry, system, l3_excitations):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = eaeom3_p_loops.update_r(
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
    X = get_lefteaeom3_p_intermediates(L, l3_excitations, T, do_l3, system)
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
    dL.a = np.einsum("e,ea->a", L.a, H.a.vv, optimize=True)
    dL.a += 0.5 * np.einsum("efn,fena->a", L.aa, H.aa.vvov, optimize=True)
    dL.a += np.einsum("efn,efan->a", L.ab, H.ab.vvvo, optimize=True)
    # parts contracted with L3 (verified against explicit 3-body hbars)
    dL.a -= np.einsum("mfan,mfn->a", H.ab.ovvo, X["ab"]["ovo"], optimize=True)
    dL.a -= np.einsum("fmna,mfn->a", H.aa.voov, X["aa"]["ovo"], optimize=True)
    dL.a -= 0.5 * np.einsum("fge,feag->a", X["aa"]["vvv"], H.aa.vvvv, optimize=True)
    dL.a -= np.einsum("eman,enm->a", H.ab.vovo, X["ab"]["voo"], optimize=True)
    dL.a -= np.einsum("efg,egaf->a", X["ab"]["vvv"], H.ab.vvvv, optimize=True)
    return dL

def build_LH_2A(dL, L, l3_excitations, H, X):
    dL.aa = np.einsum("a,jb->abj", L.a, H.a.ov, optimize=True)
    dL.aa += 0.5 * np.einsum("e,ejab->abj", L.a, H.aa.vovv, optimize=True)
    dL.aa += np.einsum("ebj,ea->abj", L.aa, H.a.vv, optimize=True)
    dL.aa -= 0.5 * np.einsum("abm,jm->abj", L.aa, H.a.oo, optimize=True)
    dL.aa += np.einsum("afn,fjnb->abj", L.aa, H.aa.voov, optimize=True)
    dL.aa += np.einsum("afn,jfbn->abj", L.ab, H.ab.ovvo, optimize=True)
    dL.aa += 0.25 * np.einsum("efj,efab->abj", L.aa, H.aa.vvvv, optimize=True)
    dL.aa -= 0.5 * np.einsum("mjab,m->abj", H.aa.oovv, X["a"]["o"], optimize=True)
    # 3-body hbar terms (verified against explicit 3-body hbars)
    dL.aa += np.einsum("mbn,jmna->abj", X["aa"]["ovo"], H.aa.ooov, optimize=True) #
    dL.aa -= np.einsum("amn,jnbm->abj", X["ab"]["voo"], H.ab.oovo, optimize=True) #
    dL.aa -= np.einsum("aef,fjeb->abj", X["aa"]["vvv"], H.aa.vovv, optimize=True) #
    dL.aa -= np.einsum("aef,jfbe->abj", X["ab"]["vvv"], H.ab.ovvv, optimize=True) #
    dL.aa -= 0.5 * np.einsum("mej,emba->abj", X["aa"]["ovo"], H.aa.vovv, optimize=True)
    dL.aa = lefteaeom3_p_loops.build_lh_2a(
            dL.aa,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            H.aa.vooo, H.aa.vvov, H.ab.ovoo, H.ab.vvvo,
    )
    return dL

def build_LH_2B(dL, L, l3_excitations, H, X):
    dL.ab = np.einsum("a,jb->abj", L.a, H.b.ov, optimize=True)
    dL.ab += np.einsum("e,ejab->abj", L.a, H.ab.vovv, optimize=True)
    dL.ab -= np.einsum("abm,jm->abj", L.ab, H.b.oo, optimize=True)
    dL.ab += np.einsum("aej,eb->abj", L.ab, H.b.vv, optimize=True)
    dL.ab += np.einsum("ebj,ea->abj", L.ab, H.a.vv, optimize=True)
    dL.ab += np.einsum("afn,fjnb->abj", L.aa, H.ab.voov, optimize=True)
    dL.ab += np.einsum("afn,fjnb->abj", L.ab, H.bb.voov, optimize=True)
    dL.ab -= np.einsum("ebm,ejam->abj", L.ab, H.ab.vovo, optimize=True)
    dL.ab += np.einsum("efj,efab->abj", L.ab, H.ab.vvvv, optimize=True)
    dL.ab -= np.einsum("mjab,m->abj", H.ab.oovv, X["a"]["o"], optimize=True)
    # 3-body hbar terms
    dL.ab += np.einsum("man,mjnb->abj", X["aa"]["ovo"], H.ab.ooov, optimize=True) # [Ia]
    dL.ab -= np.einsum("amn,njmb->abj", X["ab"]["voo"], H.bb.ooov, optimize=True) # [Ib]
    dL.ab -= np.einsum("aef,fjeb->abj", X["aa"]["vvv"], H.ab.vovv, optimize=True) # [IIa]
    dL.ab -= np.einsum("aef,fjeb->abj", X["ab"]["vvv"], H.bb.vovv, optimize=True) # [IIb]
    dL.ab += np.einsum("nbm,njam->abj", X["ab"]["ovo"], H.ab.oovo, optimize=True) # [Iab]
    dL.ab += np.einsum("efb,ejaf->abj", X["ab"]["vvv"], H.ab.vovv, optimize=True) # [IIab]
    dL.ab -= np.einsum("ejn,enab->abj", X["ab"]["voo"], H.ab.vovv, optimize=True) # [III]
    dL.ab -= np.einsum("nfj,nfab->abj", X["ab"]["ovo"], H.ab.ovvv, optimize=True) # [IV]
    dL.ab = lefteaeom3_p_loops.build_lh_2b(
            dL.ab,
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.aa.vvov, 
            H.ab.vooo, H.ab.vvov, H.ab.vvvo,
            H.bb.vooo, H.bb.vvov,
    )
    return dL

def build_LH_3A(dL, L, l3_excitations, H, X):
    dL.aaa, L.aaa, l3_excitations["aaa"] = lefteaeom3_p_loops.build_lh_3a(
            L.a, L.aa,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            H.a.ov, H.a.oo, H.a.vv,
            H.aa.vvvv, H.aa.oooo, H.aa.voov.transpose(3, 2, 1, 0), H.aa.ooov, H.aa.vovv, H.aa.oovv,
            H.ab.ovvo.transpose(2, 3, 0, 1),
            X["aa"]["ovo"], X["aa"]["vvv"],
    )
    return dL, L, l3_excitations

def build_LH_3B(dL, L, l3_excitations, H, X):
    dL.aab, L.aab, l3_excitations["aab"] = lefteaeom3_p_loops.build_lh_3b(
            L.a, L.aa, L.ab,
            L.aaa, l3_excitations["aaa"],
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.a.ov, H.b.ov, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.vvvv, H.aa.voov.transpose(3, 2, 1, 0), H.aa.vovv, H.aa.oovv,
            H.ab.vvvv, H.ab.oooo, H.ab.ovvo.transpose(2, 3, 0, 1), H.ab.vovo.transpose(2, 3, 0, 1), H.ab.ovov.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.oovv,
            H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
            H.bb.voov.transpose(3, 2, 1, 0),
            X["aa"]["ovo"], X["aa"]["vvv"],
            X["ab"]["voo"], X["ab"]["ovo"], X["ab"]["vvv"],
    )
    return dL, L, l3_excitations

def build_LH_3C(dL, L, l3_excitations, H, X):
    dL.abb, L.abb, l3_excitations["abb"] = lefteaeom3_p_loops.build_lh_3c(
            L.a, L.ab,
            L.aab, l3_excitations["aab"],
            L.abb, l3_excitations["abb"],
            H.b.ov, H.a.vv, H.b.oo, H.b.vv,
            H.ab.vvvv, H.ab.vovo.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.vovv, H.ab.oovv,
            H.bb.vvvv, H.bb.oooo, H.bb.voov.transpose(3, 2, 1, 0), H.bb.ooov, H.bb.vovv, H.bb.oovv,
            X["ab"]["voo"], X["ab"]["ovo"], X["ab"]["vvv"],
    )
    return dL, L, l3_excitations


