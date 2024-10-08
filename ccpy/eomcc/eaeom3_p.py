import numpy as np
from ccpy.lib.core import eaeom3_p_loops
from ccpy.eomcc.eaeom3_intermediates import get_eaeom3_p_intermediates

def update(R, omega, H, RHF_symmetry, system, r3_excitations):
    R.a, R.aa, R.ab, R.aaa, R.aab, R.abb = eaeom3_p_loops.update_r(
        R.a,
        R.aa,
        R.ab,
        R.aaa, 
        r3_excitations["aaa"],
        R.aab, 
        r3_excitations["aab"],
        R.abb, 
        r3_excitations["abb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system, t3_excitations, r3_excitations):

    # determine whether r3 updates should be done. Stupid compatibility with
    # empty sections r3_excitations
    do_r3 = {"aaa": True, "aab": True, "abb": True}
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["abb"] = False

    # Get intermediates
    X = get_eaeom3_p_intermediates(H, R, r3_excitations)
    # update R1
    dR = build_HR_1A(dR, R, r3_excitations, T, H)
    # update R2
    dR = build_HR_2A(dR, R, r3_excitations, T, H)
    dR = build_HR_2B(dR, R, r3_excitations, T, H)
    # update R3
    if do_r3["aaa"]:
        dR, R, r3_excitations = build_HR_3A(dR, R, r3_excitations, T, X, H)
    if do_r3["aab"]:
        dR, R, r3_excitations = build_HR_3B(dR, R, r3_excitations, T, X, H)
    if do_r3["abb"]:
        dR, R, r3_excitations = build_HR_3C(dR, R, r3_excitations, T, X, H)
    return dR.flatten()

def build_HR_1A(dR, R, r3_excitations, T, H):
    """Calculate the projection <a|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.a = np.einsum("ae,e->a", H.a.vv, R.a, optimize=True)
    dR.a += 0.5 * np.einsum("anef,efn->a", H.aa.vovv, R.aa, optimize=True)
    dR.a += np.einsum("anef,efn->a", H.ab.vovv, R.ab, optimize=True)
    dR.a += np.einsum("me,aem->a", H.a.ov, R.aa, optimize=True)
    dR.a += np.einsum("me,aem->a", H.b.ov, R.ab, optimize=True)
    dR.a = eaeom3_p_loops.build_hr_1a(
            dR.a,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            H.aa.oovv, H.ab.oovv, H.bb.oovv,
    )
    return dR

def build_HR_2A(dR, R, r3_excitations, T, H):
    """Calculate the projection <ajb|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.aa = 0.5 * np.einsum("baje,e->abj", H.aa.vvov, R.a, optimize=True)
    dR.aa -= 0.5 * np.einsum("mj,abm->abj", H.a.oo, R.aa, optimize=True)
    dR.aa += 0.25 * np.einsum("abef,efj->abj", H.aa.vvvv, R.aa, optimize=True)
    I1 = (
        0.5 * np.einsum("mnef,efn->m", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,efn->m", H.ab.oovv, R.ab, optimize=True)
    )
    dR.aa -= 0.5 * np.einsum("m,abmj->abj", I1, T.aa, optimize=True)
    dR.aa += np.einsum("ae,ebj->abj", H.a.vv, R.aa, optimize=True)
    dR.aa += np.einsum("bmje,aem->abj", H.aa.voov, R.aa, optimize=True)
    dR.aa += np.einsum("bmje,aem->abj", H.ab.voov, R.ab, optimize=True)
    dR.aa = eaeom3_p_loops.build_hr_2a(
            dR.aa,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            H.a.ov, H.b.ov,
            H.aa.ooov, H.aa.vovv, H.ab.ooov, H.ab.vovv
    )
    return dR

def build_HR_2B(dR, R, r3_excitations, T, H):
    """Calculate the projection <aj~b~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.ab = np.einsum("abej,e->abj", H.ab.vvvo, R.a, optimize=True)
    dR.ab += np.einsum("ae,ebj->abj", H.a.vv, R.ab, optimize=True)
    dR.ab += np.einsum("be,aej->abj", H.b.vv, R.ab, optimize=True)
    dR.ab -= np.einsum("mj,abm->abj", H.b.oo, R.ab, optimize=True)
    dR.ab += np.einsum("mbej,aem->abj", H.ab.ovvo, R.aa, optimize=True)
    dR.ab += np.einsum("bmje,aem->abj", H.bb.voov, R.ab, optimize=True)
    dR.ab -= np.einsum("amej,ebm->abj", H.ab.vovo, R.ab, optimize=True)
    dR.ab += np.einsum("abef,efj->abj", H.ab.vvvv, R.ab, optimize=True)
    I1 = (
        0.5 * np.einsum("mnef,efn->m", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,efn->m", H.ab.oovv, R.ab, optimize=True)
    )
    dR.ab -= np.einsum("m,abmj->abj", I1, T.ab, optimize=True)
    dR.ab = eaeom3_p_loops.build_hr_2b(
            dR.ab,
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            H.a.ov, H.b.ov,
            H.aa.vovv, 
            H.ab.oovo, H.ab.vovv, H.ab.ovvv,
            H.bb.ooov, H.bb.vovv,
    )
    return dR

def build_HR_3A(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <abcjk|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.aaa, R.aaa, r3_excitations["aaa"] = eaeom3_p_loops.build_hr_3a(
            R.aa,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            T.aa,
            H.a.oo, H.a.vv,
            H.aa.vvvv.transpose(2, 3, 0, 1), H.aa.oooo, H.aa.voov, H.aa.vooo, H.aa.vvov,
            H.ab.voov,
            X["aa"]["voo"], X["aa"]["vvv"],
    )
    return dR, R, r3_excitations

def build_HR_3B(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <abc~jk~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.aab, R.aab, r3_excitations["aab"] = eaeom3_p_loops.build_hr_3b(
            R.aa, R.ab,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            T.aa, T.ab,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.vvvv.transpose(2, 3, 0, 1), H.aa.voov, H.aa.vvov,
            H.ab.vvvv.transpose(2, 3, 0, 1), H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
            H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo,
            H.bb.voov,
            X["aa"]["voo"], X["aa"]["vvv"],
            X["ab"]["voo"], X["ab"]["ovo"], X["ab"]["vvv"],
    )
    return dR, R, r3_excitations

def build_HR_3C(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~c~j~k~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    dR.abb, R.abb, r3_excitations["abb"] = eaeom3_p_loops.build_hr_3c(
            R.ab,
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            T.ab, T.bb,
            H.a.vv, H.b.oo, H.b.vv,
            H.ab.vvvv.transpose(2, 3, 0, 1), H.ab.vovo, H.ab.ovvo, H.ab.vvvo,
            H.bb.vvvv.transpose(2, 3, 0, 1), H.bb.oooo, H.bb.voov, H.bb.vooo, H.bb.vvov,
            X["ab"]["voo"], X["ab"]["ovo"], X["ab"]["vvv"],
    )
    return dR, R, r3_excitations
