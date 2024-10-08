import numpy as np
from ccpy.eomcc.eomccsdt_intermediates import get_eomccsd_intermediates, get_eomccsdt_intermediates, add_R3_p_terms
from ccpy.lib.core import eomccsdt_p_loops

def update(R, omega, H, RHF_symmetry, system, r3_excitations):
    R.a, R.b, R.aa, R.ab, R.bb, R.aaa, R.aab, R.abb, R.bbb = eomccsdt_p_loops.update_r(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.bb,
        R.aaa,
        r3_excitations["aaa"],
        R.aab,
        r3_excitations["aab"],
        R.abb,
        r3_excitations["abb"],
        R.bbb,
        r3_excitations["bbb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    if RHF_symmetry:
        R.b = R.a.copy()
        R.bb = R.aa.copy()
        R.abb = R.aab.copy()
        R.bbb = R.aaa.copy()
    return R

def HR(dR, R, T, H, flag_RHF, system, t3_excitations, r3_excitations):

    # determine whether r3 updates should be done. Stupid compatibility with
    # empty sections of t3_excitations or r3_excitations
    do_r3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["abb"] = False
    if np.array_equal(r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["bbb"] = False

    dR = build_HR_1A(dR, R, r3_excitations, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR = build_HR_1B(dR, R, r3_excitations, H)

    # Get H*R EOMCCSD intermediates
    X0 = get_eomccsd_intermediates(H, R, system)
    dR = build_HR_2A(dR, R, r3_excitations, T, t3_excitations, H, X0)
    dR = build_HR_2B(dR, R, r3_excitations, T, t3_excitations, H, X0)
    if flag_RHF:
        dR.bb = dR.aa.copy()
    else:
        dR = build_HR_2C(dR, R, r3_excitations, T, t3_excitations, H, X0)

    # Add on terms needed to make EOMCCSDT intermediates
    X = get_eomccsdt_intermediates(H, R, T, X0, system)
    X = add_R3_p_terms(X, H, R, r3_excitations)
    if do_r3["aaa"]:
        dR, R, r3_excitations = build_HR_3A(dR, R, r3_excitations, T, t3_excitations, H, X)
    if do_r3["aab"]:
        dR, R, r3_excitations = build_HR_3B(dR, R, r3_excitations, T, t3_excitations, H, X)
    if flag_RHF:
        R.abb = R.aab.copy()
        dR.abb = dR.aab.copy()
        r3_excitations["abb"] = r3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]

        R.bbb = R.bbb.copy()
        dR.bbb = dR.aaa.copy()
        r3_excitations["bbb"] = r3_excitations["aaa"].copy()
    else:
        if do_r3["abb"]:
            dR, R, r3_excitations = build_HR_3C(dR, R, r3_excitations, T, t3_excitations, H, X)
        if do_r3["bbb"]:
            dR, R, r3_excitations = build_HR_3D(dR, R, r3_excitations, T, t3_excitations, H, X)
    return dR.flatten()

def build_HR_1A(dR, R, r3_excitations, H):
    """< ia | [H(2)*(R1+R2+R3)]_C | 0 >"""
    dR.a = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    dR.a += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    dR.a += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    dR.a += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    dR.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
    dR.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
    dR.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
    dR.a += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
    dR.a += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
    dR.a += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)
    # Parts contracted with R3
    dR.a = eomccsdt_p_loops.build_hr_1a(
                                            dR.a,
                                            R.aaa, r3_excitations["aaa"], 
                                            R.aab, r3_excitations["aab"], 
                                            R.abb, r3_excitations["abb"],
                                            H.aa.oovv, H.ab.oovv, H.bb.oovv
    )
    return dR

def build_HR_1B(dR, R, r3_excitations, H):
    """< i~a~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    dR.b = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    dR.b += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    dR.b += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    dR.b += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    dR.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
    dR.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
    dR.b += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    dR.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
    dR.b += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    dR.b += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)
    # Parts contracted with R3
    dR.b = eomccsdt_p_loops.build_hr_1b(
                                            dR.b,
                                            R.aab, r3_excitations["aab"], 
                                            R.abb, r3_excitations["abb"], 
                                            R.bbb, r3_excitations["bbb"],
                                            H.aa.oovv, H.ab.oovv, H.bb.oovv
    )
    return dR

def build_HR_2A(dR, R, r3_excitations, T, t3_excitations, H, X):
    """ < ijab | [H(2)*(R1+R2+R3)]_C | 0 > """
    dR.aa = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, R.aa, optimize=True)  # A(ij)
    dR.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, R.aa, optimize=True)  # A(ab)
    dR.aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.aa, optimize=True)
    dR.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, R.aa, optimize=True)
    dR.aa += np.einsum("amie,ebmj->abij", H.aa.voov, R.aa, optimize=True)  # A(ij)A(ab)
    dR.aa += np.einsum("amie,bejm->abij", H.ab.voov, R.ab, optimize=True)  # A(ij)A(ab)
    dR.aa -= 0.5 * np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    dR.aa += 0.5 * np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)
    dR.aa += 0.5 * np.einsum("be,aeij->abij", X.a.vv, T.aa, optimize=True)  # A(ab)
    dR.aa -= 0.5 * np.einsum("mj,abim->abij", X.a.oo, T.aa, optimize=True)  # A(ij)
    # Parts contracted with T3 and R3; antisymmetrization included
    dR.aa = eomccsdt_p_loops.build_hr_2a(
                                            dR.aa,
                                            R.aaa, r3_excitations["aaa"],
                                            R.aab, r3_excitations["aab"],
                                            T.aaa, t3_excitations["aaa"],
                                            T.aab, t3_excitations["aab"],
                                            H.a.ov, H.b.ov,
                                            H.aa.ooov, H.aa.vovv,
                                            H.ab.ooov, H.ab.vovv,
                                            X.a.ov, X.b.ov,
    )
    return dR

def build_HR_2B(dR, R, r3_excitations, T, t3_excitations, H, X):
    """< ij~ab~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    dR.ab = np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    dR.ab += np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    dR.ab -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    dR.ab -= np.einsum("mj,abim->abij", H.b.oo, R.ab, optimize=True)
    dR.ab += np.einsum("mnij,abmn->abij", H.ab.oooo, R.ab, optimize=True)
    dR.ab += np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    dR.ab += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    dR.ab += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    dR.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, R.aa, optimize=True)
    dR.ab += np.einsum("bmje,aeim->abij", H.bb.voov, R.ab, optimize=True)
    dR.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    dR.ab -= np.einsum("amej,ebim->abij", H.ab.vovo, R.ab, optimize=True)
    dR.ab += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    dR.ab += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    dR.ab -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    dR.ab -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)
    dR.ab += np.einsum("ae,ebij->abij", X.a.vv, T.ab, optimize=True)
    dR.ab -= np.einsum("mi,abmj->abij", X.a.oo, T.ab, optimize=True)
    dR.ab += np.einsum("be,aeij->abij", X.b.vv, T.ab, optimize=True)
    dR.ab -= np.einsum("mj,abim->abij", X.b.oo, T.ab, optimize=True)
    # Parts contracted with T3 and R3
    dR.ab = eomccsdt_p_loops.build_hr_2b(
                                            dR.ab,
                                            R.aab, r3_excitations["aab"],
                                            R.abb, r3_excitations["abb"],
                                            T.aab, t3_excitations["aab"],
                                            T.abb, t3_excitations["abb"],
                                            H.a.ov, H.b.ov,
                                            H.aa.ooov, H.aa.vovv, 
                                            H.ab.ooov, H.ab.vovv, H.ab.oovo, H.ab.ovvv,
                                            H.bb.ooov, H.bb.vovv,
                                            X.a.ov, X.b.ov,
    )
    return dR

def build_HR_2C(dR, R, r3_excitations, T, t3_excitations, H, X):
    """< i~j~a~b~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    dR.bb = -0.5 * np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)  # A(ij)
    dR.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)  # A(ab)
    dR.bb += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, R.bb, optimize=True)
    dR.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    dR.bb += np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)  # A(ij)A(ab)
    dR.bb += np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)  # A(ij)A(ab)
    dR.bb -= 0.5 * np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    dR.bb += 0.5 * np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)
    dR.bb += 0.5 * np.einsum("be,aeij->abij", X.b.vv, T.bb, optimize=True)  # A(ab)
    dR.bb -= 0.5 * np.einsum("mj,abim->abij", X.b.oo, T.bb, optimize=True)  # A(ij)
    # Parts contracted with T3 and R3; antisymmetrization included
    dR.bb = eomccsdt_p_loops.build_hr_2c(
                                            dR.bb,
                                            R.abb, r3_excitations["abb"],
                                            R.bbb, r3_excitations["bbb"],
                                            T.abb, t3_excitations["abb"],
                                            T.bbb, t3_excitations["bbb"],
                                            H.a.ov, H.b.ov,
                                            H.ab.oovo, H.ab.ovvv,
                                            H.bb.ooov, H.bb.vovv,
                                            X.a.ov, X.b.ov,
    )
    return dR

def build_HR_3A(dR, R, r3_excitations, T, t3_excitations, H, X):

    dR.aaa, R.aaa, r3_excitations["aaa"] = eomccsdt_p_loops.build_hr_3a(
                                            R.aa,
                                            R.aaa, r3_excitations["aaa"],
                                            R.aab, r3_excitations["aab"],
                                            T.aa,
                                            T.aaa, t3_excitations["aaa"],
                                            T.aab, t3_excitations["aab"],
                                            H.a.oo, H.a.vv,
                                            H.aa.oooo, H.aa.vooo, H.aa.oovv,
                                            H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov, H.aa.vvvv.transpose(2, 3, 0, 1),
                                            H.ab.voov,
                                            X.a.oo, X.a.vv,
                                            X.aa.oooo, X.aa.vooo, X.aa.oovv,
                                            X.aa.voov, X.aa.vvov, X.aa.vvvv.transpose(2, 3, 0, 1),
                                            X.ab.voov,
    )
    return dR, R, r3_excitations

def build_HR_3B(dR, R, r3_excitations, T, t3_excitations, H, X):

    dR.aab, R.aab, r3_excitations["aab"] = eomccsdt_p_loops.build_hr_3b(
                                            R.aa, R.ab,
                                            R.aaa, r3_excitations["aaa"],
                                            R.aab, r3_excitations["aab"],
                                            R.abb, r3_excitations["abb"],
                                            T.aa, T.ab,
                                            T.aaa, t3_excitations["aaa"],
                                            T.aab, t3_excitations["aab"],
                                            T.abb, t3_excitations["abb"],
                                            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                            H.aa.oooo, H.aa.vooo, H.aa.oovv,
                                            H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov.transpose(3, 0, 1, 2), H.aa.vvvv.transpose(2, 3, 0, 1),
                                            H.ab.oooo, H.ab.vooo, H.ab.ovoo,
                                            H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3),
                                            H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvov.transpose(3, 0, 1, 2),
                                            H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                                            H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                                            X.a.oo, X.a.vv, X.b.oo, X.b.vv,
                                            X.aa.oooo, X.aa.vooo, X.aa.oovv,
                                            X.aa.voov.transpose(1, 3, 0, 2), X.aa.vvov.transpose(3, 0, 1, 2), X.aa.vvvv.transpose(2, 3, 0, 1),
                                            X.ab.oooo, X.ab.vooo, X.ab.ovoo,
                                            X.ab.oovv, X.ab.voov.transpose(1, 3, 0, 2), X.ab.vovo.transpose(1, 2, 0, 3),
                                            X.ab.ovov.transpose(0, 3, 1, 2), X.ab.ovvo.transpose(0, 2, 1, 3), X.ab.vvov.transpose(3, 0, 1, 2),
                                            X.ab.vvvo.transpose(2, 0, 1, 3), X.ab.vvvv.transpose(3, 2, 1, 0),
                                            X.bb.oovv, X.bb.voov.transpose(1, 3, 0, 2),
    )
    return dR, R, r3_excitations

def build_HR_3C(dR, R, r3_excitations, T, t3_excitations, H, X):

    dR.abb, R.abb, r3_excitations["abb"] = eomccsdt_p_loops.build_hr_3c(
                                            R.ab, R.bb,
                                            R.aab, r3_excitations["aab"],
                                            R.abb, r3_excitations["abb"],
                                            R.bbb, r3_excitations["bbb"],
                                            T.ab, T.bb,
                                            T.aab, t3_excitations["aab"],
                                            T.abb, t3_excitations["abb"],
                                            T.bbb, t3_excitations["bbb"],
                                            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                            H.aa.oovv, H.aa.voov,
                                            H.ab.oooo, H.ab.vooo, H.ab.ovoo,
                                            H.ab.oovv, H.ab.voov, H.ab.vovo,
                                            H.ab.ovov, H.ab.ovvo, H.ab.vvov,
                                            H.ab.vvvo, H.ab.vvvv.transpose(2, 3, 0, 1),
                                            H.bb.oooo, H.bb.vooo, H.bb.oovv,
                                            H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
                                            X.a.oo, X.a.vv, X.b.oo, X.b.vv,
                                            X.aa.oovv, X.aa.voov,
                                            X.ab.oooo, X.ab.vooo, X.ab.ovoo,
                                            X.ab.oovv, X.ab.voov, X.ab.vovo,
                                            X.ab.ovov, X.ab.ovvo, X.ab.vvov,
                                            X.ab.vvvo, X.ab.vvvv.transpose(2, 3, 0, 1),
                                            X.bb.oooo, X.bb.vooo, X.bb.oovv,
                                            X.bb.voov, X.bb.vvov, X.bb.vvvv.transpose(2, 3, 0, 1),
    )
    return dR, R, r3_excitations

def build_HR_3D(dR, R, r3_excitations, T, t3_excitations, H, X):

    dR.bbb, R.bbb, r3_excitations["bbb"] = eomccsdt_p_loops.build_hr_3d(
                                            R.bb,
                                            R.abb, r3_excitations["abb"],
                                            R.bbb, r3_excitations["bbb"],
                                            T.bb,
                                            T.abb, t3_excitations["abb"],
                                            T.bbb, t3_excitations["bbb"],
                                            H.b.oo, H.b.vv,
                                            H.bb.oooo, H.bb.vooo, H.bb.oovv,
                                            H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
                                            H.ab.ovvo,
                                            X.b.oo, X.b.vv,
                                            X.bb.oooo, X.bb.vooo, X.bb.oovv,
                                            X.bb.voov, X.bb.vvov, X.bb.vvvv.transpose(2, 3, 0, 1),
                                            X.ab.ovvo,
    )
    return dR, R, r3_excitations
