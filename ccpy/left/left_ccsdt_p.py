import numpy as np
from ccpy.left.left_cc_intermediates import build_left_ccsdt_p_intermediates
from ccpy.lib.core import leftccsdt_p_loops, eomccsdt_p_loops

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system, t3_excitations, l3_excitations, pspace=None):

    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations
    do_l3 = {"aaa" : True, "aab" : True, "abb" : True, "bbb" : True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    if np.array_equal(l3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(l3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(l3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["abb"] = False
    if np.array_equal(l3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["bbb"] = False

    # get LT intermediates
    X = build_left_ccsdt_p_intermediates(L, l3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=flag_RHF)

    # build L1
    LH = build_LH_1A(L, LH, T, H, X)
    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, H, X)

    # build L2
    LH = build_LH_2A(L, LH, T, H, X, l3_excitations)
    LH = build_LH_2B(L, LH, T, H, X, l3_excitations)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H, X, l3_excitations)

    # build L3
    if do_l3["aaa"]:
        LH, L, l3_excitations = build_LH_3A(L, LH, H, X, l3_excitations)
    if do_l3["aab"]:
        LH, L, l3_excitations = build_LH_3B(L, LH, H, X, l3_excitations)
    if flag_RHF:
        L.abb = L.aab.copy()
        LH.abb = LH.aab.copy()
        l3_excitations["abb"] = l3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]

        L.bbb = L.aaa.copy()
        LH.bbb = LH.aaa.copy()
        l3_excitations["bbb"] = l3_excitations["aaa"].copy()
    else:
        if do_l3["abb"]:
            LH, L, l3_excitations = build_LH_3C(L, LH, H, X, l3_excitations)
        if do_l3["bbb"]:
            LH, L, l3_excitations = build_LH_3D(L, LH, H, X, l3_excitations)

    # Add Hamiltonian if ground-state calculation
    if is_ground:
        LH.a += np.transpose(H.a.ov, (1, 0))
        LH.b += np.transpose(H.b.ov, (1, 0))
        LH.aa += np.transpose(H.aa.oovv, (2, 3, 0, 1))
        LH.ab += np.transpose(H.ab.oovv, (2, 3, 0, 1))
        LH.bb += np.transpose(H.bb.oovv, (2, 3, 0, 1))

    # Update the L vector and LH residual measure
    L.a, L.b, LH.a, LH.b = leftccsdt_p_loops.update_l1(L.a, L.b, LH.a, LH.b,
                                                                         omega,
                                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                         shift)
    L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb = leftccsdt_p_loops.update_l2(L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb,
                                                                                          omega,
                                                                                          H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                                          shift)
    L.aaa, L.aab, L.abb, L.bbb, LH.aaa, LH.aab, LH.abb, LH.bbb = leftccsdt_p_loops.update_l3(
                                                         L.aaa, l3_excitations["aaa"],
                                                         L.aab, l3_excitations["aab"],
                                                         L.abb, l3_excitations["abb"],
                                                         L.bbb, l3_excitations["bbb"],
                                                         LH.aaa, LH.aab, LH.abb, LH.bbb,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)

    if flag_RHF:
        L.b = L.a.copy()
        LH.b = LH.a.copy()

        L.bb = L.aa.copy()
        LH.bb = LH.aa.copy()

        L.abb = L.aab.copy()
        LH.abb = LH.aab.copy()
        l3_excitations["abb"] = l3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]

        L.bbb = L.aaa.copy()
        LH.bbb = LH.aaa.copy()
        l3_excitations["bbb"] = l3_excitations["aaa"].copy()

    return L, LH

def update_l(L, omega, H, RHF_symmetry, system, l3_excitations):
    L.a, L.b, L.aa, L.ab, L.bb, L.aaa, L.aab, L.abb, L.bbb = eomccsdt_p_loops.update_r(
        L.a,
        L.b,
        L.aa,
        L.ab,
        L.bb,
        L.aaa,
        l3_excitations["aaa"],
        L.aab,
        l3_excitations["aab"],
        L.abb,
        l3_excitations["abb"],
        L.bbb,
        l3_excitations["bbb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    if RHF_symmetry:
        L.b = L.a.copy()
        L.bb = L.aa.copy()
        L.abb = L.aab.copy()
        L.bbb = L.aaa.copy()
    return L

def LH_fun(LH, L, T, H, flag_RHF, system, t3_excitations, l3_excitations):
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    if np.array_equal(l3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(l3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(l3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["abb"] = False
    if np.array_equal(l3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["bbb"] = False

    # get LT intermediates
    X = build_left_ccsdt_p_intermediates(L, l3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=flag_RHF)

    # build L1
    LH = build_LH_1A(L, LH, T, H, X)
    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, H, X)

    # build L2
    LH = build_LH_2A(L, LH, T, H, X, l3_excitations)
    LH = build_LH_2B(L, LH, T, H, X, l3_excitations)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H, X, l3_excitations)

    # build L3
    if do_l3["aaa"]:
        LH, L, l3_excitations = build_LH_3A(L, LH, H, X, l3_excitations)
    if do_l3["aab"]:
        LH, L, l3_excitations = build_LH_3B(L, LH, H, X, l3_excitations)
    if flag_RHF:
        L.abb = L.aab.copy()
        LH.abb = LH.aab.copy()
        l3_excitations["abb"] = l3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]

        L.bbb = L.aaa.copy()
        LH.bbb = LH.aaa.copy()
        l3_excitations["bbb"] = l3_excitations["aaa"].copy()
    else:
        if do_l3["abb"]:
            LH, L, l3_excitations = build_LH_3C(L, LH, H, X, l3_excitations)
        if do_l3["bbb"]:
            LH, L, l3_excitations = build_LH_3D(L, LH, H, X, l3_excitations)

    return LH.flatten()

def build_LH_1A(L, LH, T, H, X):

    LH.a = np.einsum("ea,ei->ai", H.a.vv, L.a, optimize=True)
    LH.a -= np.einsum("im,am->ai", H.a.oo, L.a, optimize=True)
    LH.a += np.einsum("eima,em->ai", H.aa.voov, L.a, optimize=True)
    LH.a += np.einsum("ieam,em->ai", H.ab.ovvo, L.b, optimize=True)
    ## These terms contain T3 in them from the 2-body HBar components used here ##
    LH.a += 0.5 * np.einsum("fena,efin->ai", H.aa.vvov, L.aa, optimize=True)
    LH.a += np.einsum("efan,efin->ai", H.ab.vvvo, L.ab, optimize=True)
    LH.a -= 0.5 * np.einsum("finm,afmn->ai", H.aa.vooo, L.aa, optimize=True)
    LH.a -= np.einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab, optimize=True)
    I1 = 0.25 * np.einsum("efmn,fgnm->ge", L.aa, T.aa, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", L.aa, T.aa, optimize=True)
    I3 = -0.25 * np.einsum("efmo,efno->mn", L.aa, T.aa, optimize=True)
    I4 = 0.25 * np.einsum("efmo,efnm->on", L.aa, T.aa, optimize=True)
    LH.a += np.einsum("ge,eiga->ai", I1, H.aa.vovv, optimize=True)
    LH.a += np.einsum("gf,figa->ai", I2, H.aa.vovv, optimize=True)
    LH.a += np.einsum("mn,nima->ai", I3, H.aa.ooov, optimize=True)
    LH.a += np.einsum("on,nioa->ai", I4, H.aa.ooov, optimize=True)
    I1 = -np.einsum("abij,abin->jn", L.ab, T.ab, optimize=True)
    I2 = np.einsum("abij,afij->fb", L.ab, T.ab, optimize=True)
    I3 = np.einsum("abij,fbij->fa", L.ab, T.ab, optimize=True)
    I4 = -np.einsum("abij,abnj->in", L.ab, T.ab, optimize=True)
    LH.a += np.einsum("jn,mnej->em", I1, H.ab.oovo, optimize=True)
    LH.a += np.einsum("fb,mbef->em", I2, H.ab.ovvv, optimize=True)
    LH.a += np.einsum("fa,amfe->em", I3, H.aa.vovv, optimize=True)
    LH.a += np.einsum("in,nmie->em", I4, H.aa.ooov, optimize=True)
    I1 = 0.25 * np.einsum("abij,fbij->fa", L.bb, T.bb, optimize=True)
    I2 = -0.25 * np.einsum("abij,faij->fb", L.bb, T.bb, optimize=True)
    I3 = -0.25 * np.einsum("abij,abnj->in", L.bb, T.bb, optimize=True)
    I4 = 0.25 * np.einsum("abij,abni->jn", L.bb, T.bb, optimize=True)
    LH.a += np.einsum("fa,maef->em", I1, H.ab.ovvv, optimize=True)
    LH.a += np.einsum("fb,mbef->em", I2, H.ab.ovvv, optimize=True)
    LH.a += np.einsum("in,mnei->em", I3, H.ab.oovo, optimize=True)
    LH.a += np.einsum("jn,mnej->em", I4, H.ab.oovo, optimize=True)
    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH.a += np.einsum("em,imae->ai", X.a.vo, H.aa.oovv, optimize=True)
    LH.a += np.einsum("em,imae->ai", X.b.vo, H.ab.oovv, optimize=True)
    # 4-body Hbar
    I1A_vo = (
          -0.5 * np.einsum("nomg,egno->em", X.aa.ooov, T.aa, optimize=True)
          - np.einsum("nomg,egno->em", X.ab.ooov, T.ab, optimize=True)
    )
    I1B_vo = (
          -0.5 * np.einsum("nomg,egno->em", X.bb.ooov, T.bb, optimize=True)
          - np.einsum("ongm,geon->em", X.ab.oovo, T.ab, optimize=True)
    )
    LH.a += np.einsum("em,imae->ai", I1A_vo, H.aa.oovv, optimize=True)
    LH.a += np.einsum("em,imae->ai", I1B_vo, H.ab.oovv, optimize=True)
    LH.a -= np.einsum("nm,mina->ai", X.a.oo, H.aa.ooov, optimize=True)
    LH.a -= np.einsum("nm,iman->ai", X.b.oo, H.ab.oovo, optimize=True)
    LH.a -= np.einsum("ef,fiea->ai", X.a.vv, H.aa.vovv, optimize=True)
    LH.a -= np.einsum("ef,ifae->ai", X.b.vv, H.ab.ovvv, optimize=True)
    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
    LH.a += np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    LH.a -= np.einsum("ma,im->ai", H.a.ov, X.a.oo, optimize=True)
    LH.a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    LH.a += np.einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo, optimize=True)
    LH.a += np.einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov, optimize=True)
    LH.a += np.einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo, optimize=True)
    LH.a += np.einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov, optimize=True)
    LH.a -= 0.5 * np.einsum("gife,efag->ai", X.aa.vovv, H.aa.vvvv, optimize=True)
    LH.a -= np.einsum("igef,efag->ai", X.ab.ovvv, H.ab.vvvv, optimize=True)
    LH.a -= np.einsum("imne,enma->ai", X.aa.ooov, H.aa.voov, optimize=True)
    LH.a -= np.einsum("imne,neam->ai", X.ab.ooov, H.ab.ovvo, optimize=True)
    LH.a -= np.einsum("imen,enam->ai", X.ab.oovo, H.ab.vovo, optimize=True)
    LH.a += 0.5 * np.einsum("nmoa,iomn->ai", H.aa.ooov, X.aa.oooo, optimize=True)
    LH.a += np.einsum("mnao,iomn->ai", H.ab.oovo, X.ab.oooo, optimize=True)
    LH.a += np.einsum("fmae,eimf->ai", H.aa.vovv, X.aa.voov, optimize=True)
    LH.a += np.einsum("fmae,iefm->ai", H.ab.vovv, X.ab.ovvo, optimize=True)
    LH.a += np.einsum("mfae,iemf->ai", H.ab.ovvv, X.ab.ovov, optimize=True)
    LH.a -= 0.5 * np.einsum("gife,efag->ai", H.aa.vovv, X.aa.vvvv, optimize=True)
    LH.a -= np.einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv, optimize=True)
    LH.a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    LH.a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    LH.a -= np.einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo, optimize=True)
    return LH

def build_LH_1B(L, LH, T, H, X):
    LH.b = np.einsum("ea,ei->ai", H.b.vv, L.b, optimize=True)
    LH.b -= np.einsum("im,am->ai", H.b.oo, L.b, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.bb.voov, L.b, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.ab.voov, L.a, optimize=True)
    ## These terms contain T3 in them from the 2-body HBar components used here ##
    LH.b += 0.5 * np.einsum("fena,efin->ai", H.bb.vvov, L.bb, optimize=True)
    LH.b += np.einsum("fena,feni->ai", H.ab.vvov, L.ab, optimize=True)
    LH.b -= 0.5 * np.einsum("finm,afmn->ai", H.bb.vooo, L.bb, optimize=True)
    LH.b -= np.einsum("finm,fanm->ai", H.ab.vooo, L.ab, optimize=True)
    I1 = 0.25 * np.einsum("efmn,fgnm->ge", L.bb, T.bb, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", L.bb, T.bb, optimize=True)
    I3 = -0.25 * np.einsum("efmo,efno->mn", L.bb, T.bb, optimize=True)
    I4 = 0.25 * np.einsum("efmo,efnm->on", L.bb, T.bb, optimize=True)
    LH.b += np.einsum("ge,eiga->ai", I1, H.bb.vovv, optimize=True)
    LH.b += np.einsum("gf,figa->ai", I2, H.bb.vovv, optimize=True)
    LH.b += np.einsum("mn,nima->ai", I3, H.bb.ooov, optimize=True)
    LH.b += np.einsum("on,nioa->ai", I4, H.bb.ooov, optimize=True)
    I1 = -np.einsum("baji,bani->jn", L.ab, T.ab, optimize=True)
    I2 = np.einsum("baji,faji->fb", L.ab, T.ab, optimize=True)
    I3 = np.einsum("baji,bfji->fa", L.ab, T.ab, optimize=True)
    I4 = -np.einsum("baji,bajn->in", L.ab, T.ab, optimize=True)
    LH.b += np.einsum("jn,nmje->em", I1, H.ab.ooov, optimize=True)
    LH.b += np.einsum("fb,bmfe->em", I2, H.ab.vovv, optimize=True)
    LH.b += np.einsum("fa,amfe->em", I3, H.bb.vovv, optimize=True)
    LH.b += np.einsum("in,nmie->em", I4, H.bb.ooov, optimize=True)
    I1 = 0.25 * np.einsum("abij,fbij->fa", L.aa, T.aa, optimize=True)
    I2 = -0.25 * np.einsum("abij,faij->fb", L.aa, T.aa, optimize=True)
    I3 = -0.25 * np.einsum("abij,abnj->in", L.aa, T.aa, optimize=True)
    I4 = 0.25 * np.einsum("abij,abni->jn", L.aa, T.aa, optimize=True)
    LH.b += np.einsum("fa,amfe->em", I1, H.ab.vovv, optimize=True)
    LH.b += np.einsum("fb,bmfe->em", I2, H.ab.vovv, optimize=True)
    LH.b += np.einsum("in,nmie->em", I3, H.ab.ooov, optimize=True)
    LH.b += np.einsum("jn,nmje->em", I4, H.ab.ooov, optimize=True)
    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH.b += np.einsum("em,imae->ai", X.b.vo, H.bb.oovv, optimize=True)
    LH.b += np.einsum("em,miea->ai", X.a.vo, H.ab.oovv, optimize=True)
    # 4-body Hbar
    I1B_vo = (
            -0.5 * np.einsum("nomg,egno->em", X.bb.ooov, T.bb, optimize=True)
            - np.einsum("ongm,geon->em", X.ab.oovo, T.ab, optimize=True)
    )
    I1A_vo = (
            -0.5 * np.einsum("nomg,egno->em", X.aa.ooov, T.aa, optimize=True)
            - np.einsum("nomg,egno->em", X.ab.ooov, T.ab, optimize=True)
    )
    LH.b += np.einsum("em,imae->ai", I1B_vo, H.bb.oovv, optimize=True)
    LH.b += np.einsum("em,miea->ai", I1A_vo, H.ab.oovv, optimize=True)
    LH.b -= np.einsum("nm,mina->ai", X.b.oo, H.bb.ooov, optimize=True)
    LH.b -= np.einsum("nm,mina->ai", X.a.oo, H.ab.ooov, optimize=True)
    LH.b -= np.einsum("ef,fiea->ai", X.b.vv, H.bb.vovv, optimize=True)
    LH.b -= np.einsum("ef,fiea->ai", X.a.vv, H.ab.vovv, optimize=True)
    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
    LH.b += np.einsum("ie,ea->ai", H.b.ov, X.b.vv, optimize=True)
    LH.b -= np.einsum("ma,im->ai", H.b.ov, X.b.oo, optimize=True)
    LH.b += 0.5 * np.einsum("nmoa,iomn->ai", X.bb.ooov, H.bb.oooo, optimize=True)
    LH.b += np.einsum("nmoa,oinm->ai", X.ab.ooov, H.ab.oooo, optimize=True)
    LH.b += np.einsum("fmae,eimf->ai", X.bb.vovv, H.bb.voov, optimize=True)
    LH.b += np.einsum("mfea,eimf->ai", X.ab.ovvv, H.ab.voov, optimize=True)
    LH.b += np.einsum("fmea,eifm->ai", X.ab.vovv, H.ab.vovo, optimize=True)
    LH.b -= 0.5 * np.einsum("gife,efag->ai", X.bb.vovv, H.bb.vvvv, optimize=True)
    LH.b -= np.einsum("gife,fega->ai", X.ab.vovv, H.ab.vvvv, optimize=True)
    LH.b -= np.einsum("imne,enma->ai", X.bb.ooov, H.bb.voov, optimize=True)
    LH.b -= np.einsum("mien,enma->ai", X.ab.oovo, H.ab.voov, optimize=True)
    LH.b -= np.einsum("mine,nema->ai", X.ab.ooov, H.ab.ovov, optimize=True)
    LH.b += 0.5 * np.einsum("nmoa,iomn->ai", H.bb.ooov, X.bb.oooo, optimize=True)
    LH.b += np.einsum("nmoa,oinm->ai", H.ab.ooov, X.ab.oooo, optimize=True)
    LH.b += np.einsum("fmae,eimf->ai", H.bb.vovv, X.bb.voov, optimize=True)
    LH.b += np.einsum("mfea,eimf->ai", H.ab.ovvv, X.ab.voov, optimize=True)
    LH.b += np.einsum("fmea,eifm->ai", H.ab.vovv, X.ab.vovo, optimize=True)
    LH.b -= 0.5 * np.einsum("gife,efag->ai", H.bb.vovv, X.bb.vvvv, optimize=True)
    LH.b -= np.einsum("gife,fega->ai", H.ab.vovv, X.ab.vvvv, optimize=True)
    LH.b -= np.einsum("imne,enma->ai", H.bb.ooov, X.bb.voov, optimize=True)
    LH.b -= np.einsum("mien,enma->ai", H.ab.oovo, X.ab.voov, optimize=True)
    LH.b -= np.einsum("mine,nema->ai", H.ab.ooov, X.ab.ovov, optimize=True)
    return LH

def build_LH_2A(L, LH, T, H, X, l3_excitations):

    LH.aa = 0.5 * np.einsum("ea,ebij->abij", H.a.vv, L.aa, optimize=True)
    LH.aa -= 0.5 * np.einsum("im,abmj->abij", H.a.oo, L.aa, optimize=True)
    LH.aa += np.einsum("jb,ai->abij", H.a.ov, L.a, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", L.aa, T.aa, optimize=True)
          - np.einsum("afmn,efmn->ea", L.ab, T.ab, optimize=True)
    )
    LH.aa += 0.5 * np.einsum("ea,ijeb->abij", I1, H.aa.oovv, optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", L.aa, T.aa, optimize=True)
          + np.einsum("efin,efmn->im", L.ab, T.ab, optimize=True)
    )
    LH.aa -= 0.5 * np.einsum("im,mjab->abij", I1, H.aa.oovv, optimize=True)
    LH.aa += np.einsum("eima,ebmj->abij", H.aa.voov, L.aa, optimize=True)
    LH.aa += np.einsum("ieam,bejm->abij", H.ab.ovvo, L.ab, optimize=True)
    LH.aa += 0.125 * np.einsum("ijmn,abmn->abij", H.aa.oooo, L.aa, optimize=True)
    LH.aa += 0.125 * np.einsum("efab,efij->abij", H.aa.vvvv, L.aa, optimize=True)
    LH.aa += 0.5 * np.einsum("ejab,ei->abij", H.aa.vovv, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ijmb,am->abij", H.aa.ooov, L.a, optimize=True)
    # < 0 | L3 * H(2) | ijab >
    LH.aa -= np.einsum("ejfb,fiea->abij", X.aa.vovv, H.aa.vovv, optimize=True) # 1
    LH.aa -= np.einsum("njmb,mina->abij", X.aa.ooov, H.aa.ooov, optimize=True) # 2
    LH.aa -= 0.25 * np.einsum("enab,jine->abij", X.aa.vovv, H.aa.ooov, optimize=True) # 3
    LH.aa -= 0.25 * np.einsum("jine,enab->abij", X.aa.ooov, H.aa.vovv, optimize=True) # 4
    LH.aa -= np.einsum("jebf,ifae->abij", X.ab.ovvv, H.ab.ovvv, optimize=True) # 5
    LH.aa -= np.einsum("jnbm,iman->abij", X.ab.oovo, H.ab.oovo, optimize=True) # 6
    # < 0 | L3 * (H(2) * T3) | ijab >
    LH.aa += np.einsum("ejmb,imae->abij", X.aa.voov, H.aa.oovv, optimize=True) # 1
    LH.aa += np.einsum("jebm,imae->abij", X.ab.ovvo, H.ab.oovv, optimize=True) # 2
    LH.aa += 0.125 * np.einsum("efab,ijef->abij", X.aa.vvvv, H.aa.oovv, optimize=True) # 3
    LH.aa += 0.125 * np.einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv, optimize=True) # 4
    # 4-body HBar
    LH.aa += 0.5 * np.einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv, optimize=True) # 1
    LH.aa -= 0.5 * np.einsum("im,jmba->abij", X.a.oo, H.aa.oovv, optimize=True) # 2
    # Moment-like terms
    LH.aa = leftccsdt_p_loops.build_lh_2a(
                                            LH.aa,
                                            L.aaa, l3_excitations["aaa"],
                                            L.aab, l3_excitations["aab"],
                                            H.aa.vooo, H.aa.vvov, H.ab.ovoo, H.ab.vvvo,
    )
    return LH

def build_LH_2B(L, LH, T, H, X, l3_excitations):

    LH.ab = -np.einsum("ijmb,am->abij", H.ab.ooov, L.a, optimize=True)
    LH.ab -= np.einsum("ijam,bm->abij", H.ab.oovo, L.b, optimize=True)
    LH.ab += np.einsum("ejab,ei->abij", H.ab.vovv, L.a, optimize=True)
    LH.ab += np.einsum("ieab,ej->abij", H.ab.ovvv, L.b, optimize=True)
    LH.ab += np.einsum("ijmn,abmn->abij", H.ab.oooo, L.ab, optimize=True)
    LH.ab += np.einsum("efab,efij->abij", H.ab.vvvv, L.ab, optimize=True)
    LH.ab += np.einsum("ejmb,aeim->abij", H.ab.voov, L.aa, optimize=True)
    LH.ab += np.einsum("eima,ebmj->abij", H.aa.voov, L.ab, optimize=True)
    LH.ab += np.einsum("ejmb,aeim->abij", H.bb.voov, L.ab, optimize=True)
    LH.ab += np.einsum("ieam,ebmj->abij", H.ab.ovvo, L.bb, optimize=True)
    LH.ab -= np.einsum("iemb,aemj->abij", H.ab.ovov, L.ab, optimize=True)
    LH.ab -= np.einsum("ejam,ebim->abij", H.ab.vovo, L.ab, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", L.aa, T.aa, optimize=True)
          - np.einsum("afmn,efmn->ea", L.ab, T.ab, optimize=True)
    )
    LH.ab += np.einsum("ea,ijeb->abij", I1, H.ab.oovv, optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", L.aa, T.aa, optimize=True)
          + np.einsum("efin,efmn->im", L.ab, T.ab, optimize=True)
    )
    LH.ab -= np.einsum("im,mjab->abij", I1, H.ab.oovv, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", L.bb, T.bb, optimize=True)
          - np.einsum("fanm,fenm->ea", L.ab, T.ab, optimize=True)
    )
    LH.ab += np.einsum("ea,jibe->baji", I1, H.ab.oovv, optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", L.bb, T.bb, optimize=True)
          + np.einsum("feni,fenm->im", L.ab, T.ab, optimize=True)
    )
    LH.ab -= np.einsum("im,jmba->baji", I1, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("ea,ebij->abij", H.a.vv, L.ab, optimize=True)
    LH.ab += np.einsum("eb,aeij->abij", H.b.vv, L.ab, optimize=True)
    LH.ab -= np.einsum("im,abmj->abij", H.a.oo, L.ab, optimize=True)
    LH.ab -= np.einsum("jm,abim->abij", H.b.oo, L.ab, optimize=True)
    LH.ab += np.einsum("jb,ai->abij", H.b.ov, L.a, optimize=True)
    LH.ab += np.einsum("ia,bj->abij", H.a.ov, L.b, optimize=True)
    # < 0 | L3 * H(2) | ij~ab~ >
    LH.ab -= np.einsum("ejfb,fiea->abij", X.ab.vovv, H.aa.vovv, optimize=True) # 1
    LH.ab -= np.einsum("ejfb,ifae->abij", X.bb.vovv, H.ab.ovvv, optimize=True) # 2
    LH.ab -= np.einsum("eifa,fjeb->abij", X.aa.vovv, H.ab.vovv, optimize=True) # 3
    LH.ab -= np.einsum("ieaf,fjeb->abij", X.ab.ovvv, H.bb.vovv, optimize=True) # 4
    LH.ab -= np.einsum("njmb,mina->abij", X.ab.ooov, H.aa.ooov, optimize=True) # 5
    LH.ab -= np.einsum("njmb,iman->abij", X.bb.ooov, H.ab.oovo, optimize=True) # 6
    LH.ab -= np.einsum("nima,mjnb->abij", X.aa.ooov, H.ab.ooov, optimize=True) # 7
    LH.ab -= np.einsum("inam,mjnb->abij", X.ab.oovo, H.bb.ooov, optimize=True) # 8
    LH.ab += np.einsum("inmb,mjan->abij", X.ab.ooov, H.ab.oovo, optimize=True) # 9
    LH.ab += np.einsum("ifeb,ejaf->abij", X.ab.ovvv, H.ab.vovv, optimize=True) # 10
    LH.ab += np.einsum("ejaf,ifeb->abij", X.ab.vovv, H.ab.ovvv, optimize=True) # 11
    LH.ab += np.einsum("mjan,inmb->abij", X.ab.oovo, H.ab.ooov, optimize=True) # 12
    LH.ab -= np.einsum("enab,ijen->abij", X.ab.vovv, H.ab.oovo, optimize=True) # 13
    LH.ab -= np.einsum("mfab,ijmf->abij", X.ab.ovvv, H.ab.ooov, optimize=True) # 14
    LH.ab -= np.einsum("ijmf,mfab->abij", X.ab.ooov, H.ab.ovvv, optimize=True) # 15
    LH.ab -= np.einsum("ijen,enab->abij", X.ab.oovo, H.ab.vovv, optimize=True) # 16
    # < 0 | L3 * (H(2) * T3)_C | ij~ab~ >
    LH.ab += (
               np.einsum("ejmb,miea->abij", X.ab.voov, H.aa.oovv, optimize=True)
               + np.einsum("ejmb,imae->abij", X.bb.voov, H.ab.oovv, optimize=True)
    ) # 1
    LH.ab += (
               np.einsum("eima,mjeb->abij", X.aa.voov, H.ab.oovv, optimize=True)
               + np.einsum("ieam,mjeb->abij", X.ab.ovvo, H.bb.oovv, optimize=True)
    ) # 2
    LH.ab -= np.einsum("iemb,mjae->abij", X.ab.ovov, H.ab.oovv, optimize=True) # 3
    LH.ab -= np.einsum("ejam,imeb->abij", X.ab.vovo, H.ab.oovv, optimize=True) # 4
    LH.ab += np.einsum("efab,ijef->abij", X.ab.vvvv, H.ab.oovv, optimize=True) # 5
    LH.ab += np.einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv, optimize=True) # 6
    # 4-body HBar
    LH.ab += np.einsum("ea,ijeb->abij", X.a.vv, H.ab.oovv, optimize=True) # 1
    LH.ab += np.einsum("eb,ijae->abij", X.b.vv, H.ab.oovv, optimize=True) # 2
    LH.ab -= np.einsum("im,mjab->abij", X.a.oo, H.ab.oovv, optimize=True) # 3
    LH.ab -= np.einsum("jm,imab->abij", X.b.oo, H.ab.oovv, optimize=True) # 4
    # Moment-like terms
    LH.ab = leftccsdt_p_loops.build_lh_2b(
                                            LH.ab,
                                            L.aab, l3_excitations["aab"],
                                            L.abb, l3_excitations["abb"],
                                            H.aa.vooo, H.aa.vvov, 
                                            H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo,
                                            H.bb.vooo, H.bb.vvov,
    )
    return LH

def build_LH_2C(L, LH, T, H, X, l3_excitations):

    LH.bb = 0.5 * np.einsum("ea,ebij->abij", H.b.vv, L.bb, optimize=True)
    LH.bb -= 0.5 * np.einsum("im,abmj->abij", H.b.oo, L.bb, optimize=True)
    LH.bb += np.einsum("jb,ai->abij", H.b.ov, L.b, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", L.bb, T.bb, optimize=True)
          - np.einsum("fanm,fenm->ea", L.ab, T.ab, optimize=True)
    )
    LH.bb += 0.5 * np.einsum("ea,ijeb->abij", I1, H.bb.oovv, optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", L.bb, T.bb, optimize=True)
          + np.einsum("feni,fenm->im", L.ab, T.ab, optimize=True)
    )
    LH.bb -= 0.5 * np.einsum("im,mjab->abij", I1, H.bb.oovv, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb += 0.125 * np.einsum("ijmn,abmn->abij", H.bb.oooo, L.bb, optimize=True)
    LH.bb += 0.125 * np.einsum("efab,efij->abij", H.bb.vvvv, L.bb, optimize=True)
    LH.bb += 0.5 * np.einsum("ejab,ei->abij", H.bb.vovv, L.b, optimize=True)
    LH.bb -= 0.5 * np.einsum("ijmb,am->abij", H.bb.ooov, L.b, optimize=True)
    # < 0 | L3 * H(2) | ijab >
    LH.bb -= np.einsum("ejfb,fiea->abij", X.bb.vovv, H.bb.vovv, optimize=True) # 1
    LH.bb -= np.einsum("njmb,mina->abij", X.bb.ooov, H.bb.ooov, optimize=True) # 2
    LH.bb -= 0.25 * np.einsum("enab,jine->abij", X.bb.vovv, H.bb.ooov, optimize=True) # 3
    LH.bb -= 0.25 * np.einsum("jine,enab->abij", X.bb.ooov, H.bb.vovv, optimize=True) # 4
    LH.bb -= np.einsum("ejfb,fiea->abij", X.ab.vovv, H.ab.vovv, optimize=True) # 5
    LH.bb -= np.einsum("njmb,mina->abij", X.ab.ooov, H.ab.ooov, optimize=True) # 6
    # < 0 | L3 * (H(2) * T3) | ijab >
    LH.bb += np.einsum("ejmb,imae->abij", X.bb.voov, H.bb.oovv, optimize=True) # 1
    LH.bb += np.einsum("ejmb,miea->abij", X.ab.voov, H.ab.oovv, optimize=True) # 2
    LH.bb += 0.125 * np.einsum("efab,ijef->abij", X.bb.vvvv, H.bb.oovv, optimize=True) # 3
    LH.bb += 0.125 * np.einsum("ijmn,mnab->abij", X.bb.oooo, H.bb.oovv, optimize=True) # 4
    # 4-body HBar
    LH.bb += 0.5 * np.einsum("ea,ijeb->abij", X.b.vv, H.bb.oovv, optimize=True) # 1
    LH.bb -= 0.5 * np.einsum("im,jmba->abij", X.b.oo, H.bb.oovv, optimize=True) # 2
    # Moment-like terms
    LH.bb = leftccsdt_p_loops.build_lh_2c(
                                            LH.bb,
                                            L.abb, l3_excitations["abb"],
                                            L.bbb, l3_excitations["bbb"],
                                            H.bb.vooo, H.bb.vvov, H.ab.vooo, H.ab.vvov,
    )
    return LH

def build_LH_3A(L, LH, H, X, l3_excitations):
    LH.aaa, L.aaa, l3_excitations["aaa"] = leftccsdt_p_loops.build_lh_3a(
                                            L.a, L.aa,
                                            L.aaa, l3_excitations["aaa"],
                                            L.aab, l3_excitations["aab"],
                                            H.a.ov, H.a.oo, H.a.vv,
                                            H.aa.oooo, H.aa.ooov, H.aa.oovv,
                                            H.aa.voov, H.aa.vovv, H.aa.vvvv,
                                            H.ab.ovvo,
                                            X.aa.ooov, X.aa.vovv,
    )
    return LH, L, l3_excitations

def build_LH_3B(L, LH, H, X, l3_excitations):
    LH.aab, L.aab, l3_excitations["aab"] = leftccsdt_p_loops.build_lh_3b(
                                            L.a, L.b, L.aa, L.ab,
                                            L.aaa, l3_excitations["aaa"],
                                            L.aab, l3_excitations["aab"],
                                            L.abb, l3_excitations["abb"],
                                            H.a.ov, H.a.oo, H.a.vv,
                                            H.b.ov, H.b.oo, H.b.vv,
                                            H.aa.oooo, H.aa.ooov, H.aa.oovv,
                                            H.aa.voov, H.aa.vovv, H.aa.vvvv,
                                            H.ab.oooo, H.ab.ooov, H.ab.oovo,
                                            H.ab.oovv,
                                            H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                                            H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                                            H.bb.voov,
                                            X.aa.ooov, X.aa.vovv,
                                            X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv
    )
    return LH, L, l3_excitations

def build_LH_3C(L, LH, H, X, l3_excitations):
    LH.abb, L.abb, l3_excitations["abb"] = leftccsdt_p_loops.build_lh_3c(
                                            L.a, L.b, L.ab, L.bb,
                                            L.aab, l3_excitations["aab"],
                                            L.abb, l3_excitations["abb"],
                                            L.bbb, l3_excitations["bbb"],
                                            H.a.ov, H.a.oo, H.a.vv,
                                            H.b.ov, H.b.oo, H.b.vv,
                                            H.aa.voov,
                                            H.ab.oooo, H.ab.ooov, H.ab.oovo,
                                            H.ab.oovv,
                                            H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                                            H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                                            H.bb.oooo, H.bb.ooov, H.bb.oovv,
                                            H.bb.voov, H.bb.vovv, H.bb.vvvv,
                                            X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
                                            X.bb.ooov, X.bb.vovv,
    )
    return LH, L, l3_excitations

def build_LH_3D(L, LH, H, X, l3_excitations):
    LH.bbb, L.bbb, l3_excitations["bbb"] = leftccsdt_p_loops.build_lh_3d(
                                            L.b, L.bb,
                                            L.abb, l3_excitations["abb"],
                                            L.bbb, l3_excitations["bbb"],
                                            H.b.ov, H.b.oo, H.b.vv,
                                            H.ab.voov,
                                            H.bb.oooo, H.bb.ooov, H.bb.oovv,
                                            H.bb.voov, H.bb.vovv, H.bb.vvvv,
                                            X.bb.ooov, X.bb.vovv,
    )
    return LH, L, l3_excitations
