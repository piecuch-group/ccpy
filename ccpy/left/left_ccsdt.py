import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.lib.core import cc_loops2
from ccpy.left.left_cc_intermediates import build_left_ccsdt_intermediates

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

    # get LT intermediates
    X = build_left_ccsdt_intermediates(L, T, system)

    # build L1
    LH = build_LH_1A(L, LH, T, H, X)

    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, H, X)

    # build L2
    LH = build_LH_2A(L, LH, T, H, X)
    LH = build_LH_2B(L, LH, T, H, X)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H, X)

    # build L3
    LH = build_LH_3A(L, LH, H, X)
    LH = build_LH_3B(L, LH, H, X)
    if flag_RHF:
        LH.abb = np.transpose(LH.aab, (2, 0, 1, 5, 3, 4))
        LH.bbb = LH.aaa.copy()
    else:
        LH = build_LH_3C(L, LH, H, X)
        LH = build_LH_3D(L, LH, H, X)

    # Add Hamiltonian if ground-state calculation
    if is_ground:
        LH.a += np.transpose(H.a.ov, (1, 0))
        LH.b += np.transpose(H.b.ov, (1, 0))
        LH.aa += np.transpose(H.aa.oovv, (2, 3, 0, 1))
        LH.ab += np.transpose(H.ab.oovv, (2, 3, 0, 1))
        LH.bb += np.transpose(H.bb.oovv, (2, 3, 0, 1))


    L.a, L.b, LH.a, LH.b = cc_loops2.update_l1(L.a, L.b, LH.a, LH.b,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)
    L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb = cc_loops2.update_l2(L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)
    L.aaa, L.aab, L.abb, L.bbb, LH.aaa, LH.aab, LH.abb, LH.bbb = cc_loops2.update_l3(L.aaa, L.aab, L.abb, L.bbb, LH.aaa, LH.aab, LH.abb, LH.bbb,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)

    if flag_RHF:
        L.b = L.a.copy()
        L.bb = L.aa.copy()
        L.bbb = L.aaa.copy()
        L.abb = np.transpose(L.aab, (2, 0, 1, 5, 3, 4))
        L.bbb = L.aaa.copy()

        LH.b = LH.a.copy()
        LH.bb = LH.aa.copy()
        LH.bbb = LH.aaa.copy()
        LH.abb = np.transpose(LH.aab, (2, 0, 1, 5, 3, 4))
        LH.bbb = LH.aaa.copy()

    return L, LH

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.b, L.aa, L.ab, L.bb, L.aaa, L.aab, L.abb, L.bbb = cc_loops2.update_r_ccsdt(
        L.a,
        L.b,
        L.aa,
        L.ab,
        L.bb,
        L.aaa,
        L.aab,
        L.abb,
        L.bbb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    if RHF_symmetry:
        L.b = L.a.copy()
        L.bb = L.aa.copy()
        L.abb = L.aab.transpose((2, 1, 0, 5, 4, 3))
        L.bbb = L.aaa.copy()
    return L

def LH_fun(LH, L, T, H, flag_RHF, system):
    # get LT intermediates
    X = build_left_ccsdt_intermediates(L, T, system)
    # build L1
    LH = build_LH_1A(L, LH, T, H, X)
    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, H, X)
    # build L2
    LH = build_LH_2A(L, LH, T, H, X)
    LH = build_LH_2B(L, LH, T, H, X)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H, X)
    # build L3
    LH = build_LH_3A(L, LH, H, X)
    LH = build_LH_3B(L, LH, H, X)
    if flag_RHF:
        LH.abb = np.transpose(LH.aab, (2, 0, 1, 5, 3, 4))
        LH.bbb = LH.aaa.copy()
    else:
        LH = build_LH_3C(L, LH, H, X)
        LH = build_LH_3D(L, LH, H, X)
    return LH.flatten()

def build_LH_1A(L, LH, T, H, X):

    LH.a = ccpy_einsum("ea,ei->ai", H.a.vv, L.a)
    LH.a -= ccpy_einsum("im,am->ai", H.a.oo, L.a)
    LH.a += ccpy_einsum("eima,em->ai", H.aa.voov, L.a)
    LH.a += ccpy_einsum("ieam,em->ai", H.ab.ovvo, L.b)

    ## These terms contain T3 in them from the 2-body HBar components used here ##
    LH.a += 0.5 * ccpy_einsum("fena,efin->ai", H.aa.vvov, L.aa)
    LH.a += ccpy_einsum("efan,efin->ai", H.ab.vvvo, L.ab)
    LH.a -= 0.5 * ccpy_einsum("finm,afmn->ai", H.aa.vooo, L.aa)
    LH.a -= ccpy_einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab)

    I1 = 0.25 * ccpy_einsum("efmn,fgnm->ge", L.aa, T.aa)
    I2 = -0.25 * ccpy_einsum("efmn,egnm->gf", L.aa, T.aa)
    I3 = -0.25 * ccpy_einsum("efmo,efno->mn", L.aa, T.aa)
    I4 = 0.25 * ccpy_einsum("efmo,efnm->on", L.aa, T.aa)
    LH.a += ccpy_einsum("ge,eiga->ai", I1, H.aa.vovv)
    LH.a += ccpy_einsum("gf,figa->ai", I2, H.aa.vovv)
    LH.a += ccpy_einsum("mn,nima->ai", I3, H.aa.ooov)
    LH.a += ccpy_einsum("on,nioa->ai", I4, H.aa.ooov)

    I1 = -ccpy_einsum("abij,abin->jn", L.ab, T.ab)
    I2 = ccpy_einsum("abij,afij->fb", L.ab, T.ab)
    I3 = ccpy_einsum("abij,fbij->fa", L.ab, T.ab)
    I4 = -ccpy_einsum("abij,abnj->in", L.ab, T.ab)
    LH.a += ccpy_einsum("jn,mnej->em", I1, H.ab.oovo)
    LH.a += ccpy_einsum("fb,mbef->em", I2, H.ab.ovvv)
    LH.a += ccpy_einsum("fa,amfe->em", I3, H.aa.vovv)
    LH.a += ccpy_einsum("in,nmie->em", I4, H.aa.ooov)

    I1 = 0.25 * ccpy_einsum("abij,fbij->fa", L.bb, T.bb)
    I2 = -0.25 * ccpy_einsum("abij,faij->fb", L.bb, T.bb)
    I3 = -0.25 * ccpy_einsum("abij,abnj->in", L.bb, T.bb)
    I4 = 0.25 * ccpy_einsum("abij,abni->jn", L.bb, T.bb)
    LH.a += ccpy_einsum("fa,maef->em", I1, H.ab.ovvv)
    LH.a += ccpy_einsum("fb,mbef->em", I2, H.ab.ovvv)
    LH.a += ccpy_einsum("in,mnei->em", I3, H.ab.oovo)
    LH.a += ccpy_einsum("jn,mnej->em", I4, H.ab.oovo)

    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH.a += ccpy_einsum("em,imae->ai", X.a.vo, H.aa.oovv)
    LH.a += ccpy_einsum("em,imae->ai", X.b.vo, H.ab.oovv)

    # 4-body Hbar
    I1A_vo = (
          -0.5 * ccpy_einsum("nomg,egno->em", X.aa.ooov, T.aa)
          - ccpy_einsum("nomg,egno->em", X.ab.ooov, T.ab)
    )
    I1B_vo = (
          -0.5 * ccpy_einsum("nomg,egno->em", X.bb.ooov, T.bb)
          - ccpy_einsum("ongm,geon->em", X.ab.oovo, T.ab)
    )
    LH.a += ccpy_einsum("em,imae->ai", I1A_vo, H.aa.oovv)
    LH.a += ccpy_einsum("em,imae->ai", I1B_vo, H.ab.oovv)

    LH.a -= ccpy_einsum("nm,mina->ai", X.a.oo, H.aa.ooov)
    LH.a -= ccpy_einsum("nm,iman->ai", X.b.oo, H.ab.oovo)
    LH.a -= ccpy_einsum("ef,fiea->ai", X.a.vv, H.aa.vovv)
    LH.a -= ccpy_einsum("ef,ifae->ai", X.b.vv, H.ab.ovvv)

    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
    LH.a += ccpy_einsum("ie,ea->ai", H.a.ov, X.a.vv)
    LH.a -= ccpy_einsum("ma,im->ai", H.a.ov, X.a.oo)

    LH.a += 0.5 * ccpy_einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo)
    LH.a += ccpy_einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo)
    LH.a += ccpy_einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov)
    LH.a += ccpy_einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo)
    LH.a += ccpy_einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov)

    LH.a -= 0.5 * ccpy_einsum("gife,efag->ai", X.aa.vovv, H.aa.vvvv)
    LH.a -= ccpy_einsum("igef,efag->ai", X.ab.ovvv, H.ab.vvvv)
    LH.a -= ccpy_einsum("imne,enma->ai", X.aa.ooov, H.aa.voov)
    LH.a -= ccpy_einsum("imne,neam->ai", X.ab.ooov, H.ab.ovvo)
    LH.a -= ccpy_einsum("imen,enam->ai", X.ab.oovo, H.ab.vovo)

    LH.a += 0.5 * ccpy_einsum("nmoa,iomn->ai", H.aa.ooov, X.aa.oooo)
    LH.a += ccpy_einsum("mnao,iomn->ai", H.ab.oovo, X.ab.oooo)
    LH.a += ccpy_einsum("fmae,eimf->ai", H.aa.vovv, X.aa.voov)
    LH.a += ccpy_einsum("fmae,iefm->ai", H.ab.vovv, X.ab.ovvo)
    LH.a += ccpy_einsum("mfae,iemf->ai", H.ab.ovvv, X.ab.ovov)

    LH.a -= 0.5 * ccpy_einsum("gife,efag->ai", H.aa.vovv, X.aa.vvvv)
    LH.a -= ccpy_einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv)
    LH.a -= ccpy_einsum("imne,enma->ai", H.aa.ooov, X.aa.voov)
    LH.a -= ccpy_einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo)
    LH.a -= ccpy_einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo)
    
    return LH


def build_LH_1B(L, LH, T, H, X):
    LH.b = ccpy_einsum("ea,ei->ai", H.b.vv, L.b)
    LH.b -= ccpy_einsum("im,am->ai", H.b.oo, L.b)
    LH.b += ccpy_einsum("eima,em->ai", H.bb.voov, L.b)
    LH.b += ccpy_einsum("eima,em->ai", H.ab.voov, L.a)

    ## These terms contain T3 in them from the 2-body HBar components used here ##
    LH.b += 0.5 * ccpy_einsum("fena,efin->ai", H.bb.vvov, L.bb)
    LH.b += ccpy_einsum("fena,feni->ai", H.ab.vvov, L.ab)
    LH.b -= 0.5 * ccpy_einsum("finm,afmn->ai", H.bb.vooo, L.bb)
    LH.b -= ccpy_einsum("finm,fanm->ai", H.ab.vooo, L.ab)

    I1 = 0.25 * ccpy_einsum("efmn,fgnm->ge", L.bb, T.bb)
    I2 = -0.25 * ccpy_einsum("efmn,egnm->gf", L.bb, T.bb)
    I3 = -0.25 * ccpy_einsum("efmo,efno->mn", L.bb, T.bb)
    I4 = 0.25 * ccpy_einsum("efmo,efnm->on", L.bb, T.bb)
    LH.b += ccpy_einsum("ge,eiga->ai", I1, H.bb.vovv)
    LH.b += ccpy_einsum("gf,figa->ai", I2, H.bb.vovv)
    LH.b += ccpy_einsum("mn,nima->ai", I3, H.bb.ooov)
    LH.b += ccpy_einsum("on,nioa->ai", I4, H.bb.ooov)

    I1 = -ccpy_einsum("baji,bani->jn", L.ab, T.ab)
    I2 = ccpy_einsum("baji,faji->fb", L.ab, T.ab)
    I3 = ccpy_einsum("baji,bfji->fa", L.ab, T.ab)
    I4 = -ccpy_einsum("baji,bajn->in", L.ab, T.ab)
    LH.b += ccpy_einsum("jn,nmje->em", I1, H.ab.ooov)
    LH.b += ccpy_einsum("fb,bmfe->em", I2, H.ab.vovv)
    LH.b += ccpy_einsum("fa,amfe->em", I3, H.bb.vovv)
    LH.b += ccpy_einsum("in,nmie->em", I4, H.bb.ooov)

    I1 = 0.25 * ccpy_einsum("abij,fbij->fa", L.aa, T.aa)
    I2 = -0.25 * ccpy_einsum("abij,faij->fb", L.aa, T.aa)
    I3 = -0.25 * ccpy_einsum("abij,abnj->in", L.aa, T.aa)
    I4 = 0.25 * ccpy_einsum("abij,abni->jn", L.aa, T.aa)
    LH.b += ccpy_einsum("fa,amfe->em", I1, H.ab.vovv)
    LH.b += ccpy_einsum("fb,bmfe->em", I2, H.ab.vovv)
    LH.b += ccpy_einsum("in,nmie->em", I3, H.ab.ooov)
    LH.b += ccpy_einsum("jn,nmje->em", I4, H.ab.ooov)

    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH.b += ccpy_einsum("em,imae->ai", X.b.vo, H.bb.oovv)
    LH.b += ccpy_einsum("em,miea->ai", X.a.vo, H.ab.oovv)

    # 4-body Hbar
    I1B_vo = (
            -0.5 * ccpy_einsum("nomg,egno->em", X.bb.ooov, T.bb)
            - ccpy_einsum("ongm,geon->em", X.ab.oovo, T.ab)
    )
    I1A_vo = (
            -0.5 * ccpy_einsum("nomg,egno->em", X.aa.ooov, T.aa)
            - ccpy_einsum("nomg,egno->em", X.ab.ooov, T.ab)
    )
    LH.b += ccpy_einsum("em,imae->ai", I1B_vo, H.bb.oovv)
    LH.b += ccpy_einsum("em,miea->ai", I1A_vo, H.ab.oovv)

    LH.b -= ccpy_einsum("nm,mina->ai", X.b.oo, H.bb.ooov)
    LH.b -= ccpy_einsum("nm,mina->ai", X.a.oo, H.ab.ooov)
    LH.b -= ccpy_einsum("ef,fiea->ai", X.b.vv, H.bb.vovv)
    LH.b -= ccpy_einsum("ef,fiea->ai", X.a.vv, H.ab.vovv)

    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
    LH.b += ccpy_einsum("ie,ea->ai", H.b.ov, X.b.vv)
    LH.b -= ccpy_einsum("ma,im->ai", H.b.ov, X.b.oo)

    LH.b += 0.5 * ccpy_einsum("nmoa,iomn->ai", X.bb.ooov, H.bb.oooo)
    LH.b += ccpy_einsum("nmoa,oinm->ai", X.ab.ooov, H.ab.oooo)
    LH.b += ccpy_einsum("fmae,eimf->ai", X.bb.vovv, H.bb.voov)
    LH.b += ccpy_einsum("mfea,eimf->ai", X.ab.ovvv, H.ab.voov)
    LH.b += ccpy_einsum("fmea,eifm->ai", X.ab.vovv, H.ab.vovo)

    LH.b -= 0.5 * ccpy_einsum("gife,efag->ai", X.bb.vovv, H.bb.vvvv)
    LH.b -= ccpy_einsum("gife,fega->ai", X.ab.vovv, H.ab.vvvv)
    LH.b -= ccpy_einsum("imne,enma->ai", X.bb.ooov, H.bb.voov)
    LH.b -= ccpy_einsum("mien,enma->ai", X.ab.oovo, H.ab.voov)
    LH.b -= ccpy_einsum("mine,nema->ai", X.ab.ooov, H.ab.ovov)

    LH.b += 0.5 * ccpy_einsum("nmoa,iomn->ai", H.bb.ooov, X.bb.oooo)
    LH.b += ccpy_einsum("nmoa,oinm->ai", H.ab.ooov, X.ab.oooo)
    LH.b += ccpy_einsum("fmae,eimf->ai", H.bb.vovv, X.bb.voov)
    LH.b += ccpy_einsum("mfea,eimf->ai", H.ab.ovvv, X.ab.voov)
    LH.b += ccpy_einsum("fmea,eifm->ai", H.ab.vovv, X.ab.vovo)

    LH.b -= 0.5 * ccpy_einsum("gife,efag->ai", H.bb.vovv, X.bb.vvvv)
    LH.b -= ccpy_einsum("gife,fega->ai", H.ab.vovv, X.ab.vvvv)
    LH.b -= ccpy_einsum("imne,enma->ai", H.bb.ooov, X.bb.voov)
    LH.b -= ccpy_einsum("mien,enma->ai", H.ab.oovo, X.ab.voov)
    LH.b -= ccpy_einsum("mine,nema->ai", H.ab.ooov, X.ab.ovov)

    return LH


def build_LH_2A(L, LH, T, H, X):

    LH.aa = 0.5 * ccpy_einsum("ea,ebij->abij", H.a.vv, L.aa)
    LH.aa -= 0.5 * ccpy_einsum("im,abmj->abij", H.a.oo, L.aa)

    LH.aa += ccpy_einsum("jb,ai->abij", H.a.ov, L.a)

    I1 = (
          -0.5 * ccpy_einsum("afmn,efmn->ea", L.aa, T.aa)
          - ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
    )
    LH.aa += 0.5 * ccpy_einsum("ea,ijeb->abij", I1, H.aa.oovv)

    I1 = (
          0.5 * ccpy_einsum("efin,efmn->im", L.aa, T.aa)
          + ccpy_einsum("efin,efmn->im", L.ab, T.ab)
    )
    LH.aa -= 0.5 * ccpy_einsum("im,mjab->abij", I1, H.aa.oovv)

    LH.aa += ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.aa)
    LH.aa += ccpy_einsum("ieam,bejm->abij", H.ab.ovvo, L.ab)

    LH.aa += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.aa.oooo, L.aa)
    LH.aa += 0.125 * ccpy_einsum("efab,efij->abij", H.aa.vvvv, L.aa)

    LH.aa += 0.5 * ccpy_einsum("ejab,ei->abij", H.aa.vovv, L.a)
    LH.aa -= 0.5 * ccpy_einsum("ijmb,am->abij", H.aa.ooov, L.a)

    # < 0 | L3 * H(2) | ijab >
    LH.aa -= ccpy_einsum("ejfb,fiea->abij", X.aa.vovv, H.aa.vovv) # 1
    LH.aa -= ccpy_einsum("njmb,mina->abij", X.aa.ooov, H.aa.ooov) # 2
    LH.aa -= 0.25 * ccpy_einsum("enab,jine->abij", X.aa.vovv, H.aa.ooov) # 3
    LH.aa -= 0.25 * ccpy_einsum("jine,enab->abij", X.aa.ooov, H.aa.vovv) # 4
    LH.aa -= ccpy_einsum("jebf,ifae->abij", X.ab.ovvv, H.ab.ovvv) # 5
    LH.aa -= ccpy_einsum("jnbm,iman->abij", X.ab.oovo, H.ab.oovo) # 6

    # < 0 | L3 * (H(2) * T3) | ijab >
    LH.aa += ccpy_einsum("ejmb,imae->abij", X.aa.voov, H.aa.oovv) # 1
    LH.aa += ccpy_einsum("jebm,imae->abij", X.ab.ovvo, H.ab.oovv) # 2
    LH.aa += 0.125 * ccpy_einsum("efab,ijef->abij", X.aa.vvvv, H.aa.oovv) # 3
    LH.aa += 0.125 * ccpy_einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv) # 4

    # 4-body HBar
    LH.aa += 0.5 * ccpy_einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv) # 1
    LH.aa -= 0.5 * ccpy_einsum("im,jmba->abij", X.a.oo, H.aa.oovv) # 2

    # Moment-like terms
    LH.aa += 0.25 * ccpy_einsum("ebfijn,fena->abij", L.aaa, H.aa.vvov) # 1
    LH.aa += 0.5 * ccpy_einsum("ebfijn,efan->abij", L.aab, H.ab.vvvo) # 2
    LH.aa -= 0.25 * ccpy_einsum("abfmjn,finm->abij", L.aaa, H.aa.vooo) # 3
    LH.aa -= 0.5 * ccpy_einsum("abfmjn,ifmn->abij", L.aab, H.ab.ovoo) # 4

    LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))

    return LH


def build_LH_2B(L, LH, T, H, X):

    LH.ab = -ccpy_einsum("ijmb,am->abij", H.ab.ooov, L.a)
    LH.ab -= ccpy_einsum("ijam,bm->abij", H.ab.oovo, L.b)

    LH.ab += ccpy_einsum("ejab,ei->abij", H.ab.vovv, L.a)
    LH.ab += ccpy_einsum("ieab,ej->abij", H.ab.ovvv, L.b)

    LH.ab += ccpy_einsum("ijmn,abmn->abij", H.ab.oooo, L.ab)
    LH.ab += ccpy_einsum("efab,efij->abij", H.ab.vvvv, L.ab)

    LH.ab += ccpy_einsum("ejmb,aeim->abij", H.ab.voov, L.aa)
    LH.ab += ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.ab)
    LH.ab += ccpy_einsum("ejmb,aeim->abij", H.bb.voov, L.ab)
    LH.ab += ccpy_einsum("ieam,ebmj->abij", H.ab.ovvo, L.bb)
    LH.ab -= ccpy_einsum("iemb,aemj->abij", H.ab.ovov, L.ab)
    LH.ab -= ccpy_einsum("ejam,ebim->abij", H.ab.vovo, L.ab)

    I1 = (
          -0.5 * ccpy_einsum("afmn,efmn->ea", L.aa, T.aa)
          - ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
    )
    LH.ab += ccpy_einsum("ea,ijeb->abij", I1, H.ab.oovv)

    I1 = (
          0.5 * ccpy_einsum("efin,efmn->im", L.aa, T.aa)
          + ccpy_einsum("efin,efmn->im", L.ab, T.ab)
    )
    LH.ab -= ccpy_einsum("im,mjab->abij", I1, H.ab.oovv)

    I1 = (
          -0.5 * ccpy_einsum("afmn,efmn->ea", L.bb, T.bb)
          - ccpy_einsum("fanm,fenm->ea", L.ab, T.ab)
    )
    LH.ab += ccpy_einsum("ea,jibe->baji", I1, H.ab.oovv)

    I1 = (
          0.5 * ccpy_einsum("efin,efmn->im", L.bb, T.bb)
          + ccpy_einsum("feni,fenm->im", L.ab, T.ab)
    )
    LH.ab -= ccpy_einsum("im,jmba->baji", I1, H.ab.oovv)

    LH.ab += ccpy_einsum("ea,ebij->abij", H.a.vv, L.ab)
    LH.ab += ccpy_einsum("eb,aeij->abij", H.b.vv, L.ab)
    LH.ab -= ccpy_einsum("im,abmj->abij", H.a.oo, L.ab)
    LH.ab -= ccpy_einsum("jm,abim->abij", H.b.oo, L.ab)
    LH.ab += ccpy_einsum("jb,ai->abij", H.b.ov, L.a)
    LH.ab += ccpy_einsum("ia,bj->abij", H.a.ov, L.b)

    # < 0 | L3 * H(2) | ij~ab~ >
    LH.ab -= ccpy_einsum("ejfb,fiea->abij", X.ab.vovv, H.aa.vovv) # 1
    LH.ab -= ccpy_einsum("ejfb,ifae->abij", X.bb.vovv, H.ab.ovvv) # 2
    LH.ab -= ccpy_einsum("eifa,fjeb->abij", X.aa.vovv, H.ab.vovv) # 3
    LH.ab -= ccpy_einsum("ieaf,fjeb->abij", X.ab.ovvv, H.bb.vovv) # 4
    LH.ab -= ccpy_einsum("njmb,mina->abij", X.ab.ooov, H.aa.ooov) # 5
    LH.ab -= ccpy_einsum("njmb,iman->abij", X.bb.ooov, H.ab.oovo) # 6
    LH.ab -= ccpy_einsum("nima,mjnb->abij", X.aa.ooov, H.ab.ooov) # 7
    LH.ab -= ccpy_einsum("inam,mjnb->abij", X.ab.oovo, H.bb.ooov) # 8
    LH.ab += ccpy_einsum("inmb,mjan->abij", X.ab.ooov, H.ab.oovo) # 9
    LH.ab += ccpy_einsum("ifeb,ejaf->abij", X.ab.ovvv, H.ab.vovv) # 10
    LH.ab += ccpy_einsum("ejaf,ifeb->abij", X.ab.vovv, H.ab.ovvv) # 11
    LH.ab += ccpy_einsum("mjan,inmb->abij", X.ab.oovo, H.ab.ooov) # 12
    LH.ab -= ccpy_einsum("enab,ijen->abij", X.ab.vovv, H.ab.oovo) # 13
    LH.ab -= ccpy_einsum("mfab,ijmf->abij", X.ab.ovvv, H.ab.ooov) # 14
    LH.ab -= ccpy_einsum("ijmf,mfab->abij", X.ab.ooov, H.ab.ovvv) # 15
    LH.ab -= ccpy_einsum("ijen,enab->abij", X.ab.oovo, H.ab.vovv) # 16

    # < 0 | L3 * (H(2) * T3)_C | ij~ab~ >
    LH.ab += (
               ccpy_einsum("ejmb,miea->abij", X.ab.voov, H.aa.oovv)
               + ccpy_einsum("ejmb,imae->abij", X.bb.voov, H.ab.oovv)
    ) # 1
    LH.ab += (
               ccpy_einsum("eima,mjeb->abij", X.aa.voov, H.ab.oovv)
               + ccpy_einsum("ieam,mjeb->abij", X.ab.ovvo, H.bb.oovv)
    ) # 2
    LH.ab -= ccpy_einsum("iemb,mjae->abij", X.ab.ovov, H.ab.oovv) # 3
    LH.ab -= ccpy_einsum("ejam,imeb->abij", X.ab.vovo, H.ab.oovv) # 4
    LH.ab += ccpy_einsum("efab,ijef->abij", X.ab.vvvv, H.ab.oovv) # 5
    LH.ab += ccpy_einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv) # 6

    # 4-body HBar
    LH.ab += ccpy_einsum("ea,ijeb->abij", X.a.vv, H.ab.oovv) # 1
    LH.ab += ccpy_einsum("eb,ijae->abij", X.b.vv, H.ab.oovv) # 2
    LH.ab -= ccpy_einsum("im,mjab->abij", X.a.oo, H.ab.oovv) # 3
    LH.ab -= ccpy_einsum("jm,imab->abij", X.b.oo, H.ab.oovv) # 4

    # Moment-like terms
    LH.ab -= 0.5 * ccpy_einsum("afbmnj,finm->abij", L.aab, H.aa.vooo) # 1
    LH.ab -= ccpy_einsum("afbmnj,ifmn->abij", L.abb, H.ab.ovoo) # 2
    LH.ab -= ccpy_einsum("afbinm,fjnm->abij", L.aab, H.ab.vooo) # 3
    LH.ab -= 0.5 * ccpy_einsum("afbinm,fjnm->abij", L.abb, H.bb.vooo) # 4

    LH.ab += 0.5 * ccpy_einsum("efbinj,fena->abij", L.aab, H.aa.vvov) # 5
    LH.ab += ccpy_einsum("efbinj,efan->abij", L.abb, H.ab.vvvo) # 6
    LH.ab += ccpy_einsum("afeinj,fenb->abij", L.aab, H.ab.vvov) # 7
    LH.ab += 0.5 * ccpy_einsum("afeinj,fenb->abij", L.abb, H.bb.vvov) # 8

    return LH

def build_LH_2C(L, LH, T, H, X):

    LH.bb = 0.5 * ccpy_einsum("ea,ebij->abij", H.b.vv, L.bb)
    LH.bb -= 0.5 * ccpy_einsum("im,abmj->abij", H.b.oo, L.bb)

    LH.bb += ccpy_einsum("jb,ai->abij", H.b.ov, L.b)

    I1 = (
          -0.5 * ccpy_einsum("afmn,efmn->ea", L.bb, T.bb)
          - ccpy_einsum("fanm,fenm->ea", L.ab, T.ab)
    )
    LH.bb += 0.5 * ccpy_einsum("ea,ijeb->abij", I1, H.bb.oovv)

    I1 = (
          0.5 * ccpy_einsum("efin,efmn->im", L.bb, T.bb)
          + ccpy_einsum("feni,fenm->im", L.ab, T.ab)
    )
    LH.bb -= 0.5 * ccpy_einsum("im,mjab->abij", I1, H.bb.oovv)

    LH.bb += ccpy_einsum("eima,ebmj->abij", H.bb.voov, L.bb)
    LH.bb += ccpy_einsum("eima,ebmj->abij", H.ab.voov, L.ab)

    LH.bb += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.bb.oooo, L.bb)
    LH.bb += 0.125 * ccpy_einsum("efab,efij->abij", H.bb.vvvv, L.bb)

    LH.bb += 0.5 * ccpy_einsum("ejab,ei->abij", H.bb.vovv, L.b)
    LH.bb -= 0.5 * ccpy_einsum("ijmb,am->abij", H.bb.ooov, L.b)

    # < 0 | L3 * H(2) | ijab >
    LH.bb -= ccpy_einsum("ejfb,fiea->abij", X.bb.vovv, H.bb.vovv) # 1
    LH.bb -= ccpy_einsum("njmb,mina->abij", X.bb.ooov, H.bb.ooov) # 2
    LH.bb -= 0.25 * ccpy_einsum("enab,jine->abij", X.bb.vovv, H.bb.ooov) # 3
    LH.bb -= 0.25 * ccpy_einsum("jine,enab->abij", X.bb.ooov, H.bb.vovv) # 4
    LH.bb -= ccpy_einsum("ejfb,fiea->abij", X.ab.vovv, H.ab.vovv) # 5
    LH.bb -= ccpy_einsum("njmb,mina->abij", X.ab.ooov, H.ab.ooov) # 6

    # < 0 | L3 * (H(2) * T3) | ijab >
    LH.bb += ccpy_einsum("ejmb,imae->abij", X.bb.voov, H.bb.oovv) # 1
    LH.bb += ccpy_einsum("ejmb,miea->abij", X.ab.voov, H.ab.oovv) # 2
    LH.bb += 0.125 * ccpy_einsum("efab,ijef->abij", X.bb.vvvv, H.bb.oovv) # 3
    LH.bb += 0.125 * ccpy_einsum("ijmn,mnab->abij", X.bb.oooo, H.bb.oovv) # 4

    # 4-body HBar
    LH.bb += 0.5 * ccpy_einsum("ea,ijeb->abij", X.b.vv, H.bb.oovv) # 1
    LH.bb -= 0.5 * ccpy_einsum("im,jmba->abij", X.b.oo, H.bb.oovv) # 2

    # Moment-like terms
    LH.bb += 0.25 * ccpy_einsum("ebfijn,fena->abij", L.bbb, H.bb.vvov) # 1
    LH.bb += 0.5 * ccpy_einsum("fbenji,fena->abij", L.abb, H.ab.vvov) # 2
    LH.bb -= 0.25 * ccpy_einsum("abfmjn,finm->abij", L.bbb, H.bb.vooo) # 3
    LH.bb -= 0.5 * ccpy_einsum("fbanjm,finm->abij", L.abb, H.ab.vooo) # 4

    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))

    return LH

def build_LH_3A(L, LH, H, X):

    # < 0 | L1 * H(2) | ijkabc >
    LH.aaa = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.a, H.aa.oovv)

    # < 0 | L2 * H(2) | ijkabc >
    LH.aaa += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.aa, H.a.ov)

    LH.aaa += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv)
    LH.aaa -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov)

    # < 0 | L3 * H(2) | ijkabc >
    LH.aaa += (3.0 / 36.0) * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aaa)
    LH.aaa -= (3.0 / 36.0) * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aaa)
    LH.aaa += (9.0 / 36.0) * ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa)
    LH.aaa += (9.0 / 36.0) * ccpy_einsum("ieam,bcejkm->abcijk", H.ab.ovvo, L.aab)
    LH.aaa += (3.0 / 72.0) * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aaa)
    LH.aaa += (3.0 / 72.0) * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aaa)

    LH.aaa += (9.0 / 36.0) * ccpy_einsum("ijeb,ekac->abcijk", H.aa.oovv, X.aa.vovv)
    LH.aaa -= (9.0 / 36.0) * ccpy_einsum("mjab,ikmc->abcijk", H.aa.oovv, X.aa.ooov)

    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(LH.aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    LH.aaa -= np.transpose(LH.aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(LH.aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return LH

def build_LH_3B(L, LH, H, X):

    # < 0 | L1 * H(2) | ijk~abc~ >
    LH.aab = ccpy_einsum("ai,jkbc->abcijk", L.a, H.ab.oovv)
    LH.aab += 0.25 * ccpy_einsum("ck,ijab->abcijk", L.b, H.aa.oovv)

    # < 0 | L2 * H(2) | ijk~abc~ >

    LH.aab += ccpy_einsum("bcjk,ia->abcijk", L.ab, H.a.ov)
    LH.aab += 0.25 * ccpy_einsum("abij,kc->abcijk", L.aa, H.b.ov)

    LH.aab += 0.5 * ccpy_einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa)
    LH.aab -= 0.5 * ccpy_einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa)
    LH.aab += ccpy_einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab)
    LH.aab -= ccpy_einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab)
    LH.aab += 0.5 * ccpy_einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab)
    LH.aab -= 0.5 * ccpy_einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab)

    LH.aab += 0.5 * ccpy_einsum("ekbc,ijae->abcijk", X.ab.vovv, H.aa.oovv)
    LH.aab -= 0.5 * ccpy_einsum("jkmc,imab->abcijk", X.ab.ooov, H.aa.oovv)
    LH.aab += ccpy_einsum("ieac,jkbe->abcijk", X.ab.ovvv, H.ab.oovv)
    LH.aab -= ccpy_einsum("ikam,jmbc->abcijk", X.ab.oovo, H.ab.oovv)
    LH.aab += 0.5 * ccpy_einsum("eiba,jkec->abcijk", X.aa.vovv, H.ab.oovv)
    LH.aab -= 0.5 * ccpy_einsum("jima,mkbc->abcijk", X.aa.ooov, H.ab.oovv)

    # < 0 | L3 * H(2) | ijk~abc~ >
    LH.aab -= 0.5 * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aab)
    LH.aab -= 0.25 * ccpy_einsum("km,abcijm->abcijk", H.b.oo, L.aab)
    LH.aab += 0.5 * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aab)
    LH.aab += 0.25 * ccpy_einsum("ec,abeijk->abcijk", H.b.vv, L.aab)
    LH.aab += 0.125 * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aab)
    LH.aab += 0.5 * ccpy_einsum("jkmn,abcimn->abcijk", H.ab.oooo, L.aab)
    LH.aab += 0.125 * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aab)
    LH.aab += 0.5 * ccpy_einsum("efbc,aefijk->abcijk", H.ab.vvvv, L.aab)
    LH.aab += ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aab)
    LH.aab += ccpy_einsum("ieam,becjmk->abcijk", H.ab.ovvo, L.abb)
    LH.aab += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.ab.voov, L.aaa)
    LH.aab += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.bb.voov, L.aab)
    LH.aab -= 0.5 * ccpy_einsum("ekam,ebcijm->abcijk", H.ab.vovo, L.aab)
    LH.aab -= 0.5 * ccpy_einsum("iemc,abemjk->abcijk", H.ab.ovov, L.aab)

    LH.aab -= np.transpose(LH.aab, (1, 0, 2, 3, 4, 5))
    LH.aab -= np.transpose(LH.aab, (0, 1, 2, 4, 3, 5))

    return LH

def build_LH_3C(L, LH, H, X):

    # < 0 | L1 * H(2) | ijk~abc~ >
    LH.abb = ccpy_einsum("ai,kjcb->cbakji", L.b, H.ab.oovv)
    LH.abb += 0.25 * ccpy_einsum("ck,ijab->cbakji", L.a, H.bb.oovv)

    # < 0 | L2 * H(2) | ijk~abc~ >
    LH.abb += ccpy_einsum("cbkj,ia->cbakji", L.ab, H.b.ov)
    LH.abb += 0.25 * ccpy_einsum("abij,kc->cbakji", L.bb, H.a.ov)

    LH.abb += 0.5 * ccpy_einsum("kecb,aeij->cbakji", H.ab.ovvv, L.bb)
    LH.abb -= 0.5 * ccpy_einsum("kjcm,abim->cbakji", H.ab.oovo, L.bb)
    LH.abb += ccpy_einsum("eica,ebkj->cbakji", H.ab.vovv, L.ab)
    LH.abb -= ccpy_einsum("kima,cbmj->cbakji", H.ab.ooov, L.ab)
    LH.abb += 0.5 * ccpy_einsum("eiba,cekj->cbakji", H.bb.vovv, L.ab)
    LH.abb -= 0.5 * ccpy_einsum("jima,cbkm->cbakji", H.bb.ooov, L.ab)

    LH.abb += 0.5 * ccpy_einsum("kecb,ijae->cbakji", X.ab.ovvv, H.bb.oovv)
    LH.abb -= 0.5 * ccpy_einsum("kjcm,imab->cbakji", X.ab.oovo, H.bb.oovv)
    LH.abb += ccpy_einsum("eica,kjeb->cbakji", X.ab.vovv, H.ab.oovv)
    LH.abb -= ccpy_einsum("kima,mjcb->cbakji", X.ab.ooov, H.ab.oovv)
    LH.abb += 0.5 * ccpy_einsum("eiba,kjce->cbakji", X.bb.vovv, H.ab.oovv)
    LH.abb -= 0.5 * ccpy_einsum("jima,kmcb->cbakji", X.bb.ooov, H.ab.oovv)

    # < 0 | L3 * H(2) | ijk~abc~ >
    LH.abb -= 0.5 * ccpy_einsum("im,cbakjm->cbakji", H.b.oo, L.abb)
    LH.abb -= 0.25 * ccpy_einsum("km,cbamji->cbakji", H.a.oo, L.abb)
    LH.abb += 0.5 * ccpy_einsum("ea,cbekji->cbakji", H.b.vv, L.abb)
    LH.abb += 0.25 * ccpy_einsum("ec,ebakji->cbakji", H.a.vv, L.abb)
    LH.abb += 0.125 * ccpy_einsum("ijmn,cbaknm->cbakji", H.bb.oooo, L.abb)
    LH.abb += 0.5 * ccpy_einsum("kjnm,cbanmi->cbakji", H.ab.oooo, L.abb)
    LH.abb += 0.125 * ccpy_einsum("efab,cfekji->cbakji", H.bb.vvvv, L.abb)
    LH.abb += 0.5 * ccpy_einsum("fecb,feakji->cbakji", H.ab.vvvv, L.abb)
    LH.abb += ccpy_einsum("eima,cbekjm->cbakji", H.bb.voov, L.abb)
    LH.abb += ccpy_einsum("eima,cebkmj->cbakji", H.ab.voov, L.aab)
    LH.abb += 0.25 * ccpy_einsum("kecm,abeijm->cbakji", H.ab.ovvo, L.bbb)
    LH.abb += 0.25 * ccpy_einsum("ekmc,ebamji->cbakji", H.aa.voov, L.abb)
    LH.abb -= 0.5 * ccpy_einsum("kema,cbemji->cbakji", H.ab.ovov, L.abb)
    LH.abb -= 0.5 * ccpy_einsum("eicm,ebakjm->cbakji", H.ab.vovo, L.abb)

    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4, 5))
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 3, 5, 4))

    return LH

def build_LH_3D(L, LH, H, X):

    # < 0 | L1 * H(2) | ijkabc >
    LH.bbb = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.b, H.bb.oovv)

    # < 0 | L2 * H(2) | ijkabc >
    LH.bbb += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.bb, H.b.ov)

    LH.bbb += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.bb, H.bb.vovv)
    LH.bbb -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.bb, H.bb.ooov)

    # < 0 | L3 * H(2) | ijkabc >
    LH.bbb += (3.0 / 36.0) * ccpy_einsum("ea,ebcijk->abcijk", H.b.vv, L.bbb)
    LH.bbb -= (3.0 / 36.0) * ccpy_einsum("im,abcmjk->abcijk", H.b.oo, L.bbb)
    LH.bbb += (9.0 / 36.0) * ccpy_einsum("eima,ebcmjk->abcijk", H.bb.voov, L.bbb)
    LH.bbb += (9.0 / 36.0) * ccpy_einsum("eima,ecbmkj->abcijk", H.ab.voov, L.abb)
    LH.bbb += (3.0 / 72.0) * ccpy_einsum("ijmn,abcmnk->abcijk", H.bb.oooo, L.bbb)
    LH.bbb += (3.0 / 72.0) * ccpy_einsum("efab,efcijk->abcijk", H.bb.vvvv, L.bbb)

    LH.bbb += (9.0 / 36.0) * ccpy_einsum("ijeb,ekac->abcijk", H.bb.oovv, X.bb.vovv)
    LH.bbb -= (9.0 / 36.0) * ccpy_einsum("mjab,ikmc->abcijk", H.bb.oovv, X.bb.ooov)

    LH.bbb -= np.transpose(LH.bbb, (0, 1, 2, 3, 5, 4)) # (jk)
    LH.bbb -= np.transpose(LH.bbb, (0, 1, 2, 4, 3, 5)) + np.transpose(LH.bbb, (0, 1, 2, 5, 4, 3)) # (i/jk)
    LH.bbb -= np.transpose(LH.bbb, (0, 2, 1, 3, 4, 5)) # (bc)
    LH.bbb -= np.transpose(LH.bbb, (2, 1, 0, 3, 4, 5)) + np.transpose(LH.bbb, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return LH