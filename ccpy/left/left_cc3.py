import numpy as np

from ccpy.lib.core import cc3_loops
from ccpy.left.left_cc_intermediates import build_left_cc3_intermediates

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

    # get LT intermediates
    X = build_left_cc3_intermediates(L, T, system)

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
    if flag_RHF:
        L.b = L.a.copy()
        L.bb = L.aa.copy()
        LH.b = LH.a.copy()
        LH.bb = LH.aa.copy()
    return L, LH

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.b, L.aa, L.ab, L.bb = cc3_loops.update_r(
        L.a,
        L.b,
        L.aa,
        L.ab,
        L.bb,
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
    return L

def LH_fun(LH, L, T, H, flag_RHF, system):
    # get LT intermediates
    X = build_left_cc3_intermediates(L, T, system)
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
    # < 0 | L2 * (H(1) * T3)_C | ia >
    LH.a += np.einsum("em,imae->ai", X.a.vo, H.aa.oovv, optimize=True)
    LH.a += np.einsum("em,imae->ai", X.b.vo, H.ab.oovv, optimize=True)
    # < 0 | L3 * H(2) | ia >
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
    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
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
    return LH

def build_LH_2A(L, LH, T, H, X):
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
    return LH

def build_LH_2B(L, LH, T, H, X):
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
    return LH

def build_LH_2C(L, LH, T, H, X):
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
    return LH