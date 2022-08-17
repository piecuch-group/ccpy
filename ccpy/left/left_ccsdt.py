import numpy as np

from ccpy.utilities.updates import cc_loops
from ccpy.left.left_cc_intermediates import build_left_ccsdt_intermediates

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF):

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
    LH = build_LH_3A(L, LH, T, H, X)
    LH = build_LH_3B(L, LH, T, H, X)
    if flag_RHF:
        LH.abb = np.transpose(LH.aab, (2, 1, 0, 5, 4, 3))
        LH.bbb = LH.aaa.copy()
    else:
        LH = build_LH_3C(L, LH, T, H, X)
        LH = build_LH_3D(L, LH, T, H, X)

    # Add Hamiltonian if ground-state calculation
    if is_ground:
        LH.a += np.transpose(H.a.ov, (1, 0))
        LH.b += np.transpose(H.b.ov, (1, 0))
        LH.aa += np.transpose(H.aa.oovv, (2, 3, 0, 1))
        LH.ab += np.transpose(H.ab.oovv, (2, 3, 0, 1))
        LH.bb += np.transpose(H.bb.oovv, (2, 3, 0, 1))

    L.a, L.b, L.aa, L.ab, L.bb, L.aaa, L.aab, L.abb, L.bbb,\
    LH.a, LH.b, LH.aa, LH.ab. LH.bb, LH.aaa, LH.aab, LH.abb, LH.bbb = cc_loops.cc_loops.update_l_ccsdt(L.a, L.b, L.aa, L.ab, L.bb,
                                                                                                       L.aaa, L.aab, L.abb, L.bbb,
                                                                                                       LH.a, LH.b, LH.aa, LH.ab, LH.bb,
                                                                                                       LH.aaa, LH.aab, LH.abb, LH.bbb,
                                                                                                       omega,
                                                                                                       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                                                       shift)
    return L, LH

def build_LH_1A(L, LH, T, H, X):

    LH.a = np.einsum("ea,ei->ai", H.a.vv, L.a, optimize=True)
    LH.a -= np.einsum("im,am->ai", H.a.oo, L.a, optimize=True)
    LH.a += np.einsum("eima,em->ai", H.aa.voov, L.a, optimize=True)
    LH.a += np.einsum("ieam,em->ai", H.ab.ovvo, L.b, optimize=True)
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

    LH.a += 0.5 * np.einsum("efin,fena->ai", L.aa, H.aa.vvov, optimize=True) 
    LH.a += np.einsum("efin,efna->ai", L.ab, H.ab.vvvo, optimize=True) 
    LH.a -= 0.5 * np.einsum("afmn,finm->ai", L.aa, H.aa.vooo, optimize=True)
    LH.a -= np.einsum("afmn,ifmn->ai", L.ab, H.ab.ovoo, optimize=True)

    # < 0 | L3 * H(2) | ia >
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

    LH.a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    LH.a += np.einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo, optimize=True)
    LH.a += np.einsum("mfea,eimf->ai", X.aa.ovvv, H.aa.voov, optimize=True)
    LH.a += np.einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo, optimize=True)
    LH.a += np.einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov, optimize=True)

    LH.a -= 0.5 * np.einsum("igef,efag->ai", X.aa.ovvv, H.aa.vvvv, optimize=True)
    LH.a -= np.einsum("igef,efag->ai", X.ab.ovvv, H.ab.vvvv, optimize=True)
    LH.a -= np.einsum("imne,enma->ai", X.aa.ooov, H.aa.voov, optimize=True)
    LH.a -= np.einsum("imne,neam->ai", X.ab.ooov, H.ab.ovvo, optimize=True)
    LH.a -= np.einsum("imen,enam->ai", X.ab.oovo, H.ab.vovo, optimize=True)

    # < 0 | L3 * (H(2) * T3)_C | ia >
    LH.a -= np.einsum("nm,iman->ai", X.a.oo, H.aa.ooov, optimize=True)
    LH.a -= np.einsum("nm,iman->ai", X.b.oo, H.ab.ooov, optimize=True)
    LH.a += np.einsum("ef,fiea->ai", X.a.vv, H.aa.vovv, optimize=True)
    LH.a += np.einsum("ef,ifae->ai", X.b.vv, H.ab.vovv, optimize=True)

    LH.a += 0.5 * np.einsum("nmoa,iomn->ai", H.aa.ooov, X.aa.oooo, optimize=True)
    LH.a += np.einsum("mnao,iomn->ai", H.ab.oovo, X.ab.oooo, optimize=True)
    LH.a += np.einsum("mfea,eimf->ai", H.aa.ovvv, X.aa.voov, optimize=True)
    LH.a += np.einsum("fmae,iefm->ai", H.ab.vovv, X.ab.ovvo, optimize=True)
    LH.a += np.einsum("mfae,iemf->ai", H.ab.ovvv, X.ab.ovov, optimize=True)

    LH.a -= 0.5 * np.einsum("igef,efag->ai", H.aa.ovvv, X.aa.vvvv, optimize=True)
    LH.a -= np.einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv, optimize=True)
    LH.a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    LH.a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    LH.a -= np.einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo, optimize=True)
    
    return LH


def build_LH_1B(L, LH, T, H, X):

    LH.b = np.einsum("ea,ei->ai", H.b.vv, L.b, optimize=True)
    LH.b -= np.einsum("im,am->ai", H.b.oo, L.b, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.ab.voov, L.a, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.bb.voov, L.b, optimize=True)
    LH.b -= 0.5 * np.einsum("finm,afmn->ai", H.bb.vooo, L.bb, optimize=True)
    LH.b -= np.einsum("finm,fanm->ai", H.ab.vooo, L.ab, optimize=True)
    LH.b += np.einsum("fena,feni->ai", H.ab.vvov, L.ab, optimize=True)
    LH.b += 0.5 * np.einsum("fena,efin->ai", H.bb.vvov, L.bb, optimize=True)

    I1 = 0.25 * np.einsum("efmn,fgnm->ge", L.bb, T.bb, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", L.bb, T.bb, optimize=True)
    I3 = -0.25 * np.einsum("efmn,efon->mo", L.bb, T.bb, optimize=True)
    I4 = 0.25 * np.einsum("efmn,efom->no", L.bb, T.bb, optimize=True)
    LH.b += (
        np.einsum("ge,eiga->ai", I1, H.bb.vovv, optimize=True)
        + np.einsum("gf,figa->ai", I2, H.bb.vovv, optimize=True)
        + np.einsum("mo,oima->ai", I3, H.bb.ooov, optimize=True)
        + np.einsum("no,oina->ai", I4, H.bb.ooov, optimize=True)
    )

    I1 = 0.25 * np.einsum("efmn,fgnm->ge", L.aa, T.aa, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", L.aa, T.aa, optimize=True)
    I3 = -0.25 * np.einsum("efmn,efon->mo", L.aa, T.aa, optimize=True)
    I4 = 0.25 * np.einsum("efmn,efom->no", L.aa, T.aa, optimize=True)
    LH.b += (
        np.einsum("ge,eiga->ai", I1, H.ab.vovv, optimize=True)
        + np.einsum("gf,figa->ai", I2, H.ab.vovv, optimize=True)
        + np.einsum("mo,oima->ai", I3, H.ab.ooov, optimize=True)
        + np.einsum("no,oina->ai", I4, H.ab.ooov, optimize=True)
    )

    I1 = np.einsum("efmn,gfmn->ge", L.ab, T.ab, optimize=True)
    I2 = np.einsum("fenm,fgnm->ge", L.ab, T.ab, optimize=True)
    I3 = -np.einsum("efmn,efon->mo", L.ab, T.ab, optimize=True)
    I4 = -np.einsum("fenm,feno->mo", L.ab, T.ab, optimize=True)
    LH.b += (
        np.einsum("ge,eiga->ai", I1, H.ab.vovv, optimize=True)
        + np.einsum("ge,eiga->ai", I2, H.bb.vovv, optimize=True)
        + np.einsum("mo,oima->ai", I3, H.ab.ooov, optimize=True)
        + np.einsum("mo,oima->ai", I4, H.bb.ooov, optimize=True)
    )

    return LH


def build_LH_2A(L, LH, T, H, X):

    LH.aa = 0.5 * np.einsum("ea,ebij->abij", H.a.vv, L.aa, optimize=True)
    LH.aa -= 0.5 * np.einsum("im,abmj->abij", H.a.oo, L.aa, optimize=True)

    LH.aa += np.einsum("jb,ai->abij", H.a.ov, L.a, optimize=True)

    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", L.aa, T.aa, optimize=True)
          - np.einsum("afmn,efmn->ea", L.ab, T.ab, optimize=True)
          + X.a.vv
    )
    LH.aa += 0.5 * np.einsum("ea,ijeb->abij", I1, H.aa.oovv, optimize=True)

    I1 = (
          0.5 * np.einsum("efin,efmn->im", L.aa, T.aa, optimize=True)
          + np.einsum("efin,efmn->im", L.ab, T.ab, optimize=True)
          + X.a.oo
    )
    LH.aa -= 0.5 * np.einsum("im,mjab->abij", I1, H.aa.oovv, optimize=True)

    LH.aa += np.einsum("eima,ebmj->abij", H.aa.voov, L.aa, optimize=True)
    LH.aa += np.einsum("ieam,bejm->abij", H.ab.ovvo, L.ab, optimize=True)

    LH.aa += 0.125 * np.einsum("ijmn,abmn->abij", H.aa.oooo, L.aa, optimize=True)
    LH.aa += 0.125 * np.einsum("efab,efij->abij", H.aa.vvvv, L.aa, optimize=True)

    LH.aa += 0.5 * np.einsum("ejab,ei->abij", H.aa.vovv, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ijmb,am->abij", H.aa.ooov, L.a, optimize=True)

    # < 0 | L3 * H(2) | ijab >
    LH.aa -= np.einsum("ejfb,fiea->abij", X.aa.vovv, H.aa.vovv, optimize=True)
    LH.aa -= np.einsum("njmb,mina->abij", X.aa.ooov, H.aa.ooov, optimize=True)
    LH.aa -= 0.25 * np.einsum("enab,jine->abij", X.aa.vovv, H.aa.ooov, optimize=True)
    LH.aa -= 0.25 * np.einsum("jine,enab->abij", X.aa.ooov, H.aa.vovv, optimize=True)
    LH.aa -= np.einsum("jebf,ifae->abij", X.ab.ovvv, H.ab.ovvv, optimize=True)
    LH.aa -= np.einsum("jnbm,iman->abij", X.ab.oovo, H.ab.oovo, optimize=True)

    # < 0 | L3 * (H(2) * T3) |
    LH.aa += np.einsum("ejmb,imae->abij", X.aa.voov, H.aa.oovv, optimize=True)
    LH.aa += np.einsum("jebm,imae->abij", X.ab.ovvo, H.ab.oovv, optimize=True)
    LH.aa += 0.125 * np.einsum("efab,ijef->abij", X.aa.vvvv, H.aa.oovv, optimize=True)
    LH.aa += 0.125 * np.einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv, optimize=True)

    LH.aa += 0.25 * np.einsum("ebfijn,fena->abij", L.aaa, H.aa.vvov, optimize=True)
    LH.aa += 0.5 * np.einsum("ebfijn,efan->abij", L.aab, H.ab.vvvo, optimize=True)
    LH.aa -= 0.25 * np.einsum("abfmjn,finm->abij", L.aaa, H.aa.vooo, optimize=True)
    LH.aa -= 0.5 * np.einsum("abfmjn,ifmn->abij", L.aab, H.ab.ovoo, optimize=True)

    LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))

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

    I1 = -0.5 * np.einsum("abij,fbij->fa", L.aa, T.aa, optimize=True)
    I2 = -np.einsum("afmn,efmn->ea", L.ab, T.ab, optimize=True)
    I3 = -np.einsum("fbnm,fenm->eb", L.ab, T.ab, optimize=True)
    I4 = -0.5 * np.einsum("bfmn,efmn->eb", L.bb, T.bb, optimize=True)
    LH.ab += np.einsum("fa,nmfe->aenm", I1, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("ea,ijeb->abij", I2, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("eb,ijae->abij", I3, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("eb,ijae->abij", I4, H.ab.oovv, optimize=True)

    I1 = -0.5 * np.einsum("efin,efmn->im", L.aa, T.aa, optimize=True)
    I2 = -np.einsum("efin,efmn->im", L.ab, T.ab, optimize=True)
    I3 = -np.einsum("fenj,fenm->jm", L.ab, T.ab, optimize=True)
    I4 = -0.5 * np.einsum("efjn,efmn->jm", L.bb, T.bb, optimize=True)
    LH.ab += np.einsum("im,mjab->abij", I1, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("im,mjab->abij", I2, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("jm,imab->abij", I3, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("jm,imab->abij", I4, H.ab.oovv, optimize=True)

    LH.ab += np.einsum("ea,ebij->abij", H.a.vv, L.ab, optimize=True)
    LH.ab += np.einsum("eb,aeij->abij", H.b.vv, L.ab, optimize=True)
    LH.ab -= np.einsum("im,abmj->abij", H.a.oo, L.ab, optimize=True)
    LH.ab -= np.einsum("jm,abim->abij", H.b.oo, L.ab, optimize=True)
    LH.ab += np.einsum("jb,ai->abij", H.b.ov, L.a, optimize=True)
    LH.ab += np.einsum("ia,bj->abij", H.a.ov, L.b, optimize=True)

    return LH


def build_LH_2C(L, LH, T, H):

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

    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))

    return LH

