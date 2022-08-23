import numpy as np

from ccpy.utilities.updates import cc_loops2
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
    LH = build_LH_3A(L, LH, T, H, X)
    LH = build_LH_3B(L, LH, T, H, X)
    if flag_RHF:
        LH.abb = np.transpose(LH.aab, (2, 0, 1, 5, 3, 4))
        LH.bbb = LH.aaa.copy()
    #else:
        #LH = build_LH_3C(L, LH, T, H, X)
        #LH = build_LH_3D(L, LH, T, H, X)

    # Add Hamiltonian if ground-state calculation
    if is_ground:
        LH.a += np.transpose(H.a.ov, (1, 0))
        LH.b += np.transpose(H.b.ov, (1, 0))
        LH.aa += np.transpose(H.aa.oovv, (2, 3, 0, 1))
        LH.ab += np.transpose(H.ab.oovv, (2, 3, 0, 1))
        LH.bb += np.transpose(H.bb.oovv, (2, 3, 0, 1))


    L.a, L.b, LH.a, LH.b = cc_loops2.cc_loops2.update_l1(L.a, L.b, LH.a, LH.b,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)
    L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb = cc_loops2.cc_loops2.update_l2(L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)
    L.aaa, L.aab, L.abb, L.bbb, LH.aaa, LH.aab, LH.abb, LH.bbb = cc_loops2.cc_loops2.update_l3(L.aaa, L.aab, L.abb, L.bbb, LH.aaa, LH.aab, LH.abb, LH.bbb,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)

    return L, LH

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

    # < 0 | L3 * H(2) | ia >

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
    LH.aa += 0.25 * np.einsum("ebfijn,fena->abij", L.aaa, H.aa.vvov, optimize=True) # 1
    LH.aa += 0.5 * np.einsum("ebfijn,efan->abij", L.aab, H.ab.vvvo, optimize=True) # 2
    LH.aa -= 0.25 * np.einsum("abfmjn,finm->abij", L.aaa, H.aa.vooo, optimize=True) # 3
    LH.aa -= 0.5 * np.einsum("abfmjn,ifmn->abij", L.aab, H.ab.ovoo, optimize=True) # 4

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
    LH.ab -= 0.5 * np.einsum("afbmnj,finm->abij", L.aab, H.aa.vooo, optimize=True) # 1
    LH.ab -= np.einsum("afbmnj,ifmn->abij", L.abb, H.ab.ovoo, optimize=True) # 2
    LH.ab -= np.einsum("afbinm,fjnm->abij", L.aab, H.ab.vooo, optimize=True) # 3
    LH.ab -= 0.5 * np.einsum("afbinm,fjnm->abij", L.abb, H.bb.vooo, optimize=True) # 4

    LH.ab += 0.5 * np.einsum("efbinj,fena->abij", L.aab, H.aa.vvov, optimize=True) # 5
    LH.ab += np.einsum("efbinj,efan->abij", L.abb, H.ab.vvvo, optimize=True) # 6
    LH.ab += np.einsum("afeinj,fenb->abij", L.aab, H.ab.vvov, optimize=True) # 7
    LH.ab += 0.5 * np.einsum("afeinj,fenb->abij", L.abb, H.bb.vvov, optimize=True) # 8

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

    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))

    return LH

def build_LH_3A(L, LH, T, H, X):

    # < 0 | L1 * H(2) | ijkabc >
    LH.aaa = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", L.a, H.aa.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    LH.aaa += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", L.aa, H.a.ov, optimize=True)

    LH.aaa += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv, optimize=True)
    LH.aaa -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov, optimize=True)

    # < 0 | L3 * H(2) | ijkabc >
    LH.aaa += (3.0 / 36.0) * np.einsum("ea,ebcijk->abcijk", H.a.vv, L.aaa, optimize=True)
    LH.aaa -= (3.0 / 36.0) * np.einsum("im,abcmjk->abcijk", H.a.oo, L.aaa, optimize=True)
    LH.aaa += (9.0 / 36.0) * np.einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa, optimize=True)
    LH.aaa += (9.0 / 36.0) * np.einsum("ieam,bcejkm->abcijk", H.ab.ovvo, L.aab, optimize=True)
    LH.aaa += (3.0 / 72.0) * np.einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aaa, optimize=True)
    LH.aaa += (3.0 / 72.0) * np.einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aaa, optimize=True)

    LH.aaa += (9.0 / 36.0) * np.einsum("ijeb,ekac->abcijk", H.aa.oovv, X.aa.vovv, optimize=True)
    LH.aaa -= (9.0 / 36.0) * np.einsum("mjab,ikmc->abcijk", H.aa.oovv, X.aa.ooov, optimize=True)

    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(LH.aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    LH.aaa -= np.transpose(LH.aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(LH.aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return LH


def build_LH_3B(L, LH, T, H, X):

    # < 0 | L1 * H(2) | ijk~abc~ >
    LH.aab = np.einsum("ai,jkbc->abcijk", L.a, H.ab.oovv, optimize=True)
    LH.aab += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.aa.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijk~abc~ >

    LH.aab += np.einsum("bcjk,ia->abcijk", L.ab, H.a.ov, optimize=True)
    LH.aab += 0.25 * np.einsum("abij,kc->abcijk", L.aa, H.b.ov, optimize=True)

    LH.aab += 0.5 * np.einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa, optimize=True)
    LH.aab -= 0.5 * np.einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa, optimize=True)
    LH.aab += np.einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab, optimize=True)
    LH.aab -= np.einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab, optimize=True)
    LH.aab += 0.5 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab, optimize=True)
    LH.aab -= 0.5 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab, optimize=True)

    LH.aab += 0.5 * np.einsum("ekbc,ijae->abcijk", X.ab.vovv, H.aa.oovv, optimize=True)
    LH.aab -= 0.5 * np.einsum("jkmc,imab->abcijk", X.ab.ooov, H.aa.oovv, optimize=True)
    LH.aab += np.einsum("ieac,jkbe->abcijk", X.ab.ovvv, H.ab.oovv, optimize=True)
    LH.aab -= np.einsum("ikam,jmbc->abcijk", X.ab.oovo, H.ab.oovv, optimize=True)
    LH.aab += 0.5 * np.einsum("eiba,jkec->abcijk", X.aa.vovv, H.ab.oovv, optimize=True)
    LH.aab -= 0.5 * np.einsum("jima,mkbc->abcijk", X.aa.ooov, H.ab.oovv, optimize=True)

    # < 0 | L3 * H(2) | ijk~abc~ >
    LH.aab -= 0.5 * np.einsum("im,abcmjk->abcijk", H.a.oo, L.aab, optimize=True)
    LH.aab -= 0.25 * np.einsum("km,abcijm->abcijk", H.b.oo, L.aab, optimize=True)
    LH.aab += 0.5 * np.einsum("ea,ebcijk->abcijk", H.a.vv, L.aab, optimize=True)
    LH.aab += 0.25 * np.einsum("ec,abeijk->abcijk", H.b.vv, L.aab, optimize=True)
    LH.aab += 0.125 * np.einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aab, optimize=True)
    LH.aab += 0.5 * np.einsum("jkmn,abcimn->abcijk", H.ab.oooo, L.aab, optimize=True)
    LH.aab += 0.125 * np.einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aab, optimize=True)
    LH.aab += 0.5 * np.einsum("efbc,aefijk->abcijk", H.ab.vvvv, L.aab, optimize=True)
    LH.aab += np.einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aab, optimize=True)
    LH.aab += np.einsum("ieam,becjmk->abcijk", H.ab.ovvo, L.abb, optimize=True)
    LH.aab += 0.25 * np.einsum("ekmc,abeijm->abcijk", H.ab.voov, L.aaa, optimize=True)
    LH.aab += 0.25 * np.einsum("ekmc,abeijm->abcijk", H.bb.voov, L.aab, optimize=True)
    LH.aab -= 0.5 * np.einsum("ekam,ebcijm->abcijk", H.ab.vovo, L.aab, optimize=True)
    LH.aab -= 0.5 * np.einsum("iemc,abemjk->abcijk", H.ab.ovov, L.aab, optimize=True)

    LH.aab -= np.transpose(LH.aab, (1, 0, 2, 3, 4, 5))
    LH.aab -= np.transpose(LH.aab, (0, 1, 2, 4, 3, 5))

    return LH

