import numpy as np

from ccpy.utilities.updates import cc_loops

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF):

    # build L1
    LH = build_LH_1A(L, LH, T, H)

    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, H)

    # build L2
    LH = build_LH_2A(L, LH, T, H)
    LH = build_LH_2B(L, LH, T, H)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H)

    # Add Hamiltonian if ground-state calculation
    if is_ground:
        LH.a += np.transpose(H.a.ov, (1, 0))
        LH.b += np.transpose(H.b.ov, (1, 0))
        LH.aa += np.transpose(H.aa.oovv, (2, 3, 0, 1))
        LH.ab += np.transpose(H.ab.oovv, (2, 3, 0, 1))
        LH.bb += np.transpose(H.bb.oovv, (2, 3, 0, 1))

    # update L; this function takes in LH and builds residual LH - omega * L
    # should make this so that it computes residual and returns in, just like in
    # ground-state CC functions
    L.a, L.b, L.aa, L.ab, L.bb = cc_loops.cc_loops.update_l(L.a, L.b, L.aa, L.ab, L.bb,
                                                            LH.a, LH.b, LH.aa, LH.ab, LH.bb,
                                                            omega,
                                                            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                            shift,
    )

    # build residual
    LH.a -= omega * L.a
    LH.b -= omega * L.b
    LH.aa -= omega * L.aa
    LH.ab -= omega * L.ab
    LH.bb -= omega * L.bb

    return L, LH


def build_LH_1A(L, LH, T, H):

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

    return LH


def build_LH_1B(L, LH, T, H):

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


def build_LH_2A(L, LH, T, H):

    LH.aa = np.einsum("ea,ebij->abij", H.a.vv, L.aa, optimize=True) - np.einsum(
        "eb,eaij->abij", H.a.vv, L.aa, optimize=True
    )
    LH.aa += -np.einsum("im,abmj->abij", H.a.oo, L.aa, optimize=True) + np.einsum(
        "jm,abmi->abij", H.a.oo, L.aa, optimize=True
    )
    LH.aa += (
        np.einsum("jb,ai->abij", H.a.ov, L.a, optimize=True)
        - np.einsum("ja,bi->abij", H.a.ov, L.a, optimize=True)
        - np.einsum("ib,aj->abij", H.a.ov, L.a, optimize=True)
        + np.einsum("ia,bj->abij", H.a.ov, L.a, optimize=True)
    )

    I1 = np.einsum("afmn,efmn->ea", L.aa, T.aa, optimize=True)
    I2 = np.einsum("bfmn,efmn->eb", L.aa, T.aa, optimize=True)
    LH.aa += -0.5 * np.einsum(
        "ea,ijeb->abij", I1, H.aa.oovv, optimize=True
    ) + 0.5 * np.einsum("eb,ijea->abij", I2, H.aa.oovv, optimize=True)

    I1 = np.einsum("afmn,efmn->ea", L.ab, T.ab, optimize=True)
    I2 = np.einsum("bfmn,efmn->eb", L.ab, T.ab, optimize=True)
    LH.aa += -np.einsum("ea,ijeb->abij", I1, H.aa.oovv, optimize=True) + np.einsum(
        "eb,ijea->abij", I2, H.aa.oovv, optimize=True
    )

    I1 = np.einsum("efin,efmn->im", L.aa, T.aa, optimize=True)
    I2 = np.einsum("efjn,efmn->jm", L.aa, T.aa, optimize=True)
    LH.aa += -0.5 * np.einsum(
        "im,mjab->abij", I1, H.aa.oovv, optimize=True
    ) + 0.5 * np.einsum("jm,miab->abij", I2, H.aa.oovv, optimize=True)

    I1 = np.einsum("efin,efmn->im", L.ab, T.ab, optimize=True)
    I2 = np.einsum("efjn,efmn->jm", L.ab, T.ab, optimize=True)
    LH.aa += -np.einsum("im,mjab->abij", I1, H.aa.oovv, optimize=True) + np.einsum(
        "jm,miab->abij", I2, H.aa.oovv, optimize=True
    )

    LH.aa += (
        np.einsum("eima,ebmj->abij", H.aa.voov, L.aa, optimize=True)
        - np.einsum("ejma,ebmi->abij", H.aa.voov, L.aa, optimize=True)
        - np.einsum("eimb,eamj->abij", H.aa.voov, L.aa, optimize=True)
        + np.einsum("ejmb,eami->abij", H.aa.voov, L.aa, optimize=True)
    )

    LH.aa += (
        +np.einsum("ieam,bejm->abij", H.ab.ovvo, L.ab, optimize=True)
        - np.einsum("jeam,beim->abij", H.ab.ovvo, L.ab, optimize=True)
        - np.einsum("iebm,aejm->abij", H.ab.ovvo, L.ab, optimize=True)
        + np.einsum("jebm,aeim->abij", H.ab.ovvo, L.ab, optimize=True)
    )

    LH.aa += 0.5 * np.einsum("ijmn,abmn->abij", H.aa.oooo, L.aa, optimize=True)
    LH.aa += +0.5 * np.einsum("efab,efij->abij", H.aa.vvvv, L.aa, optimize=True)
    LH.aa += np.einsum("ejab,ei->abij", H.aa.vovv, L.a, optimize=True) - np.einsum(
        "eiab,ej->abij", H.aa.vovv, L.a, optimize=True
    )
    LH.aa += -np.einsum("ijmb,am->abij", H.aa.ooov, L.a, optimize=True) + np.einsum(
        "ijma,bm->abij", H.aa.ooov, L.a, optimize=True
    )

    return LH


def build_LH_2B(L, LH, T, H):

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

    LH.bb = np.einsum("ea,ebij->abij", H.b.vv, L.bb, optimize=True)
    LH.bb -= np.einsum("eb,eaij->abij", H.b.vv, L.bb, optimize=True)
    LH.bb -= np.einsum("im,abmj->abij", H.b.oo, L.bb, optimize=True)
    LH.bb += np.einsum("jm,abmi->abij", H.b.oo, L.bb, optimize=True)
    LH.bb -= np.einsum("ijmb,am->abij", H.bb.ooov, L.b, optimize=True)
    LH.bb += np.einsum("ijma,bm->abij", H.bb.ooov, L.b, optimize=True)
    LH.bb += np.einsum("ejab,ei->abij", H.bb.vovv, L.b, optimize=True)
    LH.bb -= np.einsum("eiab,ej->abij", H.bb.vovv, L.b, optimize=True)

    LH.bb += 0.5 * np.einsum("efab,efij->abij", H.bb.vvvv, L.bb, optimize=True)
    LH.bb += 0.5 * np.einsum("ijmn,abmn->abij", H.bb.oooo, L.bb, optimize=True)

    LH.bb += np.einsum("ejmb,aeim->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb -= np.einsum("eimb,aejm->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb -= np.einsum("ejma,beim->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb += np.einsum("eima,bejm->abij", H.bb.voov, L.bb, optimize=True)

    LH.bb += np.einsum("ejmb,eami->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb -= np.einsum("eimb,eamj->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb -= np.einsum("ejma,ebmi->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.ab.voov, L.ab, optimize=True)

    I1 = np.einsum("fanm,fenm->ea", L.ab, T.ab, optimize=True)
    I2 = np.einsum("fbnm,fenm->eb", L.ab, T.ab, optimize=True)
    LH.bb -= np.einsum("ea,ijeb->abij", I1, H.bb.oovv, optimize=True)
    LH.bb += np.einsum("eb,ijea->abij", I2, H.bb.oovv, optimize=True)

    I1 = np.einsum("afmn,efmn->ea", L.bb, T.bb, optimize=True)
    I2 = np.einsum("bfmn,efmn->eb", L.bb, T.bb, optimize=True)
    LH.bb -= 0.5 * np.einsum("ea,ijeb->abij", I1, H.bb.oovv, optimize=True)
    LH.bb += 0.5 * np.einsum("eb,ijea->abij", I2, H.bb.oovv, optimize=True)

    I1 = np.einsum("feni,fenm->im", L.ab, T.ab, optimize=True)
    I2 = np.einsum("fenj,fenm->jm", L.ab, T.ab, optimize=True)
    LH.bb -= np.einsum("im,mjab->abij", I1, H.bb.oovv, optimize=True)
    LH.bb += np.einsum("jm,miab->abij", I2, H.bb.oovv, optimize=True)

    I1 = np.einsum("efin,efmn->im", L.bb, T.bb, optimize=True)
    I2 = np.einsum("efjn,efmn->jm", L.bb, T.bb, optimize=True)
    LH.bb -= 0.5 * np.einsum("im,mjab->abij", I1, H.bb.oovv, optimize=True)
    LH.bb += 0.5 * np.einsum("jm,miab->abij", I2, H.bb.oovv, optimize=True)

    LH.bb += np.einsum("jb,ai->abij", H.b.ov, L.b, optimize=True)
    LH.bb -= np.einsum("ib,aj->abij", H.b.ov, L.b, optimize=True)
    LH.bb -= np.einsum("ja,bi->abij", H.b.ov, L.b, optimize=True)
    LH.bb += np.einsum("ia,bj->abij", H.b.ov, L.b, optimize=True)

    return LH

