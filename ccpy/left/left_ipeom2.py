import numpy as np

from ccpy.utilities.updates import cc_loops2


def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

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
        LH.ba = LH.ab.copy()
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, H)
        LH = build_LH_2D(L, LH, T, H)

    L.a, L.b, L.aa, L.ab, L.ba, L.bb,\
    LH.a, LH.b, LH.aa, LH.ab, LH.ba, LH.bb = cc_loops2.cc_loops2.update_l_2h1p(L.a, L.b, L.aa, L.ab, L.ba, L.bb,
                                                                               LH.a, LH.b, LH.aa, LH.ab, LH.ba, LH.bb,
                                                                               omega,
                                                                               H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                               shift)
    return L, LH


def build_LH_1A(L, LH, T, H):

    LH.a = -1.0 * np.einsum("m,im->i", L.a, H.a.oo, optimize=True)
    LH.a -= 0.5 * np.einsum("fnm,finm->i", L.aa, H.aa.vooo, optimize=True)
    LH.a -= np.einsum("fnm,ifmn->i", L.ba, H.ab.ovoo, optimize=True)

    return LH


def build_LH_1B(L, LH, T, H):

    LH.b = -1.0 * np.einsum("m,im->i", L.b, H.b.oo, optimize=True)
    LH.b -= 0.5 * np.einsum("fnm,finm->i", L.bb, H.bb.vooo, optimize=True)
    LH.b -= np.einsum("fnm,finm->i", L.ab, H.ab.vooo, optimize=True)

    return LH


def build_LH_2A(L, LH, T, H):

    LH.aa = np.einsum("i,jb->bji", L.a, H.a.ov, optimize=True)
    LH.aa -= 0.5 * np.einsum("m,ijmb->bji", L.a, H.aa.ooov, optimize=True)
    LH.aa += 0.5 * np.einsum("eji,eb->bji", L.aa, H.a.vv, optimize=True)
    LH.aa -= np.einsum("bmi,jm->bji", L.aa, H.a.oo, optimize=True)
    LH.aa += 0.25 * np.einsum("bnm,ijmn->bji", L.aa, H.aa.oooo, optimize=True)
    LH.aa += np.einsum("emi,ejmb->bji", L.aa, H.aa.voov, optimize=True)
    LH.aa += np.einsum("emi,jebm->bji", L.ba, H.ab.ovvo, optimize=True)

    I1 = (
        -0.5 * np.einsum("fnm,efmn->e", L.aa, T.aa, optimize=True)
        - np.einsum("fnm,efmn->e", L.ba, T.ab, optimize=True)
    )
    LH.aa += 0.5 * np.einsum("e,ijeb->bji", I1, H.aa.oovv, optimize=True)

    LH.aa -= np.transpose(LH.aa, (0, 2, 1))

    return LH


def build_LH_2B(L, LH, T, H):

    LH.ab = np.einsum("i,jb->bji", L.b, H.a.ov, optimize=True)
    LH.ab -= np.einsum("m,jibm->bji", L.b, H.ab.oovo, optimize=True)
    LH.ab -= np.einsum("bjm,im->bji", L.ab, H.b.oo, optimize=True)
    LH.ab -= np.einsum("bmi,jm->bji", L.ab, H.a.oo, optimize=True)
    LH.ab += np.einsum("eji,eb->bji", L.ab, H.a.vv, optimize=True)
    LH.ab += np.einsum("bmn,jimn->bji", L.ab, H.ab.oooo, optimize=True)
    LH.ab += np.einsum("emi,ejmb->bji", L.ab, H.aa.voov, optimize=True)
    LH.ab += np.einsum("emi,jebm->bji", L.bb, H.ab.ovvo, optimize=True)
    LH.ab -= np.einsum("ejm,eibm->bji", L.ab, H.ab.vovo, optimize=True)

    I1 = (
        -0.5 * np.einsum("fnm,efmn->e", L.bb, T.bb, optimize=True)
        - np.einsum("fnm,fenm->e", L.ab, T.ab, optimize=True)
    )
    LH.ab += np.einsum("e,jibe->bji", I1, H.ab.oovv, optimize=True)

    return LH


def build_LH_2C(L, LH, T, H):

    LH.ba = np.einsum("i,jb->bji", L.a, H.b.ov, optimize=True)
    LH.ba -= np.einsum("m,ijmb->bji", L.b, H.ab.ooov, optimize=True)
    LH.ba -= np.einsum("bmi,jm->bji", L.ba, H.b.oo, optimize=True)
    LH.ba -= np.einsum("bjm,im->bji", L.ba, H.a.oo, optimize=True)
    LH.ba += np.einsum("eji,eb->bji", L.ba, H.b.vv, optimize=True)
    LH.ba += np.einsum("bnm,ijmn->bji", L.ba, H.ab.oooo, optimize=True)
    LH.ba += np.einsum("emi,ejmb->bji", L.aa, H.ab.voov, optimize=True)
    LH.ba += np.einsum("emi,ejmb->bji", L.ba, H.bb.voov, optimize=True)
    LH.ba -= np.einsum("ejm,iemb->bji", L.ba, H.ab.ovov, optimize=True)

    I1 = (
        -0.5 * np.einsum("fnm,efmn->e", L.aa, T.aa, optimize=True)
        - np.einsum("fnm,efmn->e", L.ba, T.ab, optimize=True)
    )
    LH.ba += np.einsum("e,ijeb->bji", I1, H.ab.oovv, optimize=True)

    return LH


def build_LH_2D(L, LH, T, H):

    LH.bb = np.einsum("i,jb->bji", L.b, H.b.ov, optimize=True)
    LH.bb -= 0.5 * np.einsum("m,ijmb->bji", L.b, H.bb.ooov, optimize=True)
    LH.bb -= np.einsum("bjm,im->bji", L.bb, H.b.oo, optimize=True)
    LH.bb += 0.5 * np.einsum("eji,eb->bji", L.bb, H.b.vv, optimize=True)
    LH.bb += 0.25 * np.einsum("bnm,ijmn->bji", L.bb, H.bb.oooo, optimize=True)
    LH.bb += np.einsum("emi,ejmb->bji", L.ab, H.ab.voov, optimize=True)
    LH.bb += np.einsum("emi,ejmb->bji", L.bb, H.bb.voov, optimize=True)

    I1 = (
        -0.5 * np.einsum("fnm,efmn->e", L.bb, T.bb, optimize=True)
        - np.einsum("fnm,fenm->e", L.ab, T.ab, optimize=True)
    )
    LH.bb += 0.5 * np.einsum("e,jibe->bji", I1, H.bb.oovv, optimize=True)

    LH.bb -= np.transpose(LH.bb, (0, 2, 1))

    return LH
