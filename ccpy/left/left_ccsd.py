import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2
from ccpy.left.left_cc_intermediates import build_left_ccsd_intermediates

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

    # get LT intermediates
    X = build_left_ccsd_intermediates(L, T, system)

    # build L1
    LH = build_LH_1A(L, LH, T, X, H)

    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, X, H)

    # build L2
    LH = build_LH_2A(L, LH, T, X, H)
    LH = build_LH_2B(L, LH, T, X, H)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, X, H)

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

    L.a, L.b, L.aa, L.ab, L.bb = cc_loops2.update_r(
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
    X = build_left_ccsd_intermediates(L, T, system)

    # build L1
    LH = build_LH_1A(L, LH, T, X, H)
    if flag_RHF:
        LH.b = LH.a.copy()
    else:
        LH = build_LH_1B(L, LH, T, X, H)
    # build L2
    LH = build_LH_2A(L, LH, T, X, H)
    LH = build_LH_2B(L, LH, T, X, H)
    if flag_RHF:
        LH.bb = LH.aa.copy()
    else:
        LH = build_LH_2C(L, LH, T, X, H)
    return LH.flatten()

def build_LH_1A(L, LH, T, X, H):

    LH.a = ccpy_einsum("ea,ei->ai", H.a.vv, L.a)
    LH.a -= ccpy_einsum("im,am->ai", H.a.oo, L.a)
    LH.a += ccpy_einsum("eima,em->ai", H.aa.voov, L.a)
    LH.a += ccpy_einsum("ieam,em->ai", H.ab.ovvo, L.b)
    LH.a += 0.5 * ccpy_einsum("fena,efin->ai", H.aa.vvov, L.aa)
    LH.a += ccpy_einsum("efan,efin->ai", H.ab.vvvo, L.ab)
    LH.a -= 0.5 * ccpy_einsum("finm,afmn->ai", H.aa.vooo, L.aa)
    LH.a -= ccpy_einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab)
    LH.a += ccpy_einsum("ge,eiga->ai", X.a.vv, H.aa.vovv)
    LH.a += ccpy_einsum("mn,nima->ai", X.a.oo, H.aa.ooov)
    LH.a += ccpy_einsum("fa,maef->em", X.b.vv, H.ab.ovvv)
    LH.a += ccpy_einsum("in,mnei->em", X.b.oo, H.ab.oovo)
    return LH


def build_LH_1B(L, LH, T, X, H):

    LH.b = ccpy_einsum("ea,ei->ai", H.b.vv, L.b)
    LH.b -= ccpy_einsum("im,am->ai", H.b.oo, L.b)
    LH.b += ccpy_einsum("eima,em->ai", H.ab.voov, L.a)
    LH.b += ccpy_einsum("eima,em->ai", H.bb.voov, L.b)
    LH.b -= 0.5 * ccpy_einsum("finm,afmn->ai", H.bb.vooo, L.bb)
    LH.b -= ccpy_einsum("finm,fanm->ai", H.ab.vooo, L.ab)
    LH.b += ccpy_einsum("fena,feni->ai", H.ab.vvov, L.ab)
    LH.b += 0.5 * ccpy_einsum("fena,efin->ai", H.bb.vvov, L.bb)
    LH.b += (
        ccpy_einsum("ge,eiga->ai", X.b.vv, H.bb.vovv)
        + ccpy_einsum("mo,oima->ai", X.b.oo, H.bb.ooov)
    )
    LH.b += (
        ccpy_einsum("ge,eiga->ai", X.a.vv, H.ab.vovv)
        + ccpy_einsum("mo,oima->ai", X.a.oo, H.ab.ooov)
    )
    return LH

def build_LH_2A(L, LH, T, X, H):
    LH.aa = 0.5 * ccpy_einsum("ea,ebij->abij", H.a.vv, L.aa)
    LH.aa -= 0.5 * ccpy_einsum("im,abmj->abij", H.a.oo, L.aa)
    LH.aa += ccpy_einsum("jb,ai->abij", H.a.ov, L.a)
    LH.aa -= 0.5 * ccpy_einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv)
    LH.aa += 0.5 * ccpy_einsum("im,mjab->abij", X.a.oo, H.aa.oovv)
    LH.aa += ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.aa)
    LH.aa += ccpy_einsum("ieam,bejm->abij", H.ab.ovvo, L.ab)
    LH.aa += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.aa.oooo, L.aa)
    LH.aa += 0.125 * ccpy_einsum("efab,efij->abij", H.aa.vvvv, L.aa)
    LH.aa += 0.5 * ccpy_einsum("ejab,ei->abij", H.aa.vovv, L.a)
    LH.aa -= 0.5 * ccpy_einsum("ijmb,am->abij", H.aa.ooov, L.a)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))
    return LH


def build_LH_2B(L, LH, T, X, H):
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
    LH.ab -= ccpy_einsum("ea,ijeb->abij", X.a.vv, H.ab.oovv)
    LH.ab += ccpy_einsum("im,mjab->abij", X.a.oo, H.ab.oovv)
    LH.ab -= ccpy_einsum("ea,jibe->baji", X.b.vv, H.ab.oovv)
    LH.ab += ccpy_einsum("im,jmba->baji", X.b.oo, H.ab.oovv)
    LH.ab += ccpy_einsum("ea,ebij->abij", H.a.vv, L.ab)
    LH.ab += ccpy_einsum("eb,aeij->abij", H.b.vv, L.ab)
    LH.ab -= ccpy_einsum("im,abmj->abij", H.a.oo, L.ab)
    LH.ab -= ccpy_einsum("jm,abim->abij", H.b.oo, L.ab)
    LH.ab += ccpy_einsum("jb,ai->abij", H.b.ov, L.a)
    LH.ab += ccpy_einsum("ia,bj->abij", H.a.ov, L.b)
    return LH

def build_LH_2C(L, LH, T, X, H):
    LH.bb = 0.5 * ccpy_einsum("ea,ebij->abij", H.b.vv, L.bb)
    LH.bb -= 0.5 * ccpy_einsum("im,abmj->abij", H.b.oo, L.bb)
    LH.bb += ccpy_einsum("jb,ai->abij", H.b.ov, L.b)
    LH.bb -= 0.5 * ccpy_einsum("ea,ijeb->abij", X.b.vv, H.bb.oovv)
    LH.bb += 0.5 * ccpy_einsum("im,mjab->abij", X.b.oo, H.bb.oovv)
    LH.bb += ccpy_einsum("eima,ebmj->abij", H.bb.voov, L.bb)
    LH.bb += ccpy_einsum("eima,ebmj->abij", H.ab.voov, L.ab)
    LH.bb += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.bb.oooo, L.bb)
    LH.bb += 0.125 * ccpy_einsum("efab,efij->abij", H.bb.vvvv, L.bb)
    LH.bb += 0.5 * ccpy_einsum("ejab,ei->abij", H.bb.vovv, L.b)
    LH.bb -= 0.5 * ccpy_einsum("ijmb,am->abij", H.bb.ooov, L.b)
    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))
    return LH

# def build_LH_1A(L, LH, T, H):
#
#     LH.a = ccpy_einsum("ea,ei->ai", H.a.vv, L.a)
#     LH.a -= ccpy_einsum("im,am->ai", H.a.oo, L.a)
#     LH.a += ccpy_einsum("eima,em->ai", H.aa.voov, L.a)
#     LH.a += ccpy_einsum("ieam,em->ai", H.ab.ovvo, L.b)
#     LH.a += 0.5 * ccpy_einsum("fena,efin->ai", H.aa.vvov, L.aa)
#     LH.a += ccpy_einsum("efan,efin->ai", H.ab.vvvo, L.ab)
#     LH.a -= 0.5 * ccpy_einsum("finm,afmn->ai", H.aa.vooo, L.aa)
#     LH.a -= ccpy_einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab)
#
#     I1 = 0.25 * ccpy_einsum("efmn,fgnm->ge", L.aa, T.aa)
#     I2 = -0.25 * ccpy_einsum("efmn,egnm->gf", L.aa, T.aa)
#     I3 = -0.25 * ccpy_einsum("efmo,efno->mn", L.aa, T.aa)
#     I4 = 0.25 * ccpy_einsum("efmo,efnm->on", L.aa, T.aa)
#
#     LH.a += ccpy_einsum("ge,eiga->ai", I1, H.aa.vovv)
#     LH.a += ccpy_einsum("gf,figa->ai", I2, H.aa.vovv)
#     LH.a += ccpy_einsum("mn,nima->ai", I3, H.aa.ooov)
#     LH.a += ccpy_einsum("on,nioa->ai", I4, H.aa.ooov)
#
#     I1 = -ccpy_einsum("abij,abin->jn", L.ab, T.ab)
#     I2 = ccpy_einsum("abij,afij->fb", L.ab, T.ab)
#     I3 = ccpy_einsum("abij,fbij->fa", L.ab, T.ab)
#     I4 = -ccpy_einsum("abij,abnj->in", L.ab, T.ab)
#
#     LH.a += ccpy_einsum("jn,mnej->em", I1, H.ab.oovo)
#     LH.a += ccpy_einsum("fb,mbef->em", I2, H.ab.ovvv)
#     LH.a += ccpy_einsum("fa,amfe->em", I3, H.aa.vovv)
#     LH.a += ccpy_einsum("in,nmie->em", I4, H.aa.ooov)
#
#     I1 = 0.25 * ccpy_einsum("abij,fbij->fa", L.bb, T.bb)
#     I2 = -0.25 * ccpy_einsum("abij,faij->fb", L.bb, T.bb)
#     I3 = -0.25 * ccpy_einsum("abij,abnj->in", L.bb, T.bb)
#     I4 = 0.25 * ccpy_einsum("abij,abni->jn", L.bb, T.bb)
#
#     LH.a += ccpy_einsum("fa,maef->em", I1, H.ab.ovvv)
#     LH.a += ccpy_einsum("fb,mbef->em", I2, H.ab.ovvv)
#     LH.a += ccpy_einsum("in,mnei->em", I3, H.ab.oovo)
#     LH.a += ccpy_einsum("jn,mnej->em", I4, H.ab.oovo)
#
#     return LH
#
#
# def build_LH_1B(L, LH, T, H):
#
#     LH.b = ccpy_einsum("ea,ei->ai", H.b.vv, L.b)
#     LH.b -= ccpy_einsum("im,am->ai", H.b.oo, L.b)
#     LH.b += ccpy_einsum("eima,em->ai", H.ab.voov, L.a)
#     LH.b += ccpy_einsum("eima,em->ai", H.bb.voov, L.b)
#     LH.b -= 0.5 * ccpy_einsum("finm,afmn->ai", H.bb.vooo, L.bb)
#     LH.b -= ccpy_einsum("finm,fanm->ai", H.ab.vooo, L.ab)
#     LH.b += ccpy_einsum("fena,feni->ai", H.ab.vvov, L.ab)
#     LH.b += 0.5 * ccpy_einsum("fena,efin->ai", H.bb.vvov, L.bb)
#
#     I1 = 0.25 * ccpy_einsum("efmn,fgnm->ge", L.bb, T.bb)
#     I2 = -0.25 * ccpy_einsum("efmn,egnm->gf", L.bb, T.bb)
#     I3 = -0.25 * ccpy_einsum("efmn,efon->mo", L.bb, T.bb)
#     I4 = 0.25 * ccpy_einsum("efmn,efom->no", L.bb, T.bb)
#     LH.b += (
#         ccpy_einsum("ge,eiga->ai", I1, H.bb.vovv)
#         + ccpy_einsum("gf,figa->ai", I2, H.bb.vovv)
#         + ccpy_einsum("mo,oima->ai", I3, H.bb.ooov)
#         + ccpy_einsum("no,oina->ai", I4, H.bb.ooov)
#     )
#
#     I1 = 0.25 * ccpy_einsum("efmn,fgnm->ge", L.aa, T.aa)
#     I2 = -0.25 * ccpy_einsum("efmn,egnm->gf", L.aa, T.aa)
#     I3 = -0.25 * ccpy_einsum("efmn,efon->mo", L.aa, T.aa)
#     I4 = 0.25 * ccpy_einsum("efmn,efom->no", L.aa, T.aa)
#     LH.b += (
#         ccpy_einsum("ge,eiga->ai", I1, H.ab.vovv)
#         + ccpy_einsum("gf,figa->ai", I2, H.ab.vovv)
#         + ccpy_einsum("mo,oima->ai", I3, H.ab.ooov)
#         + ccpy_einsum("no,oina->ai", I4, H.ab.ooov)
#     )
#
#     I1 = ccpy_einsum("efmn,gfmn->ge", L.ab, T.ab)
#     I2 = ccpy_einsum("fenm,fgnm->ge", L.ab, T.ab)
#     I3 = -ccpy_einsum("efmn,efon->mo", L.ab, T.ab)
#     I4 = -ccpy_einsum("fenm,feno->mo", L.ab, T.ab)
#     LH.b += (
#         ccpy_einsum("ge,eiga->ai", I1, H.ab.vovv)
#         + ccpy_einsum("ge,eiga->ai", I2, H.bb.vovv)
#         + ccpy_einsum("mo,oima->ai", I3, H.ab.ooov)
#         + ccpy_einsum("mo,oima->ai", I4, H.bb.ooov)
#     )
#
#     return LH
#
#
# # def build_LH_2A(L, LH, T, H):
# #
# #     LH.aa = ccpy_einsum("ea,ebij->abij", H.a.vv, L.aa) - np.einsum(
# #         "eb,eaij->abij", H.a.vv, L.aa, optimize=True
# #     )
# #     LH.aa += -ccpy_einsum("im,abmj->abij", H.a.oo, L.aa) + np.einsum(
# #         "jm,abmi->abij", H.a.oo, L.aa, optimize=True
# #     )
# #     LH.aa += (
# #         ccpy_einsum("jb,ai->abij", H.a.ov, L.a)
# #         - ccpy_einsum("ja,bi->abij", H.a.ov, L.a)
# #         - ccpy_einsum("ib,aj->abij", H.a.ov, L.a)
# #         + ccpy_einsum("ia,bj->abij", H.a.ov, L.a)
# #     )
# #
# #     I1 = ccpy_einsum("afmn,efmn->ea", L.aa, T.aa)
# #     I2 = ccpy_einsum("bfmn,efmn->eb", L.aa, T.aa)
# #     LH.aa += -0.5 * np.einsum(
# #         "ea,ijeb->abij", I1, H.aa.oovv, optimize=True
# #     ) + 0.5 * ccpy_einsum("eb,ijea->abij", I2, H.aa.oovv)
# #
# #     I1 = ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
# #     I2 = ccpy_einsum("bfmn,efmn->eb", L.ab, T.ab)
# #     LH.aa += -ccpy_einsum("ea,ijeb->abij", I1, H.aa.oovv) + np.einsum(
# #         "eb,ijea->abij", I2, H.aa.oovv, optimize=True
# #     )
# #
# #     I1 = ccpy_einsum("efin,efmn->im", L.aa, T.aa)
# #     I2 = ccpy_einsum("efjn,efmn->jm", L.aa, T.aa)
# #     LH.aa += -0.5 * np.einsum(
# #         "im,mjab->abij", I1, H.aa.oovv, optimize=True
# #     ) + 0.5 * ccpy_einsum("jm,miab->abij", I2, H.aa.oovv)
# #
# #     I1 = ccpy_einsum("efin,efmn->im", L.ab, T.ab)
# #     I2 = ccpy_einsum("efjn,efmn->jm", L.ab, T.ab)
# #     LH.aa += -ccpy_einsum("im,mjab->abij", I1, H.aa.oovv) + np.einsum(
# #         "jm,miab->abij", I2, H.aa.oovv, optimize=True
# #     )
# #
# #     LH.aa += (
# #         ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.aa)
# #         - ccpy_einsum("ejma,ebmi->abij", H.aa.voov, L.aa)
# #         - ccpy_einsum("eimb,eamj->abij", H.aa.voov, L.aa)
# #         + ccpy_einsum("ejmb,eami->abij", H.aa.voov, L.aa)
# #     )
# #
# #     LH.aa += (
# #         +ccpy_einsum("ieam,bejm->abij", H.ab.ovvo, L.ab)
# #         - ccpy_einsum("jeam,beim->abij", H.ab.ovvo, L.ab)
# #         - ccpy_einsum("iebm,aejm->abij", H.ab.ovvo, L.ab)
# #         + ccpy_einsum("jebm,aeim->abij", H.ab.ovvo, L.ab)
# #     )
# #
# #     LH.aa += 0.5 * ccpy_einsum("ijmn,abmn->abij", H.aa.oooo, L.aa)
# #     LH.aa += +0.5 * ccpy_einsum("efab,efij->abij", H.aa.vvvv, L.aa)
# #     LH.aa += ccpy_einsum("ejab,ei->abij", H.aa.vovv, L.a) - np.einsum(
# #         "eiab,ej->abij", H.aa.vovv, L.a, optimize=True
# #     )
# #     LH.aa += -ccpy_einsum("ijmb,am->abij", H.aa.ooov, L.a) + np.einsum(
# #         "ijma,bm->abij", H.aa.ooov, L.a, optimize=True
# #     )
# #
# #     return LH
#
# def build_LH_2A(L, LH, T, H):
#
#     LH.aa = 0.5 * ccpy_einsum("ea,ebij->abij", H.a.vv, L.aa)
#     LH.aa -= 0.5 * ccpy_einsum("im,abmj->abij", H.a.oo, L.aa)
#
#     LH.aa += ccpy_einsum("jb,ai->abij", H.a.ov, L.a)
#
#     I1 = (
#           -0.5 * ccpy_einsum("afmn,efmn->ea", L.aa, T.aa)
#           - ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
#     )
#     LH.aa += 0.5 * ccpy_einsum("ea,ijeb->abij", I1, H.aa.oovv)
#
#     I1 = (
#           0.5 * ccpy_einsum("efin,efmn->im", L.aa, T.aa)
#           + ccpy_einsum("efin,efmn->im", L.ab, T.ab)
#     )
#     LH.aa -= 0.5 * ccpy_einsum("im,mjab->abij", I1, H.aa.oovv)
#
#     LH.aa += ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.aa)
#     LH.aa += ccpy_einsum("ieam,bejm->abij", H.ab.ovvo, L.ab)
#
#     LH.aa += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.aa.oooo, L.aa)
#     LH.aa += 0.125 * ccpy_einsum("efab,efij->abij", H.aa.vvvv, L.aa)
#
#     LH.aa += 0.5 * ccpy_einsum("ejab,ei->abij", H.aa.vovv, L.a)
#     LH.aa -= 0.5 * ccpy_einsum("ijmb,am->abij", H.aa.ooov, L.a)
#
#     LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))
#
#     return LH
#
#
# def build_LH_2B(L, LH, T, H):
#
#     LH.ab = -ccpy_einsum("ijmb,am->abij", H.ab.ooov, L.a)
#     LH.ab -= ccpy_einsum("ijam,bm->abij", H.ab.oovo, L.b)
#
#     LH.ab += ccpy_einsum("ejab,ei->abij", H.ab.vovv, L.a)
#     LH.ab += ccpy_einsum("ieab,ej->abij", H.ab.ovvv, L.b)
#
#     LH.ab += ccpy_einsum("ijmn,abmn->abij", H.ab.oooo, L.ab)
#     LH.ab += ccpy_einsum("efab,efij->abij", H.ab.vvvv, L.ab)
#
#     LH.ab += ccpy_einsum("ejmb,aeim->abij", H.ab.voov, L.aa)
#     LH.ab += ccpy_einsum("eima,ebmj->abij", H.aa.voov, L.ab)
#     LH.ab += ccpy_einsum("ejmb,aeim->abij", H.bb.voov, L.ab)
#     LH.ab += ccpy_einsum("ieam,ebmj->abij", H.ab.ovvo, L.bb)
#     LH.ab -= ccpy_einsum("iemb,aemj->abij", H.ab.ovov, L.ab)
#     LH.ab -= ccpy_einsum("ejam,ebim->abij", H.ab.vovo, L.ab)
#
#     # I1 = -0.5 * ccpy_einsum("abij,fbij->fa", L.aa, T.aa)
#     # I2 = -ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
#     # I3 = -ccpy_einsum("fbnm,fenm->eb", L.ab, T.ab)
#     # I4 = -0.5 * ccpy_einsum("bfmn,efmn->eb", L.bb, T.bb)
#     # LH.ab += ccpy_einsum("fa,nmfe->aenm", I1, H.ab.oovv)
#     # LH.ab += ccpy_einsum("ea,ijeb->abij", I2, H.ab.oovv)
#     # LH.ab += ccpy_einsum("eb,ijae->abij", I3, H.ab.oovv)
#     # LH.ab += ccpy_einsum("eb,ijae->abij", I4, H.ab.oovv)
#
#     I1 = (
#           -0.5 * ccpy_einsum("afmn,efmn->ea", L.aa, T.aa)
#           - ccpy_einsum("afmn,efmn->ea", L.ab, T.ab)
#     )
#     LH.ab += ccpy_einsum("ea,ijeb->abij", I1, H.ab.oovv)
#
#     I1 = (
#           0.5 * ccpy_einsum("efin,efmn->im", L.aa, T.aa)
#           + ccpy_einsum("efin,efmn->im", L.ab, T.ab)
#     )
#     LH.ab -= ccpy_einsum("im,mjab->abij", I1, H.ab.oovv)
#
#     I1 = (
#           -0.5 * ccpy_einsum("afmn,efmn->ea", L.bb, T.bb)
#           - ccpy_einsum("fanm,fenm->ea", L.ab, T.ab)
#     )
#     LH.ab += ccpy_einsum("ea,jibe->baji", I1, H.ab.oovv)
#
#     I1 = (
#           0.5 * ccpy_einsum("efin,efmn->im", L.bb, T.bb)
#           + ccpy_einsum("feni,fenm->im", L.ab, T.ab)
#     )
#     LH.ab -= ccpy_einsum("im,jmba->baji", I1, H.ab.oovv)
#
#     # I1 = -0.5 * ccpy_einsum("efin,efmn->im", L.aa, T.aa)
#     # I2 = -ccpy_einsum("efin,efmn->im", L.ab, T.ab)
#     # I3 = -ccpy_einsum("fenj,fenm->jm", L.ab, T.ab)
#     # I4 = -0.5 * ccpy_einsum("efjn,efmn->jm", L.bb, T.bb)
#     # LH.ab += ccpy_einsum("im,mjab->abij", I1, H.ab.oovv)
#     # LH.ab += ccpy_einsum("im,mjab->abij", I2, H.ab.oovv)
#     # LH.ab += ccpy_einsum("jm,imab->abij", I3, H.ab.oovv)
#     # LH.ab += ccpy_einsum("jm,imab->abij", I4, H.ab.oovv)
#
#     LH.ab += ccpy_einsum("ea,ebij->abij", H.a.vv, L.ab)
#     LH.ab += ccpy_einsum("eb,aeij->abij", H.b.vv, L.ab)
#     LH.ab -= ccpy_einsum("im,abmj->abij", H.a.oo, L.ab)
#     LH.ab -= ccpy_einsum("jm,abim->abij", H.b.oo, L.ab)
#     LH.ab += ccpy_einsum("jb,ai->abij", H.b.ov, L.a)
#     LH.ab += ccpy_einsum("ia,bj->abij", H.a.ov, L.b)
#
#     return LH
#
#
# # def build_LH_2C(L, LH, T, H):
# #
# #     LH.bb = ccpy_einsum("ea,ebij->abij", H.b.vv, L.bb)
# #     LH.bb -= ccpy_einsum("eb,eaij->abij", H.b.vv, L.bb)
# #     LH.bb -= ccpy_einsum("im,abmj->abij", H.b.oo, L.bb)
# #     LH.bb += ccpy_einsum("jm,abmi->abij", H.b.oo, L.bb)
# #     LH.bb -= ccpy_einsum("ijmb,am->abij", H.bb.ooov, L.b)
# #     LH.bb += ccpy_einsum("ijma,bm->abij", H.bb.ooov, L.b)
# #     LH.bb += ccpy_einsum("ejab,ei->abij", H.bb.vovv, L.b)
# #     LH.bb -= ccpy_einsum("eiab,ej->abij", H.bb.vovv, L.b)
# #
# #     LH.bb += 0.5 * ccpy_einsum("efab,efij->abij", H.bb.vvvv, L.bb)
# #     LH.bb += 0.5 * ccpy_einsum("ijmn,abmn->abij", H.bb.oooo, L.bb)
# #
# #     LH.bb += ccpy_einsum("ejmb,aeim->abij", H.bb.voov, L.bb)
# #     LH.bb -= ccpy_einsum("eimb,aejm->abij", H.bb.voov, L.bb)
# #     LH.bb -= ccpy_einsum("ejma,beim->abij", H.bb.voov, L.bb)
# #     LH.bb += ccpy_einsum("eima,bejm->abij", H.bb.voov, L.bb)
# #
# #     LH.bb += ccpy_einsum("ejmb,eami->abij", H.ab.voov, L.ab)
# #     LH.bb -= ccpy_einsum("eimb,eamj->abij", H.ab.voov, L.ab)
# #     LH.bb -= ccpy_einsum("ejma,ebmi->abij", H.ab.voov, L.ab)
# #     LH.bb += ccpy_einsum("eima,ebmj->abij", H.ab.voov, L.ab)
# #
# #     I1 = ccpy_einsum("fanm,fenm->ea", L.ab, T.ab)
# #     I2 = ccpy_einsum("fbnm,fenm->eb", L.ab, T.ab)
# #     LH.bb -= ccpy_einsum("ea,ijeb->abij", I1, H.bb.oovv)
# #     LH.bb += ccpy_einsum("eb,ijea->abij", I2, H.bb.oovv)
# #
# #     I1 = ccpy_einsum("afmn,efmn->ea", L.bb, T.bb)
# #     I2 = ccpy_einsum("bfmn,efmn->eb", L.bb, T.bb)
# #     LH.bb -= 0.5 * ccpy_einsum("ea,ijeb->abij", I1, H.bb.oovv)
# #     LH.bb += 0.5 * ccpy_einsum("eb,ijea->abij", I2, H.bb.oovv)
# #
# #     I1 = ccpy_einsum("feni,fenm->im", L.ab, T.ab)
# #     I2 = ccpy_einsum("fenj,fenm->jm", L.ab, T.ab)
# #     LH.bb -= ccpy_einsum("im,mjab->abij", I1, H.bb.oovv)
# #     LH.bb += ccpy_einsum("jm,miab->abij", I2, H.bb.oovv)
# #
# #     I1 = ccpy_einsum("efin,efmn->im", L.bb, T.bb)
# #     I2 = ccpy_einsum("efjn,efmn->jm", L.bb, T.bb)
# #     LH.bb -= 0.5 * ccpy_einsum("im,mjab->abij", I1, H.bb.oovv)
# #     LH.bb += 0.5 * ccpy_einsum("jm,miab->abij", I2, H.bb.oovv)
# #
# #     LH.bb += ccpy_einsum("jb,ai->abij", H.b.ov, L.b)
# #     LH.bb -= ccpy_einsum("ib,aj->abij", H.b.ov, L.b)
# #     LH.bb -= ccpy_einsum("ja,bi->abij", H.b.ov, L.b)
# #     LH.bb += ccpy_einsum("ia,bj->abij", H.b.ov, L.b)
# #
# #     return LH
#
# def build_LH_2C(L, LH, T, H):
#
#     LH.bb = 0.5 * ccpy_einsum("ea,ebij->abij", H.b.vv, L.bb)
#     LH.bb -= 0.5 * ccpy_einsum("im,abmj->abij", H.b.oo, L.bb)
#
#     LH.bb += ccpy_einsum("jb,ai->abij", H.b.ov, L.b)
#
#     I1 = (
#           -0.5 * ccpy_einsum("afmn,efmn->ea", L.bb, T.bb)
#           - ccpy_einsum("fanm,fenm->ea", L.ab, T.ab)
#     )
#     LH.bb += 0.5 * ccpy_einsum("ea,ijeb->abij", I1, H.bb.oovv)
#
#     I1 = (
#           0.5 * ccpy_einsum("efin,efmn->im", L.bb, T.bb)
#           + ccpy_einsum("feni,fenm->im", L.ab, T.ab)
#     )
#     LH.bb -= 0.5 * ccpy_einsum("im,mjab->abij", I1, H.bb.oovv)
#
#     LH.bb += ccpy_einsum("eima,ebmj->abij", H.bb.voov, L.bb)
#     LH.bb += ccpy_einsum("eima,ebmj->abij", H.ab.voov, L.ab)
#
#     LH.bb += 0.125 * ccpy_einsum("ijmn,abmn->abij", H.bb.oooo, L.bb)
#     LH.bb += 0.125 * ccpy_einsum("efab,efij->abij", H.bb.vvvv, L.bb)
#
#     LH.bb += 0.5 * ccpy_einsum("ejab,ei->abij", H.bb.vovv, L.b)
#     LH.bb -= 0.5 * ccpy_einsum("ijmb,am->abij", H.bb.ooov, L.b)
#
#     LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))
#
#     return LH

