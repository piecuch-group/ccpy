import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2, vvvv_contraction
from ccpy.cholesky.cholesky_builders import build_2index_batch_vvvv_aa_herm, build_3index_batch_vvvv_ab_herm, build_2index_batch_vvvv_bb_herm
from ccpy.left.left_cc_intermediates import build_left_ccsd_chol_intermediates

def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

    # get LT intermediates
    X = build_left_ccsd_chol_intermediates(L, T, system)

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
    X = build_left_ccsd_chol_intermediates(L, T, system)

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
    #
    LH.a += 0.5 * ccpy_einsum("fena,efin->ai", H.aa.vvov, L.aa)
    LH.a += ccpy_einsum("efan,efin->ai", H.ab.vvvo, L.ab)
    #
    LH.a -= 0.5 * ccpy_einsum("finm,afmn->ai", H.aa.vooo, L.aa)
    LH.a -= ccpy_einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab)
    #
    LH.a += ccpy_einsum("ge,eiga->ai", X.a.vv, H.aa.vovv)
    LH.a += ccpy_einsum("fa,maef->em", X.b.vv, H.ab.ovvv)
    #
    LH.a += ccpy_einsum("mn,nima->ai", X.a.oo, H.aa.ooov)
    LH.a += ccpy_einsum("in,mnei->em", X.b.oo, H.ab.oovo)
    return LH


def build_LH_1B(L, LH, T, X, H):

    LH.b = ccpy_einsum("ea,ei->ai", H.b.vv, L.b)
    LH.b -= ccpy_einsum("im,am->ai", H.b.oo, L.b)
    LH.b += ccpy_einsum("eima,em->ai", H.ab.voov, L.a)
    LH.b += ccpy_einsum("eima,em->ai", H.bb.voov, L.b)
    LH.b -= 0.5 * ccpy_einsum("finm,afmn->ai", H.bb.vooo, L.bb)
    LH.b -= ccpy_einsum("finm,fanm->ai", H.ab.vooo, L.ab)
    #
    LH.b += ccpy_einsum("fena,feni->ai", H.ab.vvov, L.ab)
    LH.b += 0.5 * ccpy_einsum("fena,efin->ai", H.bb.vvov, L.bb)
    #
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

    LH.aa += 0.125 * ccpy_einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv) # V*T2 + V*T1^2
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.a.shape[0]):
    #     for b in range(a + 1, L.a.shape[0]):
    #         # <ab|ef> = <x|ae><x|bf>
    #         batch_ints = build_2index_batch_vvvv_aa_herm(a, b, H)
    #         LH.aa[a, b, :, :] += 0.25 * ccpy_einsum("ef,efij->ij", batch_ints, L.aa)
    tmp = vvvv_contraction.vvvv_t2_sym(H.chol.a.vv, 0.5 * L.aa.transpose(3, 2, 1, 0))
    LH.aa += tmp.transpose(3, 2, 1, 0)

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

    LH.ab += ccpy_einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv)
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.a.shape[0]):
    #     # <ab|ef> = <x|ae><x|bf>
    #     batch_ints = build_3index_batch_vvvv_ab_herm(a, H)
    #     LH.ab[a, :, :, :] += ccpy_einsum("bef,efij->bij", batch_ints, L.ab)
    tmp = vvvv_contraction.vvvv_t2(H.chol.a.vv, H.chol.b.vv, L.ab.transpose(3, 2, 1, 0))
    LH.ab += tmp.transpose(3, 2, 1, 0)

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

    LH.bb += 0.125 * ccpy_einsum("ijmn,mnab->abij", X.bb.oooo, H.bb.oovv) # V*T2 + V*T1^2
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.b.shape[0]):
    #     for b in range(a + 1, L.b.shape[0]):
    #         # <ab|ef> = <x|ae><x|bf>
    #         batch_ints = build_2index_batch_vvvv_bb_herm(a, b, H)
    #         LH.bb[a, b, :, :] += 0.25 * ccpy_einsum("ef,efij->ij", batch_ints, L.bb)
    tmp = vvvv_contraction.vvvv_t2_sym(H.chol.b.vv, 0.5 * L.bb.transpose(3, 2, 1, 0))
    LH.bb += tmp.transpose(3, 2, 1, 0)

    LH.bb += 0.5 * ccpy_einsum("ejab,ei->abij", H.bb.vovv, L.b)
    LH.bb -= 0.5 * ccpy_einsum("ijmb,am->abij", H.bb.ooov, L.b)
    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))
    return LH

