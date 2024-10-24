import numpy as np
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

    LH.a = np.einsum("ea,ei->ai", H.a.vv, L.a, optimize=True)
    LH.a -= np.einsum("im,am->ai", H.a.oo, L.a, optimize=True)
    LH.a += np.einsum("eima,em->ai", H.aa.voov, L.a, optimize=True)
    LH.a += np.einsum("ieam,em->ai", H.ab.ovvo, L.b, optimize=True)
    #
    LH.a += 0.5 * np.einsum("fena,efin->ai", H.aa.vvov, L.aa, optimize=True)
    LH.a += np.einsum("efan,efin->ai", H.ab.vvvo, L.ab, optimize=True)
    #
    LH.a -= 0.5 * np.einsum("finm,afmn->ai", H.aa.vooo, L.aa, optimize=True)
    LH.a -= np.einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab, optimize=True)
    #
    LH.a += np.einsum("ge,eiga->ai", X.a.vv, H.aa.vovv, optimize=True)
    LH.a += np.einsum("fa,maef->em", X.b.vv, H.ab.ovvv, optimize=True)
    #
    LH.a += np.einsum("mn,nima->ai", X.a.oo, H.aa.ooov, optimize=True)
    LH.a += np.einsum("in,mnei->em", X.b.oo, H.ab.oovo, optimize=True)
    return LH


def build_LH_1B(L, LH, T, X, H):

    LH.b = np.einsum("ea,ei->ai", H.b.vv, L.b, optimize=True)
    LH.b -= np.einsum("im,am->ai", H.b.oo, L.b, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.ab.voov, L.a, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.bb.voov, L.b, optimize=True)
    LH.b -= 0.5 * np.einsum("finm,afmn->ai", H.bb.vooo, L.bb, optimize=True)
    LH.b -= np.einsum("finm,fanm->ai", H.ab.vooo, L.ab, optimize=True)
    #
    LH.b += np.einsum("fena,feni->ai", H.ab.vvov, L.ab, optimize=True)
    LH.b += 0.5 * np.einsum("fena,efin->ai", H.bb.vvov, L.bb, optimize=True)
    #
    LH.b += (
        np.einsum("ge,eiga->ai", X.b.vv, H.bb.vovv, optimize=True)
        + np.einsum("mo,oima->ai", X.b.oo, H.bb.ooov, optimize=True)
    )
    LH.b += (
        np.einsum("ge,eiga->ai", X.a.vv, H.ab.vovv, optimize=True)
        + np.einsum("mo,oima->ai", X.a.oo, H.ab.ooov, optimize=True)
    )
    return LH

def build_LH_2A(L, LH, T, X, H):

    LH.aa = 0.5 * np.einsum("ea,ebij->abij", H.a.vv, L.aa, optimize=True)
    LH.aa -= 0.5 * np.einsum("im,abmj->abij", H.a.oo, L.aa, optimize=True)
    LH.aa += np.einsum("jb,ai->abij", H.a.ov, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv, optimize=True)
    LH.aa += 0.5 * np.einsum("im,mjab->abij", X.a.oo, H.aa.oovv, optimize=True)
    LH.aa += np.einsum("eima,ebmj->abij", H.aa.voov, L.aa, optimize=True)
    LH.aa += np.einsum("ieam,bejm->abij", H.ab.ovvo, L.ab, optimize=True)
    LH.aa += 0.125 * np.einsum("ijmn,abmn->abij", H.aa.oooo, L.aa, optimize=True)

    LH.aa += 0.125 * np.einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv, optimize=True) # V*T2 + V*T1^2
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.a.shape[0]):
    #     for b in range(a + 1, L.a.shape[0]):
    #         # <ab|ef> = <x|ae><x|bf>
    #         batch_ints = build_2index_batch_vvvv_aa_herm(a, b, H)
    #         LH.aa[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, L.aa, optimize=True)
    tmp = vvvv_contraction.vvvv_t2_sym(H.chol.a.vv, 0.5 * L.aa.transpose(3, 2, 1, 0))
    LH.aa += tmp.transpose(3, 2, 1, 0)

    LH.aa += 0.5 * np.einsum("ejab,ei->abij", H.aa.vovv, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ijmb,am->abij", H.aa.ooov, L.a, optimize=True)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))
    return LH


def build_LH_2B(L, LH, T, X, H):

    LH.ab = -np.einsum("ijmb,am->abij", H.ab.ooov, L.a, optimize=True)
    LH.ab -= np.einsum("ijam,bm->abij", H.ab.oovo, L.b, optimize=True)
    LH.ab += np.einsum("ejab,ei->abij", H.ab.vovv, L.a, optimize=True)
    LH.ab += np.einsum("ieab,ej->abij", H.ab.ovvv, L.b, optimize=True)
    LH.ab += np.einsum("ijmn,abmn->abij", H.ab.oooo, L.ab, optimize=True)

    LH.ab += np.einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv, optimize=True)
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.a.shape[0]):
    #     # <ab|ef> = <x|ae><x|bf>
    #     batch_ints = build_3index_batch_vvvv_ab_herm(a, H)
    #     LH.ab[a, :, :, :] += np.einsum("bef,efij->bij", batch_ints, L.ab, optimize=True)
    tmp = vvvv_contraction.vvvv_t2(H.chol.a.vv, H.chol.b.vv, L.ab.transpose(3, 2, 1, 0))
    LH.ab += tmp.transpose(3, 2, 1, 0)

    LH.ab += np.einsum("ejmb,aeim->abij", H.ab.voov, L.aa, optimize=True)
    LH.ab += np.einsum("eima,ebmj->abij", H.aa.voov, L.ab, optimize=True)
    LH.ab += np.einsum("ejmb,aeim->abij", H.bb.voov, L.ab, optimize=True)
    LH.ab += np.einsum("ieam,ebmj->abij", H.ab.ovvo, L.bb, optimize=True)
    LH.ab -= np.einsum("iemb,aemj->abij", H.ab.ovov, L.ab, optimize=True)
    LH.ab -= np.einsum("ejam,ebim->abij", H.ab.vovo, L.ab, optimize=True)
    LH.ab -= np.einsum("ea,ijeb->abij", X.a.vv, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("im,mjab->abij", X.a.oo, H.ab.oovv, optimize=True)
    LH.ab -= np.einsum("ea,jibe->baji", X.b.vv, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("im,jmba->baji", X.b.oo, H.ab.oovv, optimize=True)
    LH.ab += np.einsum("ea,ebij->abij", H.a.vv, L.ab, optimize=True)
    LH.ab += np.einsum("eb,aeij->abij", H.b.vv, L.ab, optimize=True)
    LH.ab -= np.einsum("im,abmj->abij", H.a.oo, L.ab, optimize=True)
    LH.ab -= np.einsum("jm,abim->abij", H.b.oo, L.ab, optimize=True)
    LH.ab += np.einsum("jb,ai->abij", H.b.ov, L.a, optimize=True)
    LH.ab += np.einsum("ia,bj->abij", H.a.ov, L.b, optimize=True)
    return LH

def build_LH_2C(L, LH, T, X, H):

    LH.bb = 0.5 * np.einsum("ea,ebij->abij", H.b.vv, L.bb, optimize=True)
    LH.bb -= 0.5 * np.einsum("im,abmj->abij", H.b.oo, L.bb, optimize=True)
    LH.bb += np.einsum("jb,ai->abij", H.b.ov, L.b, optimize=True)
    LH.bb -= 0.5 * np.einsum("ea,ijeb->abij", X.b.vv, H.bb.oovv, optimize=True)
    LH.bb += 0.5 * np.einsum("im,mjab->abij", X.b.oo, H.bb.oovv, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb += 0.125 * np.einsum("ijmn,abmn->abij", H.bb.oooo, L.bb, optimize=True)

    LH.bb += 0.125 * np.einsum("ijmn,mnab->abij", X.bb.oooo, H.bb.oovv, optimize=True) # V*T2 + V*T1^2
    # deal with the bare (vvvv) term using Cholesky
    # for a in range(L.b.shape[0]):
    #     for b in range(a + 1, L.b.shape[0]):
    #         # <ab|ef> = <x|ae><x|bf>
    #         batch_ints = build_2index_batch_vvvv_bb_herm(a, b, H)
    #         LH.bb[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, L.bb, optimize=True)
    tmp = vvvv_contraction.vvvv_t2_sym(H.chol.b.vv, 0.5 * L.bb.transpose(3, 2, 1, 0))
    LH.bb += tmp.transpose(3, 2, 1, 0)

    LH.bb += 0.5 * np.einsum("ejab,ei->abij", H.bb.vovv, L.b, optimize=True)
    LH.bb -= 0.5 * np.einsum("ijmb,am->abij", H.bb.ooov, L.b, optimize=True)
    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))
    return LH

