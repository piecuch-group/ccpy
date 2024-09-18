import numpy as np
from ccpy.utilities.updates import cc_loops2
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

    L.a, L.b, LH.a, LH.b = cc_loops2.cc_loops2.update_l1(L.a, L.b, LH.a, LH.b,
                                                         omega,
                                                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                         shift)
    L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb = cc_loops2.cc_loops2.update_l2(L.aa, L.ab, L.bb, LH.aa, LH.ab, LH.bb,
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

    L.a, L.b, L.aa, L.ab, L.bb = cc_loops2.cc_loops2.update_r(
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
    LH.a += 0.5 * np.einsum("fena,efin->ai", H.aa.vvov, L.aa, optimize=True)
    LH.a += np.einsum("efan,efin->ai", H.ab.vvvo, L.ab, optimize=True)
    LH.a -= 0.5 * np.einsum("finm,afmn->ai", H.aa.vooo, L.aa, optimize=True)
    LH.a -= np.einsum("ifmn,afmn->ai", H.ab.ovoo, L.ab, optimize=True)
    LH.a += np.einsum("ge,eiga->ai", X.a.vv, H.aa.vovv, optimize=True)
    LH.a += np.einsum("mn,nima->ai", X.a.oo, H.aa.ooov, optimize=True)
    LH.a += np.einsum("fa,maef->em", X.b.vv, H.ab.ovvv, optimize=True)
    LH.a += np.einsum("in,mnei->em", X.b.oo, H.ab.oovo, optimize=True)
    return LH


def build_LH_1B(L, LH, T, X, H):

    LH.b = np.einsum("ea,ei->ai", H.b.vv, L.b, optimize=True)
    LH.b -= np.einsum("im,am->ai", H.b.oo, L.b, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.ab.voov, L.a, optimize=True)
    LH.b += np.einsum("eima,em->ai", H.bb.voov, L.b, optimize=True)
    LH.b -= 0.5 * np.einsum("finm,afmn->ai", H.bb.vooo, L.bb, optimize=True)
    LH.b -= np.einsum("finm,fanm->ai", H.ab.vooo, L.ab, optimize=True)
    LH.b += np.einsum("fena,feni->ai", H.ab.vvov, L.ab, optimize=True)
    LH.b += 0.5 * np.einsum("fena,efin->ai", H.bb.vvov, L.bb, optimize=True)
    LH.b += (
        np.einsum("ge,eiga->ai", X.b.vv, H.bb.vovv, optimize=True)
        + np.einsum("mo,oima->ai", X.b.oo, H.bb.ooov, optimize=True)
    )
    LH.b += (
        np.einsum("ge,eiga->ai", X.a.vv, H.ab.vovv, optimize=True)
        + np.einsum("mo,oima->ai", X.b.oo, H.ab.ooov, optimize=True)
    )
    return LH

def build_LH_2A(L, LH, T, X, H):

    nua, noa = L.a.shape
    oa = slice(noa)
    va = slice(noa, noa + nua)

    LH.aa = 0.5 * np.einsum("ea,ebij->abij", H.a.vv, L.aa, optimize=True)
    LH.aa -= 0.5 * np.einsum("im,abmj->abij", H.a.oo, L.aa, optimize=True)
    LH.aa += np.einsum("jb,ai->abij", H.a.ov, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv, optimize=True)
    LH.aa += 0.5 * np.einsum("im,mjab->abij", X.a.oo, H.aa.oovv, optimize=True)
    LH.aa += np.einsum("eima,ebmj->abij", H.aa.voov, L.aa, optimize=True)
    LH.aa += np.einsum("ieam,bejm->abij", H.ab.ovvo, L.ab, optimize=True)
    LH.aa += 0.125 * np.einsum("ijmn,abmn->abij", H.aa.oooo, L.aa, optimize=True)

    #LH.aa += 0.125 * np.einsum("efab,efij->abij", H.aa.vvvv, L.aa, optimize=True)
    I2A_vovv = H.aa.vovv + 0.5 * np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    LH.aa += 0.125 * np.einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv, optimize=True) # V*T2 + V*T1^2
    LH.aa -= 0.25 * np.einsum("jine,enab->abij", X.aa.ooov, I2A_vovv, optimize=True)
    # deal with the bare (vvvv) term using Cholesky
    for a in range(nua):
      for b in range(a + 1, nua):
          # <ab|ef> = <x|ae><x|bf>
          v_ef = np.einsum("xe,xf->ef", H.chol_a[:, va, a + noa], H.chol_a[:, va, b + noa], optimize=True)
          LH.aa[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", v_ef - v_ef.transpose(1, 0), L.aa, optimize=True)

    LH.aa += 0.5 * np.einsum("ejab,ei->abij", H.aa.vovv, L.a, optimize=True)
    LH.aa -= 0.5 * np.einsum("ijmb,am->abij", H.aa.ooov, L.a, optimize=True)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2, 3)) + np.transpose(LH.aa, (0, 1, 3, 2)) - np.transpose(LH.aa, (1, 0, 3, 2))
    return LH


def build_LH_2B(L, LH, T, X, H):

    nua, nub, noa, nob = L.ab.shape
    oa = slice(noa)
    va = slice(noa, noa + nua)
    ob = slice(nob)
    vb = slice(nob, nob + nub)

    LH.ab = -np.einsum("ijmb,am->abij", H.ab.ooov, L.a, optimize=True)
    LH.ab -= np.einsum("ijam,bm->abij", H.ab.oovo, L.b, optimize=True)
    LH.ab += np.einsum("ejab,ei->abij", H.ab.vovv, L.a, optimize=True)
    LH.ab += np.einsum("ieab,ej->abij", H.ab.ovvv, L.b, optimize=True)
    LH.ab += np.einsum("ijmn,abmn->abij", H.ab.oooo, L.ab, optimize=True)

    #LH.ab += np.einsum("efab,efij->abij", H.ab.vvvv, L.ab, optimize=True)
    I2B_ovvv = H.ab.ovvv + 0.5 * np.einsum("mnef,an->maef", H.ab.oovv, T.b, optimize=True)
    I2B_vovv = H.ab.vovv + 0.5 * np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)
    LH.ab += np.einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv, optimize=True)
    LH.ab -= np.einsum("ijmf,mfab->abij", X.ab.ooov, I2B_ovvv, optimize=True)
    LH.ab -= np.einsum("ijen,enab->abij", X.ab.oovo, I2B_vovv, optimize=True)
    # 
    # deal with the bare (vvvv) term using Cholesky
    for a in range(nua):
      # <ab|ef> = <x|ae><x|bf>
      v_bef = np.einsum("xe,xbf->bef", H.chol_a[:, va, a + noa], H.chol_b[:, va, va], optimize=True)
      LH.ab[a, :, :, :] += np.einsum("bef,efij->bij", v_bef, L.ab, optimize=True)

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

    nub, nob = L.b.shape
    ob = slice(nob)
    vb = slice(nob, nob + nub)

    LH.bb = 0.5 * np.einsum("ea,ebij->abij", H.b.vv, L.bb, optimize=True)
    LH.bb -= 0.5 * np.einsum("im,abmj->abij", H.b.oo, L.bb, optimize=True)
    LH.bb += np.einsum("jb,ai->abij", H.b.ov, L.b, optimize=True)
    LH.bb -= 0.5 * np.einsum("ea,ijeb->abij", X.b.vv, H.bb.oovv, optimize=True)
    LH.bb += 0.5 * np.einsum("im,mjab->abij", X.b.oo, H.bb.oovv, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.bb.voov, L.bb, optimize=True)
    LH.bb += np.einsum("eima,ebmj->abij", H.ab.voov, L.ab, optimize=True)
    LH.bb += 0.125 * np.einsum("ijmn,abmn->abij", H.bb.oooo, L.bb, optimize=True)

    #LH.bb += 0.125 * np.einsum("efab,efij->abij", H.bb.vvvv, L.bb, optimize=True)
    I2C_vovv = H.bb.vovv + 0.5 * np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
    LH.bb += 0.125 * np.einsum("ijmn,mnab->abij", X.bb.oooo, H.bb.oovv, optimize=True) # V*T2 + V*T1^2
    LH.bb -= 0.25 * np.einsum("jine,enab->abij", X.bb.ooov, I2C_vovv, optimize=True)
    #
    # deal with the bare (vvvv) term using Cholesky
    for a in range(nub):
      for b in range(a + 1, nub):
          # <ab|ef> = <x|ae><x|bf>
          v_ef = np.einsum("xe,xf->ef", H.chol_b[:, vb, a + nob], H.chol_b[:, vb, b + nob], optimize=True)
          LH.bb[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", v_ef - v_ef.transpose(1, 0), L.bb, optimize=True)

    LH.bb += 0.5 * np.einsum("ejab,ei->abij", H.bb.vovv, L.b, optimize=True)
    LH.bb -= 0.5 * np.einsum("ijmb,am->abij", H.bb.ooov, L.b, optimize=True)
    LH.bb -= np.transpose(LH.bb, (1, 0, 2, 3)) + np.transpose(LH.bb, (0, 1, 3, 2)) - np.transpose(LH.bb, (1, 0, 3, 2))
    return LH

