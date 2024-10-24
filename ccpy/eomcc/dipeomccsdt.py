"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DIP-EOMCCSDT approach with up to 4h-2p excitations"""
import numpy as np
from ccpy.eomcc.dipeom4_intermediates import get_dipeomccsdt_intermediates, add_ov_intermediates
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.ab, R.aba, R.abb, R.abaa, R.abab, R.abbb = cc_loops2.update_r_4h2p(
        R.ab,
        R.aba,
        R.abb,
        R.abaa,
        R.abab,
        R.abbb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    if RHF_symmetry:
       R.abb = np.transpose(R.aba, (1, 0, 2, 3))
       R.abbb = np.transpose(R.abaa, (1, 0, 2, 3, 4, 5))
    return R

def HR(dR, R, T, H, flag_RHF, system):

    X = get_dipeomccsdt_intermediates(H, R)
    # update R2
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aba = build_HR_3B(R, T, H, X)
    if flag_RHF:
       dR.abb = np.transpose(dR.aba, (1, 0, 2, 3))
    else:
       dR.abb = build_HR_3C(R, T, H, X)

    # Add H1(ov)*R1 to I_vo and I_ov intermediates
    X = add_ov_intermediates(X, R, H)

    # update R4
    dR.abaa = build_HR_4B(R, T, H, X)
    dR.abab = build_HR_4C(R, T, H, X)
    if flag_RHF:
        dR.abbb = np.transpose(dR.abaa, (1, 0, 2, 3, 4, 5))
    else:
        dR.abbb = build_HR_4D(R, T, H, X)

    return dR.flatten()

def build_HR_2B(R, T, H):
    x2b = -np.einsum("mi,mj->ij", H.a.oo, R.ab, optimize=True)
    x2b -= np.einsum("mj,im->ij", H.b.oo, R.ab, optimize=True)
    x2b += np.einsum("mnij,mn->ij", H.ab.oooo, R.ab, optimize=True)
    x2b += np.einsum("me,ijem->ij", H.a.ov, R.aba, optimize=True)
    x2b += np.einsum("me,ijem->ij", H.b.ov, R.abb, optimize=True)
    x2b -= np.einsum("nmfj,imfn->ij", H.ab.oovo, R.aba, optimize=True)
    x2b -= 0.5 * np.einsum("mnjf,imfn->ij", H.bb.ooov, R.abb, optimize=True)
    x2b -= 0.5 * np.einsum("mnif,mjfn->ij", H.aa.ooov, R.aba, optimize=True)
    x2b -= np.einsum("mnif,mjfn->ij", H.ab.ooov, R.abb, optimize=True)
    # additional R(4h-2p) terms
    x2b += 0.25 * np.einsum("mnef,ijefmn->ij", H.aa.oovv, R.abaa, optimize=True)
    x2b += np.einsum("mnef,ijefmn->ij", H.ab.oovv, R.abab, optimize=True)
    x2b += 0.25 * np.einsum("mnef,ijefmn->ij", H.bb.oovv, R.abbb, optimize=True)
    return x2b

def build_HR_3B(R, T, H, X):
    x3b = -0.5 * np.einsum("cmki,mj->ijck", H.aa.vooo, R.ab, optimize=True)
    x3b -= np.einsum("cmkj,im->ijck", H.ab.vooo, R.ab, optimize=True)
    x3b -= np.einsum("mk,ijcm->ijck", H.a.oo, R.aba, optimize=True)
    x3b -= 0.5 * np.einsum("mj,imck->ijck", H.b.oo, R.aba, optimize=True)
    x3b += 0.5 * np.einsum("ce,ijek->ijck", H.a.vv, R.aba, optimize=True)
    x3b += np.einsum("cmke,ijem->ijck", H.aa.voov, R.aba, optimize=True)
    x3b += np.einsum("cmke,ijem->ijck", H.ab.voov, R.abb, optimize=True)
    x3b += np.einsum("mnij,mnck->ijck", H.ab.oooo, R.aba, optimize=True)
    x3b += 0.25 * np.einsum("mnik,mjcn->ijck", H.aa.oooo, R.aba, optimize=True)
    x3b -= 0.5 * np.einsum("cmej,imek->ijck", H.ab.vovo, R.aba, optimize=True)
    x3b += np.einsum("ie,cekj->ijck", X["ab"]["ov"], T.ab, optimize=True)
    x3b += 0.5 * np.einsum("ej,ecik->ijck", X["ab"]["vo"], T.aa, optimize=True)
    # additional T3 terms
    x3b += 0.5 * np.einsum("ef,cefkij->ijck", X["ab"]["vv"], T.aab, optimize=True)
    # additional R(4h-2p) terms
    x3b += 0.5 * np.einsum("me,ijcekm->ijck", H.a.ov, R.abaa, optimize=True)
    x3b += 0.5 * np.einsum("me,ijcekm->ijck", H.b.ov, R.abab, optimize=True)
    x3b -= 0.5 * np.einsum("mnif,mjcfkn->ijck", H.aa.ooov, R.abaa, optimize=True)
    x3b -= np.einsum("mnif,mjcfkn->ijck", H.ab.ooov, R.abab, optimize=True)
    x3b -= 0.5 * np.einsum("nmfj,imcfkn->ijck", H.ab.oovo, R.abaa, optimize=True)
    x3b -= 0.25 * np.einsum("mnjf,imcfkn->ijck", H.bb.ooov, R.abab, optimize=True)
    x3b += 0.25 * np.einsum("cnef,ijefkn->ijck", H.aa.vovv, R.abaa, optimize=True)
    x3b += 0.5 * np.einsum("cnef,ijefkn->ijck", H.ab.vovv, R.abab, optimize=True)
    # antisymmetrize A(ik)
    x3b -= np.transpose(x3b, (3, 1, 2, 0))
    return x3b

def build_HR_3C(R, T, H, X):
    x3c = -np.einsum("mcik,mj->ijck", H.ab.ovoo, R.ab, optimize=True)
    x3c -= 0.5 * np.einsum("cmkj,im->ijck", H.bb.vooo, R.ab, optimize=True)
    x3c -= 0.5 * np.einsum("mi,mjck->ijck", H.a.oo, R.abb, optimize=True)
    x3c -= np.einsum("mj,imck->ijck", H.b.oo, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("ce,ijek->ijck", H.b.vv, R.abb, optimize=True)
    x3c += np.einsum("mnij,mnck->ijck", H.ab.oooo, R.abb, optimize=True)
    x3c += 0.25 * np.einsum("mnjk,imcn->ijck", H.bb.oooo, R.abb, optimize=True)
    x3c += np.einsum("mcek,ijem->ijck", H.ab.ovvo, R.aba, optimize=True)
    x3c += np.einsum("cmke,ijem->ijck", H.bb.voov, R.abb, optimize=True)
    x3c -= 0.5 * np.einsum("mcie,mjek->ijck", H.ab.ovov, R.abb, optimize=True)
    x3c += np.einsum("ej,ecik->ijck", X["ab"]["vo"], T.ab, optimize=True)
    x3c += 0.5 * np.einsum("ie,ecjk->ijck", X["ab"]["ov"], T.bb, optimize=True)
    # additional T3 terms
    x3c += 0.5 * np.einsum("ef,ecfikj->ijck", X["ab"]["vv"], T.abb, optimize=True)
    # additional R(4p-2h) terms
    x3c += 0.5 * np.einsum("me,ijecmk->ijck", H.a.ov, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("me,ijecmk->ijck", H.b.ov, R.abbb, optimize=True)
    x3c += 0.5 * np.einsum("ncfe,ijfenk->ijck", H.ab.ovvv, R.abab, optimize=True)
    x3c += 0.25 * np.einsum("cnef,ijefkn->ijck", H.bb.vovv, R.abbb, optimize=True)
    x3c -= np.einsum("nmfk,ijfcnm->ijck", H.ab.oovo, R.abab, optimize=True)
    x3c -= 0.5 * np.einsum("mnkf,ijcfmn->ijck", H.bb.ooov, R.abbb, optimize=True)
    x3c -= 0.25 * np.einsum("mnif,mjfcnk->ijck", H.aa.ooov, R.abab, optimize=True)
    x3c -= 0.5 * np.einsum("mnif,mjfcnk->ijck", H.ab.ooov, R.abbb, optimize=True)
    # antisymmetrize A(j~k~)
    x3c -= np.einsum("ijck->ikcj", x3c, optimize=True)
    return x3c

def build_HR_4B(R, T, H, X):
    ### Moment-like terms < ij~klcd | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4b = -(6.0 / 12.0) * np.einsum("cmkl,ijdm->ijcdkl", H.aa.vooo, R.aba, optimize=True)
    x4b -= (6.0 / 12.0) * np.einsum("cmkj,imdl->ijcdkl", H.ab.vooo, R.aba, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("cdke,ijel->ijcdkl", H.aa.vvov, R.aba, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("ijde,cekl->ijcdkl", X["aba"]["oovv"], T.aa, optimize=True)
    x4b -= (3.0 / 12.0) * np.einsum("ijml,cdkm->ijcdkl", X["aba"]["oooo"], T.aa, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("ieck,delj->ijcdkl", X["aba"]["ovvo"], T.ab, optimize=True)
    ### Terms < ij~klcd | (X(2)T3)_C | 0 > ###
    x4b += (1.0 / 12.0) * np.einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.aaa, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("ie,cdeklj->ijcdkl", X["ab"]["ov"], T.aab, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.aaa, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("ijem,cdeklm->ijcdkl", X["abb"]["oovo"], T.aab, optimize=True)
    x4b -= (3.0 / 12.0) * np.einsum("iemk,cdemlj->ijcdkl", X["aba"]["ovoo"], T.aab, optimize=True)
    #
    x4b += (6.0 / 12.0) * np.einsum("dlef,ecfikj->ijcdkl", X["aba"]["vovv"], T.aab, optimize=True)
    x4b += (2.0 / 24.0) * np.einsum("dfej,feclik->ijcdkl", X["aba"]["vvvo"], T.aaa, optimize=True)
    ### Terms < ij~klcd | (H(2)R(4h-2p)_C | 0 > ###
    x4b -= (3.0 / 12.0) * np.einsum("ml,ijcdkm->ijcdkl", H.a.oo, R.abaa, optimize=True)
    x4b -= (1.0 / 12.0) * np.einsum("mj,imcdkl->ijcdkl", H.b.oo, R.abaa, optimize=True)
    x4b += (2.0 / 12.0) * np.einsum("de,ijcekl->ijcdkl", H.a.vv, R.abaa, optimize=True)
    x4b += (1.0 / 24.0) * np.einsum("cdef,ijefkl->ijcdkl", H.aa.vvvv, R.abaa, optimize=True)
    x4b += (3.0 / 24.0) * np.einsum("mnkl,ijcdmn->ijcdkl", H.aa.oooo, R.abaa, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("mnij,mncdkl->ijcdkl", H.ab.oooo, R.abaa, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("dmle,ijcekm->ijcdkl", H.aa.voov, R.abaa, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("dmle,ijcekm->ijcdkl", H.ab.voov, R.abab, optimize=True)
    x4b -= (2.0 / 12.0) * np.einsum("dmej,imcekl->ijcdkl", H.ab.vovo, R.abaa, optimize=True)
    ### 4-body Hbar term ###
    x4b += (6.0 / 12.0) * np.einsum("ef,edil,cfkj->ijcdkl", X["ab"]["vv"], T.aa, T.ab, optimize=True)
    # antisymmetrize A(ikl)A(cd)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (4, 1, 2, 3, 0, 5)) + np.transpose(x4b, (5, 1, 2, 3, 4, 0)) # A(i/kl)
    return x4b

def build_HR_4C(R, T, H, X):
    ### Moment-like terms < ij~kl~cd~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4c = -np.einsum("mdkl,ijcm->ijcdkl", H.ab.ovoo, R.aba, optimize=True)
    x4c -= np.einsum("cmkl,ijdm->ijcdkl", H.ab.vooo, R.abb, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cdel,ijek->ijcdkl", H.ab.vvvo, R.aba, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cdke,ijel->ijcdkl", H.ab.vvov, R.abb, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("cmki,mjdl->ijcdkl", H.aa.vooo, R.abb, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("dmlj,imck->ijcdkl", H.bb.vooo, R.aba, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.ab, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("ijmk,cdml->ijcdkl", X["aba"]["oooo"], T.ab, optimize=True)
    x4c += np.einsum("ijce,edkl->ijcdkl", X["aba"]["oovv"], T.ab, optimize=True)
    x4c += np.einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.ab, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("ieck,edjl->ijcdkl", X["aba"]["ovvo"], T.bb, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("ejdl,ecik->ijcdkl", X["abb"]["vovo"], T.aa, optimize=True)
    ### Terms < ij~kl~cd~  | (X(2)T3)_C | 0 > ###
    x4c += (2.0 / 4.0) * np.einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.aab, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("ie,cedkjl->ijcdkl", X["ab"]["ov"], T.abb, optimize=True)
    x4c += np.einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.aab, optimize=True)
    x4c += np.einsum("ijem,cedkml->ijcdkl", X["abb"]["oovo"], T.abb, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("ejml,ecdikm->ijcdkl", X["abb"]["vooo"], T.aab, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("iemk,cedmjl->ijcdkl", X["aba"]["ovoo"], T.abb, optimize=True)
    #
    x4c += (2.0 / 8.0) * np.einsum("cfej,efdikl->ijcdkl", X["aba"]["vvvo"], T.aab, optimize=True)
    x4c += (2.0 / 8.0) * np.einsum("dfei,cefkjl->ijcdkl", X["abb"]["vvvo"], T.abb, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("ckef,efdijl->ijcdkl", X["aba"]["vovv"], T.abb, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("dlef,cfekij->ijcdkl", X["abb"]["vovv"], T.aab, optimize=True)
    ### Terms < ij~kl~cd~  | (H(2)R(4h-2p)_C | 0 > ###
    x4c -= (2.0 / 4.0) * np.einsum("mi,mjcdkl->ijcdkl", H.a.oo, R.abab, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("mj,imcdkl->ijcdkl", H.b.oo, R.abab, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("ce,ijedkl->ijcdkl", H.a.vv, R.abab, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("de,ijcekl->ijcdkl", H.b.vv, R.abab, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("cdef,ijefkl->ijcdkl", H.ab.vvvv, R.abab, optimize=True)
    x4c += np.einsum("mnkl,ijcdmn->ijcdkl", H.ab.oooo, R.abab, optimize=True)
    x4c += (1.0 / 8.0) * np.einsum("mnik,mjcdnl->ijcdkl", H.aa.oooo, R.abab, optimize=True)
    x4c += (1.0 / 8.0) * np.einsum("mnjl,imcdkn->ijcdkl", H.bb.oooo, R.abab, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cmke,ijedml->ijcdkl", H.aa.voov, R.abab, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cmke,ijedml->ijcdkl", H.ab.voov, R.abbb, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("mdel,ijcekm->ijcdkl", H.ab.ovvo, R.abaa, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("dmle,ijcekm->ijcdkl", H.bb.voov, R.abab, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("cmel,ijedkm->ijcdkl", H.ab.vovo, R.abab, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("mdke,ijceml->ijcdkl", H.ab.ovov, R.abab, optimize=True)
    ### 4-body HBar ###
    x4c += (1.0 / 4.0) * np.einsum("ef,ecik,fdjl->ijcdkl", X["ab"]["vv"], T.aa, T.bb, optimize=True)
    x4c += np.einsum("ef,edil,cfkj->ijcdkl", X["ab"]["vv"], T.ab, T.ab, optimize=True)
    # antisymmetrize A(ik)A(jl)
    x4c -= np.transpose(x4c, (4, 1, 2, 3, 0, 5)) # A(ik)
    x4c -= np.transpose(x4c, (0, 5, 2, 3, 4, 1)) # A(jl)
    return x4c

def build_HR_4D(R, T, H, X):
    ### Moment-like terms < ij~k~l~c~d~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4d = -(6.0 / 12.0) * np.einsum("cmkl,ijdm->ijcdkl", H.bb.vooo, R.abb, optimize=True)
    x4d -= (6.0 / 12.0) * np.einsum("mcik,mjdl->ijcdkl", H.ab.ovoo, R.abb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("cdke,ijel->ijcdkl", H.bb.vvov, R.abb, optimize=True)
    x4d -= (3.0 / 12.0) * np.einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.bb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.bb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("ejck,edil->ijcdkl", X["abb"]["vovo"], T.ab, optimize=True)
    ### Terms < ij~k~l~c~d~  | (X(2)T3)_C | 0 > ###
    x4d += (1.0 / 12.0) * np.einsum("ie,ecdjkl->ijcdkl", X["ab"]["ov"], T.bbb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.abb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.abb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("ijem,ecdmkl->ijcdkl", X["abb"]["oovo"], T.bbb, optimize=True)
    x4d -= (3.0 / 12.0) * np.einsum("ejml,ecdikm->ijcdkl", X["abb"]["vooo"], T.abb, optimize=True)
    #
    x4d += (2.0 / 24.0) * np.einsum("dfei,ecfjkl->ijcdkl", X["abb"]["vvvo"], T.bbb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("dlef,fecijk->ijcdkl", X["abb"]["vovv"], T.abb, optimize=True)
    ### Terms < ij~k~l~c~d~  | (H(2)R(4h-2p)_C | 0 > ###
    x4d -= (3.0 / 12.0) * np.einsum("ml,ijcdkm->ijcdkl", H.b.oo, R.abbb, optimize=True)
    x4d -= (1.0 / 12.0) * np.einsum("mi,mjcdkl->ijcdkl", H.a.oo, R.abbb, optimize=True)
    x4d += (2.0 / 12.0) * np.einsum("de,ijcekl->ijcdkl", H.b.vv, R.abbb, optimize=True)
    x4d += (1.0 / 24.0) * np.einsum("cdef,ijefkl->ijcdkl", H.bb.vvvv, R.abbb, optimize=True)
    x4d += (3.0 / 24.0) * np.einsum("mnkl,ijcdmn->ijcdkl", H.bb.oooo, R.abbb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("mnij,mncdkl->ijcdkl", H.ab.oooo, R.abbb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("mdel,ijecmk->ijcdkl", H.ab.ovvo, R.abab, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("dmle,ijcekm->ijcdkl", H.bb.voov, R.abbb, optimize=True)
    x4d -= (2.0 / 12.0) * np.einsum("mdie,mjcekl->ijcdkl", H.ab.ovov, R.abbb, optimize=True)
    ### 4-body Hbar term ###
    x4d += (6.0 / 12.0) * np.einsum("ef,edil,fcjk->ijcdkl", X["ab"]["vv"], T.ab, T.bb, optimize=True)
    # antisymmetrize A(jkl)A(cd)
    x4d -= np.transpose(x4d, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4d -= np.transpose(x4d, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4d -= np.transpose(x4d, (0, 4, 2, 3, 1, 5)) + np.transpose(x4d, (0, 5, 2, 3, 4, 1)) # A(j/kl)
    return x4d

