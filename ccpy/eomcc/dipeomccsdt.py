'''
Double Ionization Potential Equation-of-Motion Coupled-Cluster Method
with 2h, 3h-1p, and 4h-2p Excitations on top of CCSDT [DIP-EOMCCSDT(4h-2p)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
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
    x2b = -ccpy_einsum("mi,mj->ij", H.a.oo, R.ab)
    x2b -= ccpy_einsum("mj,im->ij", H.b.oo, R.ab)
    x2b += ccpy_einsum("mnij,mn->ij", H.ab.oooo, R.ab)
    x2b += ccpy_einsum("me,ijem->ij", H.a.ov, R.aba)
    x2b += ccpy_einsum("me,ijem->ij", H.b.ov, R.abb)
    x2b -= ccpy_einsum("nmfj,imfn->ij", H.ab.oovo, R.aba)
    x2b -= 0.5 * ccpy_einsum("mnjf,imfn->ij", H.bb.ooov, R.abb)
    x2b -= 0.5 * ccpy_einsum("mnif,mjfn->ij", H.aa.ooov, R.aba)
    x2b -= ccpy_einsum("mnif,mjfn->ij", H.ab.ooov, R.abb)
    # additional R(4h-2p) terms
    x2b += 0.25 * ccpy_einsum("mnef,ijefmn->ij", H.aa.oovv, R.abaa)
    x2b += ccpy_einsum("mnef,ijefmn->ij", H.ab.oovv, R.abab)
    x2b += 0.25 * ccpy_einsum("mnef,ijefmn->ij", H.bb.oovv, R.abbb)
    return x2b

def build_HR_3B(R, T, H, X):
    x3b = -0.5 * ccpy_einsum("cmki,mj->ijck", H.aa.vooo, R.ab)
    x3b -= ccpy_einsum("cmkj,im->ijck", H.ab.vooo, R.ab)
    x3b -= ccpy_einsum("mk,ijcm->ijck", H.a.oo, R.aba)
    x3b -= 0.5 * ccpy_einsum("mj,imck->ijck", H.b.oo, R.aba)
    x3b += 0.5 * ccpy_einsum("ce,ijek->ijck", H.a.vv, R.aba)
    x3b += ccpy_einsum("cmke,ijem->ijck", H.aa.voov, R.aba)
    x3b += ccpy_einsum("cmke,ijem->ijck", H.ab.voov, R.abb)
    x3b += ccpy_einsum("mnij,mnck->ijck", H.ab.oooo, R.aba)
    x3b += 0.25 * ccpy_einsum("mnik,mjcn->ijck", H.aa.oooo, R.aba)
    x3b -= 0.5 * ccpy_einsum("cmej,imek->ijck", H.ab.vovo, R.aba)
    x3b += ccpy_einsum("ie,cekj->ijck", X["ab"]["ov"], T.ab)
    x3b += 0.5 * ccpy_einsum("ej,ecik->ijck", X["ab"]["vo"], T.aa)
    # additional T3 terms
    x3b += 0.5 * ccpy_einsum("ef,cefkij->ijck", X["ab"]["vv"], T.aab)
    # additional R(4h-2p) terms
    x3b += 0.5 * ccpy_einsum("me,ijcekm->ijck", H.a.ov, R.abaa)
    x3b += 0.5 * ccpy_einsum("me,ijcekm->ijck", H.b.ov, R.abab)
    x3b -= 0.5 * ccpy_einsum("mnif,mjcfkn->ijck", H.aa.ooov, R.abaa)
    x3b -= ccpy_einsum("mnif,mjcfkn->ijck", H.ab.ooov, R.abab)
    x3b -= 0.5 * ccpy_einsum("nmfj,imcfkn->ijck", H.ab.oovo, R.abaa)
    x3b -= 0.25 * ccpy_einsum("mnjf,imcfkn->ijck", H.bb.ooov, R.abab)
    x3b += 0.25 * ccpy_einsum("cnef,ijefkn->ijck", H.aa.vovv, R.abaa)
    x3b += 0.5 * ccpy_einsum("cnef,ijefkn->ijck", H.ab.vovv, R.abab)
    # antisymmetrize A(ik)
    x3b -= np.transpose(x3b, (3, 1, 2, 0))
    return x3b

def build_HR_3C(R, T, H, X):
    x3c = -ccpy_einsum("mcik,mj->ijck", H.ab.ovoo, R.ab)
    x3c -= 0.5 * ccpy_einsum("cmkj,im->ijck", H.bb.vooo, R.ab)
    x3c -= 0.5 * ccpy_einsum("mi,mjck->ijck", H.a.oo, R.abb)
    x3c -= ccpy_einsum("mj,imck->ijck", H.b.oo, R.abb)
    x3c += 0.5 * ccpy_einsum("ce,ijek->ijck", H.b.vv, R.abb)
    x3c += ccpy_einsum("mnij,mnck->ijck", H.ab.oooo, R.abb)
    x3c += 0.25 * ccpy_einsum("mnjk,imcn->ijck", H.bb.oooo, R.abb)
    x3c += ccpy_einsum("mcek,ijem->ijck", H.ab.ovvo, R.aba)
    x3c += ccpy_einsum("cmke,ijem->ijck", H.bb.voov, R.abb)
    x3c -= 0.5 * ccpy_einsum("mcie,mjek->ijck", H.ab.ovov, R.abb)
    x3c += ccpy_einsum("ej,ecik->ijck", X["ab"]["vo"], T.ab)
    x3c += 0.5 * ccpy_einsum("ie,ecjk->ijck", X["ab"]["ov"], T.bb)
    # additional T3 terms
    x3c += 0.5 * ccpy_einsum("ef,ecfikj->ijck", X["ab"]["vv"], T.abb)
    # additional R(4p-2h) terms
    x3c += 0.5 * ccpy_einsum("me,ijecmk->ijck", H.a.ov, R.abab)
    x3c += 0.5 * ccpy_einsum("me,ijecmk->ijck", H.b.ov, R.abbb)
    x3c += 0.5 * ccpy_einsum("ncfe,ijfenk->ijck", H.ab.ovvv, R.abab)
    x3c += 0.25 * ccpy_einsum("cnef,ijefkn->ijck", H.bb.vovv, R.abbb)
    x3c -= ccpy_einsum("nmfk,ijfcnm->ijck", H.ab.oovo, R.abab)
    x3c -= 0.5 * ccpy_einsum("mnkf,ijcfmn->ijck", H.bb.ooov, R.abbb)
    x3c -= 0.25 * ccpy_einsum("mnif,mjfcnk->ijck", H.aa.ooov, R.abab)
    x3c -= 0.5 * ccpy_einsum("mnif,mjfcnk->ijck", H.ab.ooov, R.abbb)
    # antisymmetrize A(j~k~)
    x3c -= ccpy_einsum("ijck->ikcj", x3c)
    return x3c

def build_HR_4B(R, T, H, X):
    ### Moment-like terms < ij~klcd | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4b = -(6.0 / 12.0) * ccpy_einsum("cmkl,ijdm->ijcdkl", H.aa.vooo, R.aba)
    x4b -= (6.0 / 12.0) * ccpy_einsum("cmkj,imdl->ijcdkl", H.ab.vooo, R.aba)
    x4b += (3.0 / 12.0) * ccpy_einsum("cdke,ijel->ijcdkl", H.aa.vvov, R.aba)
    x4b += (6.0 / 12.0) * ccpy_einsum("ijde,cekl->ijcdkl", X["aba"]["oovv"], T.aa)
    x4b -= (3.0 / 12.0) * ccpy_einsum("ijml,cdkm->ijcdkl", X["aba"]["oooo"], T.aa)
    x4b += (6.0 / 12.0) * ccpy_einsum("ieck,delj->ijcdkl", X["aba"]["ovvo"], T.ab)
    ### Terms < ij~klcd | (X(2)T3)_C | 0 > ###
    x4b += (1.0 / 12.0) * ccpy_einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.aaa)
    x4b += (3.0 / 12.0) * ccpy_einsum("ie,cdeklj->ijcdkl", X["ab"]["ov"], T.aab)
    x4b += (3.0 / 12.0) * ccpy_einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.aaa)
    x4b += (3.0 / 12.0) * ccpy_einsum("ijem,cdeklm->ijcdkl", X["abb"]["oovo"], T.aab)
    x4b -= (3.0 / 12.0) * ccpy_einsum("iemk,cdemlj->ijcdkl", X["aba"]["ovoo"], T.aab)
    #
    x4b += (6.0 / 12.0) * ccpy_einsum("dlef,ecfikj->ijcdkl", X["aba"]["vovv"], T.aab)
    x4b += (2.0 / 24.0) * ccpy_einsum("dfej,feclik->ijcdkl", X["aba"]["vvvo"], T.aaa)
    ### Terms < ij~klcd | (H(2)R(4h-2p)_C | 0 > ###
    x4b -= (3.0 / 12.0) * ccpy_einsum("ml,ijcdkm->ijcdkl", H.a.oo, R.abaa)
    x4b -= (1.0 / 12.0) * ccpy_einsum("mj,imcdkl->ijcdkl", H.b.oo, R.abaa)
    x4b += (2.0 / 12.0) * ccpy_einsum("de,ijcekl->ijcdkl", H.a.vv, R.abaa)
    x4b += (1.0 / 24.0) * ccpy_einsum("cdef,ijefkl->ijcdkl", H.aa.vvvv, R.abaa)
    x4b += (3.0 / 24.0) * ccpy_einsum("mnkl,ijcdmn->ijcdkl", H.aa.oooo, R.abaa)
    x4b += (3.0 / 12.0) * ccpy_einsum("mnij,mncdkl->ijcdkl", H.ab.oooo, R.abaa)
    x4b += (6.0 / 12.0) * ccpy_einsum("dmle,ijcekm->ijcdkl", H.aa.voov, R.abaa)
    x4b += (6.0 / 12.0) * ccpy_einsum("dmle,ijcekm->ijcdkl", H.ab.voov, R.abab)
    x4b -= (2.0 / 12.0) * ccpy_einsum("dmej,imcekl->ijcdkl", H.ab.vovo, R.abaa)
    ### 4-body Hbar term ###
    x4b += (6.0 / 12.0) * ccpy_einsum("ef,edil,cfkj->ijcdkl", X["ab"]["vv"], T.aa, T.ab)
    # antisymmetrize A(ikl)A(cd)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (4, 1, 2, 3, 0, 5)) + np.transpose(x4b, (5, 1, 2, 3, 4, 0)) # A(i/kl)
    return x4b

def build_HR_4C(R, T, H, X):
    ### Moment-like terms < ij~kl~cd~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4c = -ccpy_einsum("mdkl,ijcm->ijcdkl", H.ab.ovoo, R.aba)
    x4c -= ccpy_einsum("cmkl,ijdm->ijcdkl", H.ab.vooo, R.abb)
    x4c += (2.0 / 4.0) * ccpy_einsum("cdel,ijek->ijcdkl", H.ab.vvvo, R.aba)
    x4c += (2.0 / 4.0) * ccpy_einsum("cdke,ijel->ijcdkl", H.ab.vvov, R.abb)
    x4c -= (1.0 / 4.0) * ccpy_einsum("cmki,mjdl->ijcdkl", H.aa.vooo, R.abb)
    x4c -= (1.0 / 4.0) * ccpy_einsum("dmlj,imck->ijcdkl", H.bb.vooo, R.aba)
    x4c -= (2.0 / 4.0) * ccpy_einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.ab)
    x4c -= (2.0 / 4.0) * ccpy_einsum("ijmk,cdml->ijcdkl", X["aba"]["oooo"], T.ab)
    x4c += ccpy_einsum("ijce,edkl->ijcdkl", X["aba"]["oovv"], T.ab)
    x4c += ccpy_einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.ab)
    x4c += (1.0 / 4.0) * ccpy_einsum("ieck,edjl->ijcdkl", X["aba"]["ovvo"], T.bb)
    x4c += (1.0 / 4.0) * ccpy_einsum("ejdl,ecik->ijcdkl", X["abb"]["vovo"], T.aa)
    ### Terms < ij~kl~cd~  | (X(2)T3)_C | 0 > ###
    x4c += (2.0 / 4.0) * ccpy_einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.aab)
    x4c += (2.0 / 4.0) * ccpy_einsum("ie,cedkjl->ijcdkl", X["ab"]["ov"], T.abb)
    x4c += ccpy_einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.aab)
    x4c += ccpy_einsum("ijem,cedkml->ijcdkl", X["abb"]["oovo"], T.abb)
    x4c -= (1.0 / 4.0) * ccpy_einsum("ejml,ecdikm->ijcdkl", X["abb"]["vooo"], T.aab)
    x4c -= (1.0 / 4.0) * ccpy_einsum("iemk,cedmjl->ijcdkl", X["aba"]["ovoo"], T.abb)
    #
    x4c += (2.0 / 8.0) * ccpy_einsum("cfej,efdikl->ijcdkl", X["aba"]["vvvo"], T.aab)
    x4c += (2.0 / 8.0) * ccpy_einsum("dfei,cefkjl->ijcdkl", X["abb"]["vvvo"], T.abb)
    x4c += (2.0 / 4.0) * ccpy_einsum("ckef,efdijl->ijcdkl", X["aba"]["vovv"], T.abb)
    x4c += (2.0 / 4.0) * ccpy_einsum("dlef,cfekij->ijcdkl", X["abb"]["vovv"], T.aab)
    ### Terms < ij~kl~cd~  | (H(2)R(4h-2p)_C | 0 > ###
    x4c -= (2.0 / 4.0) * ccpy_einsum("mi,mjcdkl->ijcdkl", H.a.oo, R.abab)
    x4c -= (2.0 / 4.0) * ccpy_einsum("mj,imcdkl->ijcdkl", H.b.oo, R.abab)
    x4c += (1.0 / 4.0) * ccpy_einsum("ce,ijedkl->ijcdkl", H.a.vv, R.abab)
    x4c += (1.0 / 4.0) * ccpy_einsum("de,ijcekl->ijcdkl", H.b.vv, R.abab)
    x4c += (1.0 / 4.0) * ccpy_einsum("cdef,ijefkl->ijcdkl", H.ab.vvvv, R.abab)
    x4c += ccpy_einsum("mnkl,ijcdmn->ijcdkl", H.ab.oooo, R.abab)
    x4c += (1.0 / 8.0) * ccpy_einsum("mnik,mjcdnl->ijcdkl", H.aa.oooo, R.abab)
    x4c += (1.0 / 8.0) * ccpy_einsum("mnjl,imcdkn->ijcdkl", H.bb.oooo, R.abab)
    x4c += (2.0 / 4.0) * ccpy_einsum("cmke,ijedml->ijcdkl", H.aa.voov, R.abab)
    x4c += (2.0 / 4.0) * ccpy_einsum("cmke,ijedml->ijcdkl", H.ab.voov, R.abbb)
    x4c += (2.0 / 4.0) * ccpy_einsum("mdel,ijcekm->ijcdkl", H.ab.ovvo, R.abaa)
    x4c += (2.0 / 4.0) * ccpy_einsum("dmle,ijcekm->ijcdkl", H.bb.voov, R.abab)
    x4c -= (2.0 / 4.0) * ccpy_einsum("cmel,ijedkm->ijcdkl", H.ab.vovo, R.abab)
    x4c -= (2.0 / 4.0) * ccpy_einsum("mdke,ijceml->ijcdkl", H.ab.ovov, R.abab)
    ### 4-body HBar ###
    x4c += (1.0 / 4.0) * ccpy_einsum("ef,ecik,fdjl->ijcdkl", X["ab"]["vv"], T.aa, T.bb)
    x4c += ccpy_einsum("ef,edil,cfkj->ijcdkl", X["ab"]["vv"], T.ab, T.ab)
    # antisymmetrize A(ik)A(jl)
    x4c -= np.transpose(x4c, (4, 1, 2, 3, 0, 5)) # A(ik)
    x4c -= np.transpose(x4c, (0, 5, 2, 3, 4, 1)) # A(jl)
    return x4c

def build_HR_4D(R, T, H, X):
    ### Moment-like terms < ij~k~l~c~d~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4d = -(6.0 / 12.0) * ccpy_einsum("cmkl,ijdm->ijcdkl", H.bb.vooo, R.abb)
    x4d -= (6.0 / 12.0) * ccpy_einsum("mcik,mjdl->ijcdkl", H.ab.ovoo, R.abb)
    x4d += (3.0 / 12.0) * ccpy_einsum("cdke,ijel->ijcdkl", H.bb.vvov, R.abb)
    x4d -= (3.0 / 12.0) * ccpy_einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.bb)
    x4d += (6.0 / 12.0) * ccpy_einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.bb)
    x4d += (6.0 / 12.0) * ccpy_einsum("ejck,edil->ijcdkl", X["abb"]["vovo"], T.ab)
    ### Terms < ij~k~l~c~d~  | (X(2)T3)_C | 0 > ###
    x4d += (1.0 / 12.0) * ccpy_einsum("ie,ecdjkl->ijcdkl", X["ab"]["ov"], T.bbb)
    x4d += (3.0 / 12.0) * ccpy_einsum("ej,ecdikl->ijcdkl", X["ab"]["vo"], T.abb)
    x4d += (3.0 / 12.0) * ccpy_einsum("ijem,ecdmkl->ijcdkl", X["aba"]["oovo"], T.abb)
    x4d += (3.0 / 12.0) * ccpy_einsum("ijem,ecdmkl->ijcdkl", X["abb"]["oovo"], T.bbb)
    x4d -= (3.0 / 12.0) * ccpy_einsum("ejml,ecdikm->ijcdkl", X["abb"]["vooo"], T.abb)
    #
    x4d += (2.0 / 24.0) * ccpy_einsum("dfei,ecfjkl->ijcdkl", X["abb"]["vvvo"], T.bbb)
    x4d += (6.0 / 12.0) * ccpy_einsum("dlef,fecijk->ijcdkl", X["abb"]["vovv"], T.abb)
    ### Terms < ij~k~l~c~d~  | (H(2)R(4h-2p)_C | 0 > ###
    x4d -= (3.0 / 12.0) * ccpy_einsum("ml,ijcdkm->ijcdkl", H.b.oo, R.abbb)
    x4d -= (1.0 / 12.0) * ccpy_einsum("mi,mjcdkl->ijcdkl", H.a.oo, R.abbb)
    x4d += (2.0 / 12.0) * ccpy_einsum("de,ijcekl->ijcdkl", H.b.vv, R.abbb)
    x4d += (1.0 / 24.0) * ccpy_einsum("cdef,ijefkl->ijcdkl", H.bb.vvvv, R.abbb)
    x4d += (3.0 / 24.0) * ccpy_einsum("mnkl,ijcdmn->ijcdkl", H.bb.oooo, R.abbb)
    x4d += (3.0 / 12.0) * ccpy_einsum("mnij,mncdkl->ijcdkl", H.ab.oooo, R.abbb)
    x4d += (6.0 / 12.0) * ccpy_einsum("mdel,ijecmk->ijcdkl", H.ab.ovvo, R.abab)
    x4d += (6.0 / 12.0) * ccpy_einsum("dmle,ijcekm->ijcdkl", H.bb.voov, R.abbb)
    x4d -= (2.0 / 12.0) * ccpy_einsum("mdie,mjcekl->ijcdkl", H.ab.ovov, R.abbb)
    ### 4-body Hbar term ###
    x4d += (6.0 / 12.0) * ccpy_einsum("ef,edil,fcjk->ijcdkl", X["ab"]["vv"], T.ab, T.bb)
    # antisymmetrize A(jkl)A(cd)
    x4d -= np.transpose(x4d, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4d -= np.transpose(x4d, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4d -= np.transpose(x4d, (0, 4, 2, 3, 1, 5)) + np.transpose(x4d, (0, 5, 2, 3, 4, 1)) # A(j/kl)
    return x4d

