'''
Double Electron Attachment Equation-of-Motion Coupled-Cluster
Method with 2p, 3p-1h, and 4p-2h Excitations on top of CCSD [DEA-EOMCCSD(4p-2h)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.eomcc.deaeom4_intermediates import get_deaeom4_intermediates
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.ab, R.aba, R.abb, R.abaa, R.abab, R.abbb = cc_loops2.update_r_4p2h(
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

    X = get_deaeom4_intermediates(H, R)
    # update R2
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aba = build_HR_3B(R, T, H, X)
    if flag_RHF:
        dR.abb = np.transpose(dR.aba, (1, 0, 2, 3))
    else:
        dR.abb = build_HR_3C(R, T, H, X)
    # update R4
    dR.abaa = build_HR_4B(R, T, H, X)
    dR.abab = build_HR_4C(R, T, H, X)
    if flag_RHF:
        dR.abbb = np.transpose(dR.abaa, (1, 0, 2, 3, 4, 5))
    else:
        dR.abbb = build_HR_4D(R, T, H, X)

    return dR.flatten()

def build_HR_2B(R, T, H):
    x2b = ccpy_einsum("ae,eb->ab", H.a.vv, R.ab)
    x2b += ccpy_einsum("be,ae->ab", H.b.vv, R.ab)
    x2b += ccpy_einsum("abef,ef->ab", H.ab.vvvv, R.ab)
    x2b += ccpy_einsum("me,abem->ab", H.a.ov, R.aba)
    x2b += ccpy_einsum("me,abem->ab", H.b.ov, R.abb)
    x2b += ccpy_einsum("nbfe,aefn->ab", H.ab.ovvv, R.aba)
    x2b += 0.5 * ccpy_einsum("anef,ebfn->ab", H.aa.vovv, R.aba)
    x2b += 0.5 * ccpy_einsum("bnef,aefn->ab", H.bb.vovv, R.abb)
    x2b += ccpy_einsum("anef,ebfn->ab", H.ab.vovv, R.abb)
    # additional R(4p-2h) terms
    x2b += 0.25 * ccpy_einsum("mnef,abefmn->ab", H.aa.oovv, R.abaa)
    x2b += ccpy_einsum("mnef,abefmn->ab", H.ab.oovv, R.abab)
    x2b += 0.25 * ccpy_einsum("mnef,abefmn->ab", H.bb.oovv, R.abbb)
    return x2b

def build_HR_3B(R, T, H, X):
    # (1)
    x3b = 0.5 * ccpy_einsum("cake,eb->abck", H.aa.vvov, R.ab)
    # (2)
    x3b += ccpy_einsum("cbke,ae->abck", H.ab.vvov, R.ab)
    # (4)
    x3b += ccpy_einsum("ae,ebck->abck", H.a.vv, R.aba)
    # (5)
    x3b += 0.5 * ccpy_einsum("be,aeck->abck", H.b.vv, R.aba)
    # (6)
    x3b += ccpy_einsum("abef,efck->abck", H.ab.vvvv, R.aba)
    # (7)
    x3b += 0.25 * ccpy_einsum("acef,ebfk->abck", H.aa.vvvv, R.aba)
    # (8)
    x3b += ccpy_einsum("cmke,abem->abck", H.aa.voov, R.aba)
    # (9)
    x3b += ccpy_einsum("cmke,abem->abck", H.ab.voov, R.abb)
    # (10)
    x3b -= 0.5 * ccpy_einsum("mbke,aecm->abck", H.ab.ovov, R.aba)
    # (3) + (11)
    x3b -= 0.5 * ccpy_einsum("mb,acmk->abck", X["ab"]["ov"], T.aa)
    # (3) + (12)
    x3b -= ccpy_einsum("am,cbkm->abck", X["ab"]["vo"], T.ab)
    # (13)
    x3b -= 0.5 * ccpy_einsum("mk,abcm->abck", H.a.oo, R.aba)
    # additional R(4p-2h) terms
    x3b += 0.5 * ccpy_einsum("me,abcekm->abck", H.a.ov, R.abaa)
    x3b += 0.5 * ccpy_einsum("me,abcekm->abck", H.b.ov, R.abab)
    x3b -= 0.25 * ccpy_einsum("mnkf,abcfmn->abck", H.aa.ooov, R.abaa)
    x3b -= 0.5 * ccpy_einsum("mnkf,abcfmn->abck", H.ab.ooov, R.abab)
    x3b += 0.5 * ccpy_einsum("cnef,abefkn->abck", H.aa.vovv, R.abaa)
    x3b += ccpy_einsum("cnef,abefkn->abck", H.ab.vovv, R.abab)
    x3b += 0.5 * ccpy_einsum("nbfe,aecfkn->abck", H.ab.ovvv, R.abaa)
    x3b += 0.25 * ccpy_einsum("bnef,aecfkn->abck", H.bb.vovv, R.abab)
    # antisymmetrize A(ac)
    x3b -= ccpy_einsum("abck->cbak", x3b)
    return x3b

def build_HR_3C(R, T, H, X):
    # (1)
    x3c = ccpy_einsum("acek,eb->abck", H.ab.vvvo, R.ab)
    # (2)
    x3c += 0.5 * ccpy_einsum("cbke,ae->abck", H.bb.vvov, R.ab)
    # (4)
    x3c += 0.5 * ccpy_einsum("ae,ebck->abck", H.a.vv, R.abb)
    # (5)
    x3c += ccpy_einsum("be,aeck->abck", H.b.vv, R.abb)
    # (6)
    x3c += ccpy_einsum("abef,efck->abck", H.ab.vvvv, R.abb)
    # (7)
    x3c += 0.25 * ccpy_einsum("bcef,aefk->abck", H.bb.vvvv, R.abb)
    # (8)
    x3c += ccpy_einsum("mcek,abem->abck", H.ab.ovvo, R.aba)
    # (9)
    x3c += ccpy_einsum("cmke,abem->abck", H.bb.voov, R.abb)
    # (10)
    x3c -= 0.5 * ccpy_einsum("amek,ebcm->abck", H.ab.vovo, R.abb)
    # (3) + (11)
    x3c -= ccpy_einsum("mb,acmk->abck", X["ab"]["ov"], T.ab)
    # (3) + (12)
    x3c -= 0.5 * ccpy_einsum("am,bcmk->abck", X["ab"]["vo"], T.bb)
    # (13)
    x3c -= 0.5 * ccpy_einsum("mk,abcm->abck", H.b.oo, R.abb)
    # additional R(4p-2h) terms
    x3c += 0.5 * ccpy_einsum("me,abecmk->abck", H.a.ov, R.abab)
    x3c += 0.5 * ccpy_einsum("me,abecmk->abck", H.b.ov, R.abbb)
    x3c -= 0.5 * ccpy_einsum("nmfk,abfcnm->abck", H.ab.oovo, R.abab)
    x3c -= 0.25 * ccpy_einsum("mnkf,abcfmn->abck", H.bb.ooov, R.abbb)
    x3c += ccpy_einsum("ncfe,abfenk->abck", H.ab.ovvv, R.abab)
    x3c += 0.5 * ccpy_einsum("cnef,abefkn->abck", H.bb.vovv, R.abbb)
    x3c += 0.25 * ccpy_einsum("anef,ebfcnk->abck", H.aa.vovv, R.abab)
    x3c += 0.5 * ccpy_einsum("anef,ebfcnk->abck", H.ab.vovv, R.abbb)
    # antisymmetrize A(b~c~)
    x3c -= ccpy_einsum("abck->acbk", x3c)
    return x3c

def build_HR_4B(R, T, H, X):
    ### Moment-like terms < klab~cd | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(a/cd)A(kl) h2a(dcek) r_aba(ab~el)
    x4b = (6.0 / 12.0) * ccpy_einsum("cdke,abel->abcdkl", H.aa.vvov, R.aba)
    # diagram 2: A(c/ad)A(kl) h2b(cb~ke~) r_aba(ae~dl)
    x4b += (6.0 / 12.0) * ccpy_einsum("cbke,aedl->abcdkl", H.ab.vvov, R.aba)
    # diagram 3: -A(c/ad) h2a(cmkl) r_aba(ab~dm)
    x4b -= (3.0 / 12.0) * ccpy_einsum("cmkl,abdm->abcdkl", H.aa.vooo, R.aba)
    # diagram 13: -A(a/cd)A(kl) x_aba(ab~ml) t2a(cdkm)
    x4b -= (6.0 / 12.0) * ccpy_einsum("abml,cdkm->abcdkl", X["aba"]["vvoo"], T.aa)
    # diagram 14: A(c/ad) x_aba(ab~de) t2a(cekl)
    x4b += (3.0 / 12.0) * ccpy_einsum("abde,cekl->abcdkl", X["aba"]["vvvv"], T.aa)
    # diagram 15: -A(b/ac)A(kl) x_aba(am~ck) t2b(db~lm~)
    x4b -= (6.0 / 12.0) * ccpy_einsum("amck,dblm->abcdkl", X["aba"]["vovo"], T.ab)
    ### Terms < klab~cd | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 4: A(d/ac) h1a(de) r_abaa(ab~cekl)
    x4b += (3.0 / 12.0) * ccpy_einsum("de,abcekl->abcdkl", H.a.vv, R.abaa)
    # diagram 5: h1b(b~e~) r_abaa(ae~cdkl)
    x4b += (1.0 / 12.0) * ccpy_einsum("be,aecdkl->abcdkl", H.b.vv, R.abaa)
    # diagram 6: -A(kl) h1a(ml) r_abaa(ab~cdkm)
    x4b -= (2.0 / 12.0) * ccpy_einsum("ml,abcdkm->abcdkl", H.a.oo, R.abaa)
    # diagram 7: 1/2 h2a(mnkl) r_abaa(ab~cdmn)
    x4b += (1.0 / 24.0) * ccpy_einsum("mnkl,abcdmn->abcdkl", H.aa.oooo, R.abaa)
    # diagram 8: 1/2 A(a/cd) h2a(cdef) r_abaa(ab~efkl)
    x4b += (3.0 / 24.0) * ccpy_einsum("cdef,abefkl->abcdkl", H.aa.vvvv, R.abaa)
    # diagram 9: A(a/cd) h2b(ab~ef~) r_abaa(ef~cdkl)
    x4b += (3.0 / 12.0) * ccpy_einsum("abef,efcdkl->abcdkl", H.ab.vvvv, R.abaa)
    # diagram 10: A(d/ac)A(kl) h2a(dmle) r_abaa(ab~cekm)
    x4b += (6.0 / 12.0) * ccpy_einsum("dmle,abcekm->abcdkl", H.aa.voov, R.abaa)
    # diagram 11: A(d/ac)A(kl) h2b(dm~le~) r_abab(ab~ce~km~)
    x4b += (6.0 / 12.0) * ccpy_einsum("dmle,abcekm->abcdkl", H.ab.voov, R.abab)
    # diagram 12: -A(kl) h2b(mb~le~) r_abaa(ae~cdkm)
    x4b -= (2.0 / 12.0) * ccpy_einsum("mble,aecdkm->abcdkl", H.ab.ovov, R.abaa)
    ### 4-body Hbar term ###
    # diagram 15: A(c/ad)A(kl) x_ab(mn~) t2a(adml) t2b(cb~kn~)
    x4b += (6.0 / 12.0) * ccpy_einsum("mn,adml,cbkn->abcdkl", X["ab"]["oo"], T.aa, T.ab)
    # antisymmetrize A(acd)A(kl)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (2, 1, 0, 3, 4, 5)) + np.transpose(x4b, (3, 1, 2, 0, 4, 5)) # A(a/cd)
    return x4b

def build_HR_4C(R, T, H, X):
    ### Moment-like terms < kl~ab~cd~ | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(ac)A(bd) h2b(cd~el~) r_aba(ab~ek)
    x4c = ccpy_einsum("cdel,abek->abcdkl", H.ab.vvvo, R.aba)
    # diagram 2: A(bd)A(ac) h2b(cd~ke~) r_abb(ab~e~l~)
    x4c += ccpy_einsum("cdke,abel->abcdkl", H.ab.vvov, R.abb)
    # diagram 3: -A(bd) h2b(md~kl~) r_aba(ab~cm)
    x4c -= (2.0 / 4.0) * ccpy_einsum("mdkl,abcm->abcdkl", H.ab.ovoo, R.aba)
    # diagram 4: -A(ac) h2b(cm~kl~) r_abb(ab~d~m~)
    x4c -= (2.0 / 4.0) * ccpy_einsum("cmkl,abdm->abcdkl", H.ab.vooo, R.abb)
    # diagram 25: h2a(cake) * r_abb(eb~d~l~)
    x4c += (1.0 / 4.0) * ccpy_einsum("cake,ebdl->abcdkl", H.aa.vvov, R.abb)
    # diagram 26: h2c(d~b~l~e~) r_aba(ae~ck)
    x4c += (1.0 / 4.0) * ccpy_einsum("dble,aeck->abcdkl", H.bb.vvov, R.aba)
    # diagram 19: A(ac) x_abb(ab~d~e~) t2b(ce~kl~)
    x4c += (2.0 / 4.0) * ccpy_einsum("abde,cekl->abcdkl", X["abb"]["vvvv"], T.ab)
    # diagram 20: A(bd) x_aba(ab~ce) t2b(ed~kl~)
    x4c += (2.0 / 4.0) * ccpy_einsum("abce,edkl->abcdkl", X["aba"]["vvvv"], T.ab)
    # diagram 21: -A(ac)A(bd) x_aba(ab~mk) t2b(cd~ml~)
    x4c -= ccpy_einsum("abmk,cdml->abcdkl", X["aba"]["vvoo"], T.ab)
    # diagram 22: -A(ac)A(bd) x_abb(ab~m~l~) t2b(cd~km~)
    x4c -= ccpy_einsum("abml,cdkm->abcdkl", X["abb"]["vvoo"], T.ab)
    # diagram 23: -x_aba(am~ck) t2c(b~d~m~l~)
    x4c -= (1.0 / 4.0) * ccpy_einsum("amck,bdml->abcdkl", X["aba"]["vovo"], T.bb)
    # diagram 24: -x_abb(mb~d~l~) t2a(acmk)
    x4c -= (1.0 / 4.0) * ccpy_einsum("mbdl,acmk->abcdkl", X["abb"]["ovvo"], T.aa)
    ### Terms < kl~ab~cd~ | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 5: A(ac) h1a(ae) r_abab(eb~cd~kl~)
    x4c += (2.0 / 4.0) * ccpy_einsum("ae,ebcdkl->abcdkl", H.a.vv, R.abab)
    # diagram 6: A(bd) h1b(be) r_abab(ae~cd~kl~)
    x4c += (2.0 / 4.0) * ccpy_einsum("be,aecdkl->abcdkl", H.b.vv, R.abab)
    # diagram 7: -h1a(mk) r_abab(ab~cd~ml~)
    x4c -= (1.0 / 4.0) * ccpy_einsum("mk,abcdml->abcdkl", H.a.oo, R.abab)
    # diagram 8: -h1b(ml) r_abab(ab~cd~km~)
    x4c -= (1.0 / 4.0) * ccpy_einsum("ml,abcdkm->abcdkl", H.b.oo, R.abab)
    # diagram 9: h2b(mn~kl~) r_abab(ab~cd~mn~)
    x4c += (1.0 / 4.0) * ccpy_einsum("mnkl,abcdmn->abcdkl", H.ab.oooo, R.abab)
    # diagram 10: A(ac)A(bd) h2b(cd~ef~) r_abab(ab~ef~kl~)
    x4c += ccpy_einsum("cdef,abefkl->abcdkl", H.ab.vvvv, R.abab)
    # diagram 11: 1/2 h2a(acef) r_abab(eb~fd~kl~)
    x4c += (1.0 / 8.0) * ccpy_einsum("acef,ebfdkl->abcdkl", H.aa.vvvv, R.abab)
    # diagram 12: 1/2 h2c(b~d~e~f~) r_abab(ae~cf~kl~)
    x4c += (1.0 / 8.0) * ccpy_einsum("bdef,aecfkl->abcdkl", H.bb.vvvv, R.abab)
    # diagram 13: A(ac) h2a(cmke) r_abab(ab~ed~ml~)
    x4c += (2.0 / 4.0) * ccpy_einsum("cmke,abedml->abcdkl", H.aa.voov, R.abab)
    # diagram 14: A(ac) h2b(cm~ke~) r_abbb(ab~e~d~m~l~)
    x4c += (2.0 / 4.0) * ccpy_einsum("cmke,abedml->abcdkl", H.ab.voov, R.abbb)
    # diagram 15: A(bd) h2b(md~el~) r_abaa(ab~cekm)
    x4c += (2.0 / 4.0) * ccpy_einsum("mdel,abcekm->abcdkl", H.ab.ovvo, R.abaa)
    # diagram 16: A(bd) h2c(d~m~l~e~) r_abab(ab~ce~km~)
    x4c += (2.0 / 4.0) * ccpy_einsum("dmle,abcekm->abcdkl", H.bb.voov, R.abab)
    # diagram 17: -A(bd) h2b(md~ke~) r_abab(ab~ce~ml~)
    x4c -= (2.0 / 4.0) * ccpy_einsum("mdke,abceml->abcdkl", H.ab.ovov, R.abab)
    # diagram 18: -A(ac) h2b(cm~el~) r_abab(ab~ed~km~)
    x4c -= (2.0 / 4.0) * ccpy_einsum("cmel,abedkm->abcdkl", H.ab.vovo, R.abab)
    ### 4-body HBar ###
    # diagram 23: x_ab(mn~) t2a(acmk) t2c(b~d~n~l~)
    x4c += (1.0 / 4.0) * ccpy_einsum("mn,acmk,bdnl->abcdkl", X["ab"]["oo"], T.aa, T.bb)
    # diagram 24: A(ac)A(bd) x_ab(mn~) t2b(ad~ml~) t2b(cb~kn~)
    x4c += ccpy_einsum("mn,adml,cbkn->abcdkl", X["ab"]["oo"], T.ab, T.ab)
    # antisymmetrize A(ac)A(bd)
    x4c -= np.transpose(x4c, (2, 1, 0, 3, 4, 5)) # A(ac)
    x4c -= np.transpose(x4c, (0, 3, 2, 1, 4, 5)) # A(bd)
    return x4c

def build_HR_4D(R, T, H, X):
    ### Moment-like terms < klab~cd | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(b/cd)A(kl) h2c(c~d~k~e~) r_abb(ab~e~l~)
    x4d = (6.0 / 12.0) * ccpy_einsum("cdke,abel->abcdkl", H.bb.vvov, R.abb)
    # diagram 2: A(c/bd)A(kl) h2b(ac~ek~) r_abb(eb~d~l~)
    x4d += (6.0 / 12.0) * ccpy_einsum("acek,ebdl->abcdkl", H.ab.vvvo, R.abb)
    # diagram 3: -A(c/bd) h2c(c~m~k~l~) r_abb(ab~d~m~)
    x4d -= (3.0 / 12.0) * ccpy_einsum("cmkl,abdm->abcdkl", H.bb.vooo, R.abb)
    # diagram 13: A(c/bd) x_abb(ab~d~e~) t2c(c~e~k~l~)
    x4d += (3.0 / 12.0) * ccpy_einsum("abde,cekl->abcdkl", X["abb"]["vvvv"], T.bb)
    # diagram 14: -A(b/cd)A(kl) x_abb(ab~m~l~) t2c(c~d~k~m~)
    x4d -= (6.0 / 12.0) * ccpy_einsum("abml,cdkm->abcdkl", X["abb"]["vvoo"], T.bb)
    # diagram 15: -A(d/bc)A(kl) x_abb(mb~c~k~) t2b(ad~ml~)
    x4d -= (6.0 / 12.0) * ccpy_einsum("mbck,adml->abcdkl", X["abb"]["ovvo"], T.ab)
    ### Terms < klab~cd | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 4: A(d/bc) h1b(d~e~) r_abbb(ab~c~e~k~l~)
    x4d += (3.0 / 12.0) * ccpy_einsum("de,abcekl->abcdkl", H.b.vv, R.abbb)
    # diagram 5: h1a(ae) r_abbb(eb~c~d~k~l~)
    x4d += (1.0 / 12.0) * ccpy_einsum("ae,ebcdkl->abcdkl", H.a.vv, R.abbb)
    # diagram 6: -A(kl) h1b(ml) r_abbb(ab~c~d~k~m~)
    x4d -= (2.0 / 12.0) * ccpy_einsum("ml,abcdkm->abcdkl", H.b.oo, R.abbb)
    # diagram 7: 1/2 h2c(m~n~k~l~) r_abbb(ab~c~d~m~n~)
    x4d += (1.0 / 24.0) * ccpy_einsum("mnkl,abcdmn->abcdkl", H.bb.oooo, R.abbb)
    # diagram 8: 1/2 A(b/cd) h2c(c~d~e~f~) r_abbb(ab~e~f~k~l~)
    x4d += (3.0 / 24.0) * ccpy_einsum("cdef,abefkl->abcdkl", H.bb.vvvv, R.abbb)
    # diagram 9: A(b/cd) h2b(ab~ef~) r_abbb(ef~c~d~k~l~)
    x4d += (3.0 / 12.0) * ccpy_einsum("abef,efcdkl->abcdkl", H.ab.vvvv, R.abbb)
    # diagram 10: A(d/bc)A(kl) h2b(md~el~) r_abab(ab~ec~mk~)
    x4d += (6.0 / 12.0) * ccpy_einsum("mdel,abecmk->abcdkl", H.ab.ovvo, R.abab)
    # diagram 11: A(d/bc)A(kl) h2c(d~m~l~e~) r_abbb(ab~c~e~k~m~)
    x4d += (6.0 / 12.0) * ccpy_einsum("dmle,abcekm->abcdkl", H.bb.voov, R.abbb)
    # diagram 12: -A(kl) h2b(am~el~) r_abbb(eb~c~d~k~m~)
    x4d -= (2.0 / 12.0) * ccpy_einsum("amel,ebcdkm->abcdkl", H.ab.vovo, R.abbb)
    ### 4-body Hbar term ###
    # diagram 15: A(d/bc)A(kl) x_ab(mn~) t2b(ad~ml~) t2c(b~c~n~k~)
    x4d += (6.0 / 12.0) * ccpy_einsum("mn,adml,bcnk->abcdkl", X["ab"]["oo"], T.ab, T.bb)
    # antisymmetrize A(bcd)A(kl)
    x4d -= np.transpose(x4d, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4d -= np.transpose(x4d, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4d -= np.transpose(x4d, (0, 2, 1, 3, 4, 5)) + np.transpose(x4d, (0, 3, 2, 1, 4, 5)) # A(b/cd)
    return x4d
