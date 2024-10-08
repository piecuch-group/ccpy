"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DEA-EOMCC approach with up to 4p-2h excitations"""
import numpy as np
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
    x2b = np.einsum("ae,eb->ab", H.a.vv, R.ab, optimize=True)
    x2b += np.einsum("be,ae->ab", H.b.vv, R.ab, optimize=True)
    x2b += np.einsum("abef,ef->ab", H.ab.vvvv, R.ab, optimize=True)
    x2b += np.einsum("me,abem->ab", H.a.ov, R.aba, optimize=True)
    x2b += np.einsum("me,abem->ab", H.b.ov, R.abb, optimize=True)
    x2b += np.einsum("nbfe,aefn->ab", H.ab.ovvv, R.aba, optimize=True)
    x2b += 0.5 * np.einsum("anef,ebfn->ab", H.aa.vovv, R.aba, optimize=True)
    x2b += 0.5 * np.einsum("bnef,aefn->ab", H.bb.vovv, R.abb, optimize=True)
    x2b += np.einsum("anef,ebfn->ab", H.ab.vovv, R.abb, optimize=True)
    # additional R(4p-2h) terms
    x2b += 0.25 * np.einsum("mnef,abefmn->ab", H.aa.oovv, R.abaa, optimize=True)
    x2b += np.einsum("mnef,abefmn->ab", H.ab.oovv, R.abab, optimize=True)
    x2b += 0.25 * np.einsum("mnef,abefmn->ab", H.bb.oovv, R.abbb, optimize=True)
    return x2b

def build_HR_3B(R, T, H, X):
    # (1)
    x3b = 0.5 * np.einsum("cake,eb->abck", H.aa.vvov, R.ab, optimize=True)
    # (2)
    x3b += np.einsum("cbke,ae->abck", H.ab.vvov, R.ab, optimize=True)
    # (4)
    x3b += np.einsum("ae,ebck->abck", H.a.vv, R.aba, optimize=True)
    # (5)
    x3b += 0.5 * np.einsum("be,aeck->abck", H.b.vv, R.aba, optimize=True)
    # (6)
    x3b += np.einsum("abef,efck->abck", H.ab.vvvv, R.aba, optimize=True)
    # (7)
    x3b += 0.25 * np.einsum("acef,ebfk->abck", H.aa.vvvv, R.aba, optimize=True)
    # (8)
    x3b += np.einsum("cmke,abem->abck", H.aa.voov, R.aba, optimize=True)
    # (9)
    x3b += np.einsum("cmke,abem->abck", H.ab.voov, R.abb, optimize=True)
    # (10)
    x3b -= 0.5 * np.einsum("mbke,aecm->abck", H.ab.ovov, R.aba, optimize=True)
    # (3) + (11)
    x3b -= 0.5 * np.einsum("mb,acmk->abck", X["ab"]["ov"], T.aa, optimize=True)
    # (3) + (12)
    x3b -= np.einsum("am,cbkm->abck", X["ab"]["vo"], T.ab, optimize=True)
    # (13)
    x3b -= 0.5 * np.einsum("mk,abcm->abck", H.a.oo, R.aba, optimize=True)
    # additional R(4p-2h) terms
    x3b += 0.5 * np.einsum("me,abcekm->abck", H.a.ov, R.abaa, optimize=True)
    x3b += 0.5 * np.einsum("me,abcekm->abck", H.b.ov, R.abab, optimize=True)
    x3b -= 0.25 * np.einsum("mnkf,abcfmn->abck", H.aa.ooov, R.abaa, optimize=True)
    x3b -= 0.5 * np.einsum("mnkf,abcfmn->abck", H.ab.ooov, R.abab, optimize=True)
    x3b += 0.5 * np.einsum("cnef,abefkn->abck", H.aa.vovv, R.abaa, optimize=True)
    x3b += np.einsum("cnef,abefkn->abck", H.ab.vovv, R.abab, optimize=True)
    x3b += 0.5 * np.einsum("nbfe,aecfkn->abck", H.ab.ovvv, R.abaa, optimize=True)
    x3b += 0.25 * np.einsum("bnef,aecfkn->abck", H.bb.vovv, R.abab, optimize=True)
    # antisymmetrize A(ac)
    x3b -= np.einsum("abck->cbak", x3b, optimize=True)
    return x3b

def build_HR_3C(R, T, H, X):
    # (1)
    x3c = np.einsum("acek,eb->abck", H.ab.vvvo, R.ab, optimize=True)
    # (2)
    x3c += 0.5 * np.einsum("cbke,ae->abck", H.bb.vvov, R.ab, optimize=True)
    # (4)
    x3c += 0.5 * np.einsum("ae,ebck->abck", H.a.vv, R.abb, optimize=True)
    # (5)
    x3c += np.einsum("be,aeck->abck", H.b.vv, R.abb, optimize=True)
    # (6)
    x3c += np.einsum("abef,efck->abck", H.ab.vvvv, R.abb, optimize=True)
    # (7)
    x3c += 0.25 * np.einsum("bcef,aefk->abck", H.bb.vvvv, R.abb, optimize=True)
    # (8)
    x3c += np.einsum("mcek,abem->abck", H.ab.ovvo, R.aba, optimize=True)
    # (9)
    x3c += np.einsum("cmke,abem->abck", H.bb.voov, R.abb, optimize=True)
    # (10)
    x3c -= 0.5 * np.einsum("amek,ebcm->abck", H.ab.vovo, R.abb, optimize=True)
    # (3) + (11)
    x3c -= np.einsum("mb,acmk->abck", X["ab"]["ov"], T.ab, optimize=True)
    # (3) + (12)
    x3c -= 0.5 * np.einsum("am,bcmk->abck", X["ab"]["vo"], T.bb, optimize=True)
    # (13)
    x3c -= 0.5 * np.einsum("mk,abcm->abck", H.b.oo, R.abb, optimize=True)
    # additional R(4p-2h) terms
    x3c += 0.5 * np.einsum("me,abecmk->abck", H.a.ov, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("me,abecmk->abck", H.b.ov, R.abbb, optimize=True)
    x3c -= 0.5 * np.einsum("nmfk,abfcnm->abck", H.ab.oovo, R.abab, optimize=True)
    x3c -= 0.25 * np.einsum("mnkf,abcfmn->abck", H.bb.ooov, R.abbb, optimize=True)
    x3c += np.einsum("ncfe,abfenk->abck", H.ab.ovvv, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("cnef,abefkn->abck", H.bb.vovv, R.abbb, optimize=True)
    x3c += 0.25 * np.einsum("anef,ebfcnk->abck", H.aa.vovv, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("anef,ebfcnk->abck", H.ab.vovv, R.abbb, optimize=True)
    # antisymmetrize A(b~c~)
    x3c -= np.einsum("abck->acbk", x3c, optimize=True)
    return x3c

def build_HR_4B(R, T, H, X):
    ### Moment-like terms < klab~cd | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(a/cd)A(kl) h2a(dcek) r_aba(ab~el)
    x4b = (6.0 / 12.0) * np.einsum("cdke,abel->abcdkl", H.aa.vvov, R.aba, optimize=True)
    # diagram 2: A(c/ad)A(kl) h2b(cb~ke~) r_aba(ae~dl)
    x4b += (6.0 / 12.0) * np.einsum("cbke,aedl->abcdkl", H.ab.vvov, R.aba, optimize=True)
    # diagram 3: -A(c/ad) h2a(cmkl) r_aba(ab~dm)
    x4b -= (3.0 / 12.0) * np.einsum("cmkl,abdm->abcdkl", H.aa.vooo, R.aba, optimize=True)
    # diagram 13: -A(a/cd)A(kl) x_aba(ab~ml) t2a(cdkm)
    x4b -= (6.0 / 12.0) * np.einsum("abml,cdkm->abcdkl", X["aba"]["vvoo"], T.aa, optimize=True)
    # diagram 14: A(c/ad) x_aba(ab~de) t2a(cekl)
    x4b += (3.0 / 12.0) * np.einsum("abde,cekl->abcdkl", X["aba"]["vvvv"], T.aa, optimize=True)
    # diagram 15: -A(b/ac)A(kl) x_aba(am~ck) t2b(db~lm~)
    x4b -= (6.0 / 12.0) * np.einsum("amck,dblm->abcdkl", X["aba"]["vovo"], T.ab, optimize=True)
    ### Terms < klab~cd | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 4: A(d/ac) h1a(de) r_abaa(ab~cekl)
    x4b += (3.0 / 12.0) * np.einsum("de,abcekl->abcdkl", H.a.vv, R.abaa, optimize=True)
    # diagram 5: h1b(b~e~) r_abaa(ae~cdkl)
    x4b += (1.0 / 12.0) * np.einsum("be,aecdkl->abcdkl", H.b.vv, R.abaa, optimize=True)
    # diagram 6: -A(kl) h1a(ml) r_abaa(ab~cdkm)
    x4b -= (2.0 / 12.0) * np.einsum("ml,abcdkm->abcdkl", H.a.oo, R.abaa, optimize=True)
    # diagram 7: 1/2 h2a(mnkl) r_abaa(ab~cdmn)
    x4b += (1.0 / 24.0) * np.einsum("mnkl,abcdmn->abcdkl", H.aa.oooo, R.abaa, optimize=True)
    # diagram 8: 1/2 A(a/cd) h2a(cdef) r_abaa(ab~efkl)
    x4b += (3.0 / 24.0) * np.einsum("cdef,abefkl->abcdkl", H.aa.vvvv, R.abaa, optimize=True)
    # diagram 9: A(a/cd) h2b(ab~ef~) r_abaa(ef~cdkl)
    x4b += (3.0 / 12.0) * np.einsum("abef,efcdkl->abcdkl", H.ab.vvvv, R.abaa, optimize=True)
    # diagram 10: A(d/ac)A(kl) h2a(dmle) r_abaa(ab~cekm)
    x4b += (6.0 / 12.0) * np.einsum("dmle,abcekm->abcdkl", H.aa.voov, R.abaa, optimize=True)
    # diagram 11: A(d/ac)A(kl) h2b(dm~le~) r_abab(ab~ce~km~)
    x4b += (6.0 / 12.0) * np.einsum("dmle,abcekm->abcdkl", H.ab.voov, R.abab, optimize=True)
    # diagram 12: -A(kl) h2b(mb~le~) r_abaa(ae~cdkm)
    x4b -= (2.0 / 12.0) * np.einsum("mble,aecdkm->abcdkl", H.ab.ovov, R.abaa, optimize=True)
    ### 4-body Hbar term ###
    # diagram 15: A(c/ad)A(kl) x_ab(mn~) t2a(adml) t2b(cb~kn~)
    x4b += (6.0 / 12.0) * np.einsum("mn,adml,cbkn->abcdkl", X["ab"]["oo"], T.aa, T.ab, optimize=True)
    # antisymmetrize A(acd)A(kl)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (2, 1, 0, 3, 4, 5)) + np.transpose(x4b, (3, 1, 2, 0, 4, 5)) # A(a/cd)
    return x4b

def build_HR_4C(R, T, H, X):
    ### Moment-like terms < kl~ab~cd~ | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(ac)A(bd) h2b(cd~el~) r_aba(ab~ek)
    x4c = np.einsum("cdel,abek->abcdkl", H.ab.vvvo, R.aba, optimize=True)
    # diagram 2: A(bd)A(ac) h2b(cd~ke~) r_abb(ab~e~l~)
    x4c += np.einsum("cdke,abel->abcdkl", H.ab.vvov, R.abb, optimize=True)
    # diagram 3: -A(bd) h2b(md~kl~) r_aba(ab~cm)
    x4c -= (2.0 / 4.0) * np.einsum("mdkl,abcm->abcdkl", H.ab.ovoo, R.aba, optimize=True)
    # diagram 4: -A(ac) h2b(cm~kl~) r_abb(ab~d~m~)
    x4c -= (2.0 / 4.0) * np.einsum("cmkl,abdm->abcdkl", H.ab.vooo, R.abb, optimize=True)
    # diagram 25: h2a(cake) * r_abb(eb~d~l~)
    x4c += (1.0 / 4.0) * np.einsum("cake,ebdl->abcdkl", H.aa.vvov, R.abb, optimize=True)
    # diagram 26: h2c(d~b~l~e~) r_aba(ae~ck)
    x4c += (1.0 / 4.0) * np.einsum("dble,aeck->abcdkl", H.bb.vvov, R.aba, optimize=True)
    # diagram 19: A(ac) x_abb(ab~d~e~) t2b(ce~kl~)
    x4c += (2.0 / 4.0) * np.einsum("abde,cekl->abcdkl", X["abb"]["vvvv"], T.ab, optimize=True)
    # diagram 20: A(bd) x_aba(ab~ce) t2b(ed~kl~)
    x4c += (2.0 / 4.0) * np.einsum("abce,edkl->abcdkl", X["aba"]["vvvv"], T.ab, optimize=True)
    # diagram 21: -A(ac)A(bd) x_aba(ab~mk) t2b(cd~ml~)
    x4c -= np.einsum("abmk,cdml->abcdkl", X["aba"]["vvoo"], T.ab, optimize=True)
    # diagram 22: -A(ac)A(bd) x_abb(ab~m~l~) t2b(cd~km~)
    x4c -= np.einsum("abml,cdkm->abcdkl", X["abb"]["vvoo"], T.ab, optimize=True)
    # diagram 23: -x_aba(am~ck) t2c(b~d~m~l~)
    x4c -= (1.0 / 4.0) * np.einsum("amck,bdml->abcdkl", X["aba"]["vovo"], T.bb, optimize=True)
    # diagram 24: -x_abb(mb~d~l~) t2a(acmk)
    x4c -= (1.0 / 4.0) * np.einsum("mbdl,acmk->abcdkl", X["abb"]["ovvo"], T.aa, optimize=True)
    ### Terms < kl~ab~cd~ | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 5: A(ac) h1a(ae) r_abab(eb~cd~kl~)
    x4c += (2.0 / 4.0) * np.einsum("ae,ebcdkl->abcdkl", H.a.vv, R.abab, optimize=True)
    # diagram 6: A(bd) h1b(be) r_abab(ae~cd~kl~)
    x4c += (2.0 / 4.0) * np.einsum("be,aecdkl->abcdkl", H.b.vv, R.abab, optimize=True)
    # diagram 7: -h1a(mk) r_abab(ab~cd~ml~)
    x4c -= (1.0 / 4.0) * np.einsum("mk,abcdml->abcdkl", H.a.oo, R.abab, optimize=True)
    # diagram 8: -h1b(ml) r_abab(ab~cd~km~)
    x4c -= (1.0 / 4.0) * np.einsum("ml,abcdkm->abcdkl", H.b.oo, R.abab, optimize=True)
    # diagram 9: h2b(mn~kl~) r_abab(ab~cd~mn~)
    x4c += (1.0 / 4.0) * np.einsum("mnkl,abcdmn->abcdkl", H.ab.oooo, R.abab, optimize=True)
    # diagram 10: A(ac)A(bd) h2b(cd~ef~) r_abab(ab~ef~kl~)
    x4c += np.einsum("cdef,abefkl->abcdkl", H.ab.vvvv, R.abab, optimize=True)
    # diagram 11: 1/2 h2a(acef) r_abab(eb~fd~kl~)
    x4c += (1.0 / 8.0) * np.einsum("acef,ebfdkl->abcdkl", H.aa.vvvv, R.abab, optimize=True)
    # diagram 12: 1/2 h2c(b~d~e~f~) r_abab(ae~cf~kl~)
    x4c += (1.0 / 8.0) * np.einsum("bdef,aecfkl->abcdkl", H.bb.vvvv, R.abab, optimize=True)
    # diagram 13: A(ac) h2a(cmke) r_abab(ab~ed~ml~)
    x4c += (2.0 / 4.0) * np.einsum("cmke,abedml->abcdkl", H.aa.voov, R.abab, optimize=True)
    # diagram 14: A(ac) h2b(cm~ke~) r_abbb(ab~e~d~m~l~)
    x4c += (2.0 / 4.0) * np.einsum("cmke,abedml->abcdkl", H.ab.voov, R.abbb, optimize=True)
    # diagram 15: A(bd) h2b(md~el~) r_abaa(ab~cekm)
    x4c += (2.0 / 4.0) * np.einsum("mdel,abcekm->abcdkl", H.ab.ovvo, R.abaa, optimize=True)
    # diagram 16: A(bd) h2c(d~m~l~e~) r_abab(ab~ce~km~)
    x4c += (2.0 / 4.0) * np.einsum("dmle,abcekm->abcdkl", H.bb.voov, R.abab, optimize=True)
    # diagram 17: -A(bd) h2b(md~ke~) r_abab(ab~ce~ml~)
    x4c -= (2.0 / 4.0) * np.einsum("mdke,abceml->abcdkl", H.ab.ovov, R.abab, optimize=True)
    # diagram 18: -A(ac) h2b(cm~el~) r_abab(ab~ed~km~)
    x4c -= (2.0 / 4.0) * np.einsum("cmel,abedkm->abcdkl", H.ab.vovo, R.abab, optimize=True)
    ### 4-body HBar ###
    # diagram 23: x_ab(mn~) t2a(acmk) t2c(b~d~n~l~)
    x4c += (1.0 / 4.0) * np.einsum("mn,acmk,bdnl->abcdkl", X["ab"]["oo"], T.aa, T.bb, optimize=True)
    # diagram 24: A(ac)A(bd) x_ab(mn~) t2b(ad~ml~) t2b(cb~kn~)
    x4c += np.einsum("mn,adml,cbkn->abcdkl", X["ab"]["oo"], T.ab, T.ab, optimize=True)
    # antisymmetrize A(ac)A(bd)
    x4c -= np.transpose(x4c, (2, 1, 0, 3, 4, 5)) # A(ac)
    x4c -= np.transpose(x4c, (0, 3, 2, 1, 4, 5)) # A(bd)
    return x4c

def build_HR_4D(R, T, H, X):
    ### Moment-like terms < klab~cd | (H(2)[R(2p) + R(3p-1h)])_C | 0 > ###
    # diagram 1: A(b/cd)A(kl) h2c(c~d~k~e~) r_abb(ab~e~l~)
    x4d = (6.0 / 12.0) * np.einsum("cdke,abel->abcdkl", H.bb.vvov, R.abb, optimize=True)
    # diagram 2: A(c/bd)A(kl) h2b(ac~ek~) r_abb(eb~d~l~)
    x4d += (6.0 / 12.0) * np.einsum("acek,ebdl->abcdkl", H.ab.vvvo, R.abb, optimize=True)
    # diagram 3: -A(c/bd) h2c(c~m~k~l~) r_abb(ab~d~m~)
    x4d -= (3.0 / 12.0) * np.einsum("cmkl,abdm->abcdkl", H.bb.vooo, R.abb, optimize=True)
    # diagram 13: A(c/bd) x_abb(ab~d~e~) t2c(c~e~k~l~)
    x4d += (3.0 / 12.0) * np.einsum("abde,cekl->abcdkl", X["abb"]["vvvv"], T.bb, optimize=True)
    # diagram 14: -A(b/cd)A(kl) x_abb(ab~m~l~) t2c(c~d~k~m~)
    x4d -= (6.0 / 12.0) * np.einsum("abml,cdkm->abcdkl", X["abb"]["vvoo"], T.bb, optimize=True)
    # diagram 15: -A(d/bc)A(kl) x_abb(mb~c~k~) t2b(ad~ml~)
    x4d -= (6.0 / 12.0) * np.einsum("mbck,adml->abcdkl", X["abb"]["ovvo"], T.ab, optimize=True)
    ### Terms < klab~cd | (H(2)R(4p-2h)_C | 0 > ###
    # diagram 4: A(d/bc) h1b(d~e~) r_abbb(ab~c~e~k~l~)
    x4d += (3.0 / 12.0) * np.einsum("de,abcekl->abcdkl", H.b.vv, R.abbb, optimize=True)
    # diagram 5: h1a(ae) r_abbb(eb~c~d~k~l~)
    x4d += (1.0 / 12.0) * np.einsum("ae,ebcdkl->abcdkl", H.a.vv, R.abbb, optimize=True)
    # diagram 6: -A(kl) h1b(ml) r_abbb(ab~c~d~k~m~)
    x4d -= (2.0 / 12.0) * np.einsum("ml,abcdkm->abcdkl", H.b.oo, R.abbb, optimize=True)
    # diagram 7: 1/2 h2c(m~n~k~l~) r_abbb(ab~c~d~m~n~)
    x4d += (1.0 / 24.0) * np.einsum("mnkl,abcdmn->abcdkl", H.bb.oooo, R.abbb, optimize=True)
    # diagram 8: 1/2 A(b/cd) h2c(c~d~e~f~) r_abbb(ab~e~f~k~l~)
    x4d += (3.0 / 24.0) * np.einsum("cdef,abefkl->abcdkl", H.bb.vvvv, R.abbb, optimize=True)
    # diagram 9: A(b/cd) h2b(ab~ef~) r_abbb(ef~c~d~k~l~)
    x4d += (3.0 / 12.0) * np.einsum("abef,efcdkl->abcdkl", H.ab.vvvv, R.abbb, optimize=True)
    # diagram 10: A(d/bc)A(kl) h2b(md~el~) r_abab(ab~ec~mk~)
    x4d += (6.0 / 12.0) * np.einsum("mdel,abecmk->abcdkl", H.ab.ovvo, R.abab, optimize=True)
    # diagram 11: A(d/bc)A(kl) h2c(d~m~l~e~) r_abbb(ab~c~e~k~m~)
    x4d += (6.0 / 12.0) * np.einsum("dmle,abcekm->abcdkl", H.bb.voov, R.abbb, optimize=True)
    # diagram 12: -A(kl) h2b(am~el~) r_abbb(eb~c~d~k~m~)
    x4d -= (2.0 / 12.0) * np.einsum("amel,ebcdkm->abcdkl", H.ab.vovo, R.abbb, optimize=True)
    ### 4-body Hbar term ###
    # diagram 15: A(d/bc)A(kl) x_ab(mn~) t2b(ad~ml~) t2c(b~c~n~k~)
    x4d += (6.0 / 12.0) * np.einsum("mn,adml,bcnk->abcdkl", X["ab"]["oo"], T.ab, T.bb, optimize=True)
    # antisymmetrize A(bcd)A(kl)
    x4d -= np.transpose(x4d, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4d -= np.transpose(x4d, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4d -= np.transpose(x4d, (0, 2, 1, 3, 4, 5)) + np.transpose(x4d, (0, 3, 2, 1, 4, 5)) # A(b/cd)
    return x4d
