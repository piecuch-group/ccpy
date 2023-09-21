"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DEA-EOMCC approach with up to 4p-2h excitations"""
import numpy as np
from ccpy.eomcc.deaeom4_intermediates import get_deaeom4_intermediates
from ccpy.utilities.updates import cc_loops2

def update(R, omega, H, system):
    R.ab, R.aba, R.abb, R.abaa, R.abab, R.abbb = cc_loops2.cc_loops2.update_r_4p2h(
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
    return R

def HR(dR, R, T, H, flag_RHF, system):

    X = get_deaeom4_intermediates(H, R, T, system)
    # update R2
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aba = build_HR_3B(R, T, H, X)
    dR.abb = build_HR_3C(R, T, H, X)
    # update R4
    dR.abaa = build_HR_4B(R, T, H, X)
    dR.abab = build_HR_4C(R, T, H, X)
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
    x3c += 0.5 * np.einsum("me,abeckm->abck", H.a.ov, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("me,abeckm->abck", H.b.ov, R.abbb, optimize=True)
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
    # diagram 1: A(a/cd)A(kl) h2a(dcek) r_aba(ab~el)
    x4b = (6.0 / 12.0) * np.einsum("cdke,abel->abcdkl", H.aa.vvov, R.aba, optimize=True)
    # diagram 2: A(c/ad)A(kl) h2b(cb~ke~) r_aba(ae~dl)
    x4b += (6.0 / 12.0) * np.einsum("cbke,aedl->abcdkl", H.ab.vvov, R.aba, optimize=True)
    # diagram 3: -A(c/ad) h2a(cmkl) r_aba(ab~dm)
    x4b -= (3.0 / 12.0) * np.einsum("cmkl,abdm->abcdkl", H.aa.vooo, R.aba, optimize=True)
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
    # diagram 13: -A(a/cd)A(kl) x_aba(ab~ml) t2a(cdkm)
    x4b -= (6.0 / 12.0) * np.einsum("abml,cdkm->abcdkl", X["aba"]["vvoo"], T.aa, optimize=True)
    # diagram 14: A(c/ad) x_aba(ab~de) t2a(cekl)
    x4b += (3.0 / 12.0) * np.einsum("abde,cekl->abcdkl", X["aba"]["vvvv"], T.aa, optimize=True)
    # antisymmetrize A(acd)A(kl)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (2, 1, 0, 3, 4, 5)) + np.transpose(x4b, (3, 1, 2, 0, 4, 5)) # A(a/cd)
    return x4b

def build_HR_4C(R, T, H, X):
    # diagram 1: A(ac) h2b(cdel) r_aba(ab~ek)
    x4c = (1.0 / 2.0) * np.einsum("cdel,abek->abcdkl", H.ab.vvvo, R.aba, optimize=True)
    # diagram 2:
    # antisymmetrize A(ac)
    x4c -= np.transpose(x4c, (2, 1, 0, 3, 4, 5))
    return x4c

def build_HR_4D(R, T, H, X):
    pass