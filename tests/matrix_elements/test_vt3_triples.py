"""In this script, we test out the idea of using the 'amplitude-driven' approach
to constructing the sparse triples projection < ijkabc | (H(2) * T3)_C | 0 >, where
T3 is sparse and defined over a given list of triples."""
import numpy as np
import time

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates

from ccpy.utilities.updates import ccp_linear_loops

# linear CC(P) algorithm:
# x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
#                 = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
# e, f, c, i, j, k = T3_excitations["aaa"][idet]
# x3a[:, :, c, i, j, k] += 0.5 * H.aa.vvvv[:, :, e, f] * t_amp # (1)
# x3a[:, :, e, i, j, k] -= 0.5 * H.aa.vvvv[:, :, c, f] * t_amp # (ec)
# x3a[:, :, f, i, j, k] -= 0.5 * H.aa.vvvv[:, :, e, c] * t_amp # (fc)

# quadratic CC(P) algorithm:
# x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
#              = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
# e, f, c, i, j, k = T3_excitations["aaa"][idet]
# vec1 = pspace[:, :, c, i, j, k]
# vec2 = pspace[:, :, e, i, j, k]
# vec3 = pspace[:, :, f, i, j, k]
# for a in range(nua):
#   for b in range(e + 1, nua):
#       if vec1[a, b]:
#           x3a[a, b, c, i, j, k] += H.aa.vvvv[a, b, e, f] * t_amp # (1)
#       if vec2[a, b]:
#           x3a[a, b, e, i, j, k] -= H.aa.vvvv[a, b, c, f] * t_amp # (ec)
#       if vec3[a, b]:
#           x3a[a, b, f, i, j, k] -= H.aa.vvvv[a, b, e, c] * t_amp # (fc)

# space-saving quadratic CC(P) algorithm (operates at ~2x CPU cost of the original quadratic CC(P) method):
# the function tuple_to_idx() is an invertible map, something like np.ravel/np.unravel or Cantor pairing function
# x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
#              = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
# e, f, c, i, j, k = T3_excitations["aaa"][idet]
# vec1 = pspace[:, :, c, i, j, k]
# vec2 = pspace[:, :, e, i, j, k]
# vec3 = pspace[:, :, f, i, j, k]
# for a in range(nua):
#   for b in range(e + 1, nua):
#       if vec1[a, b]:
#           idx = tuple_to_idx(a, b, c, i, j, k)
#           x3a[idx] += H.aa.vvvv[a, b, e, f] * t_amp # (1)
#       if vec2[a, b]:
#           idx = tuple_to_idx(a, b, e, i, j, k)
#           x3a[idx] -= H.aa.vvvv[a, b, c, f] * t_amp # (ec)
#       if vec3[a, b]:
#           idx = tuple_to_idx(a, b, f, i, j, k)
#           x3a[idx] -= H.aa.vvvv[a, b, e, c] * t_amp # (fc)

def get_T3_list(T):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    T3_excitations = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    T3_amplitudes = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}

    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(j + 1, noa):
                            T3_excitations["aaa"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["aaa"].append(T.aaa[a, b, c, i, j, k])
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(nob):
                            T3_excitations["aab"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["aab"].append(T.aab[a, b, c, i, j, k])
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j + 1, nob):
                            T3_excitations["abb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["abb"].append(T.abb[a, b, c, i, j, k])
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        for k in range(j + 1, nob):
                            T3_excitations["bbb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["bbb"].append(T.bbb[a, b, c, i, j, k])

    for key in T3_excitations.keys():
        T3_excitations[key] = np.asarray(T3_excitations[key])
        T3_amplitudes[key] = np.asarray(T3_amplitudes[key])

    return T3_excitations, T3_amplitudes

def contract_vt3_exact(H0, H, T):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    x1a = 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa, optimize=True)
    x1a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T.aab, optimize=True)
    x1a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)

    x1b = 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb, optimize=True)
    x1b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T.aab, optimize=True)
    x1b += np.einsum("mnef,efamni->ai", H.ab.oovv, T.abb, optimize=True)

    x2a = 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    x2a += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    x2a -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T.aab, optimize=True)
    x2a -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T.aaa, optimize=True)
    x2a += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T.aaa, optimize=True)
    x2a += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T.aab, optimize=True)

    x2b = -0.5 * np.einsum("mnif,afbmnj->abij", H0.aa.ooov + H.aa.ooov, T.aab, optimize=True)
    x2b -= np.einsum("nmfj,afbinm->abij", H0.ab.oovo + H.ab.oovo, T.aab, optimize=True)
    x2b -= 0.5 * np.einsum("mnjf,afbinm->abij", H0.bb.ooov + H.bb.ooov, T.abb, optimize=True)
    x2b -= np.einsum("mnif,afbmnj->abij", H0.ab.ooov + H.ab.ooov, T.abb, optimize=True)
    x2b += 0.5 * np.einsum("anef,efbinj->abij", H0.aa.vovv + H.aa.vovv, T.aab, optimize=True)
    x2b += np.einsum("anef,efbinj->abij", H0.ab.vovv + H.ab.vovv, T.abb, optimize=True)
    x2b += np.einsum("nbfe,afeinj->abij", H0.ab.ovvv + H.ab.ovvv, T.aab, optimize=True)
    x2b += 0.5 * np.einsum("bnef,afeinj->abij", H0.bb.vovv + H.bb.vovv, T.abb, optimize=True)
    x2b += np.einsum("me,aebimj->abij", H.a.ov, T.aab, optimize=True)
    x2b += np.einsum("me,aebimj->abij", H.b.ov, T.abb, optimize=True)

    x2c = 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    x2c += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    x2c += 0.25 * np.einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T.bbb, optimize=True)
    x2c += 0.5 * np.einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T.abb, optimize=True)
    x2c -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T.bbb, optimize=True)
    x2c -= 0.5 * np.einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T.abb, optimize=True)

    I2A_vvov = (
        H.aa.vvov - 0.5 * np.einsum("mnef,abfimn->abie", H.aa.oovv, T.aaa, optimize=True)
                  - np.einsum("mnef,abfimn->abie", H.ab.oovv, T.aab, optimize=True)
    )
    I2A_vooo = (
        H.aa.vooo + 0.5 * np.einsum("mnef,aefijn->amij", H.aa.oovv, T.aaa, optimize=True)
                  + np.einsum("mnef,aefijn->amij", H.ab.oovv, T.aab, optimize=True)
                  - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    )
    I2B_vvvo =(
        H.ab.vvvo - 0.5 * np.einsum("mnef,afbmnj->abej", H.aa.oovv, T.aab, optimize=True)
                  - np.einsum("mnef,afbmnj->abej", H.ab.oovv, T.abb, optimize=True)
    ) 
    I2B_ovoo = (
        H.ab.ovoo + 0.5 * np.einsum("mnef,efbinj->mbij", H.aa.oovv, T.aab, optimize=True)
                  + np.einsum("mnef,efbinj->mbij", H.ab.oovv, T.abb, optimize=True)
                  - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    )
    I2B_vvov = (
        H.ab.vvov - np.einsum("nmfe,afbinm->abie", H.ab.oovv, T.aab, optimize=True)
                  - 0.5 * np.einsum("nmfe,afbinm->abie", H.bb.oovv, T.abb, optimize=True)
    )
    I2B_vooo = (
        H.ab.vooo + np.einsum("nmfe,afeinj->amij", H.ab.oovv, T.aab, optimize=True)
                  + 0.5 * np.einsum("nmfe,afeinj->amij", H.bb.oovv, T.abb, optimize=True)
                  - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    )
    I2C_vvov = (
        H.bb.vvov - 0.5 * np.einsum("mnef,abfimn->abie", H.bb.oovv, T.bbb, optimize=True)
                  - np.einsum("nmfe,fbanmi->abie", H.ab.oovv, T.abb, optimize=True)
    )
    I2C_vooo = (
        H.bb.vooo + 0.5 * np.einsum("mnef,aefijn->amij", H.bb.oovv, T.bbb, optimize=True)
                  + np.einsum("nmfe,feanji->amij", H.ab.oovv, T.abb, optimize=True)
                  - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    )

    # MM(2,3)
    x3a = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    x3a += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (Hbar*T3)
    x3a -= (3.0 / 36.0) * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aaa, optimize=True)
    x3a += (3.0 / 36.0) * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aaa, optimize=True)
    x3a += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True) # (k/ij) = 3
    x3a += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) # (c/ab) = 3
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) # (c/ij)(k/ij) = 9
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) # (c/ij)(k/ij) = 9

    # MM(2,3)
    x3b = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    x3b -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    x3b += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    x3b -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    x3b += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    x3b -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
    # (Hbar*T3)
    x3b -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
    x3b -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
    x3b += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True)
    x3b += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True)
    x3b += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
    x3b += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
    x3b += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True)
    x3b += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True)
    x3b += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True)
    x3b += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True)
    x3b += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True)
    x3b += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True)
    x3b -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True)
    x3b -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True)

    # MM(2,3)
    x3c = 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, T.bb, optimize=True)
    x3c -= 0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, T.bb, optimize=True)
    x3c += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, T.ab, optimize=True)
    x3c -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, T.ab, optimize=True)
    x3c += np.einsum("abej,ecik->abcijk", I2B_vvvo, T.ab, optimize=True)
    x3c -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab, optimize=True)
    # (Hbar*T3)
    x3c -= 0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
    x3c -= 0.5 * np.einsum("mj,abcimk->abcijk", H.b.oo, T.abb, optimize=True)
    x3c += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.abb, optimize=True)
    x3c += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, T.abb, optimize=True)
    x3c += 0.125 * np.einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb, optimize=True)
    x3c += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb, optimize=True)
    x3c += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb, optimize=True)
    x3c += 0.5 * np.einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb, optimize=True)
    x3c += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True)
    x3c += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True)
    x3c += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True)
    x3c += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True)
    x3c -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True)
    x3c -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True)

    # MM(2,3)
    x3d = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
    x3d += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)
    # (Hbar*T3)
    x3d -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    x3d += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    x3d += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    x3d += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    x3d += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    x3d += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)

    x2a -= np.transpose(x2a, (1, 0, 2, 3))
    x2a -= np.transpose(x2a, (0, 1, 3, 2))

    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    x2c -= np.transpose(x2c, (0, 1, 3, 2))

    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)

    x3b -= np.transpose(x3b, (1, 0, 2, 3, 4, 5)) # (ab)
    x3b -= np.transpose(x3b, (0, 1, 2, 4, 3, 5)) # (ij)

    x3c -= np.transpose(x3c, (0, 2, 1, 3, 4, 5)) # (bc)
    x3c -= np.transpose(x3c, (0, 1, 2, 3, 5, 4)) # (jk)

    x3d -= np.transpose(x3d, (0, 1, 2, 3, 5, 4)) # (jk)
    x3d -= np.transpose(x3d, (0, 1, 2, 4, 3, 5)) + np.transpose(x3d, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3d -= np.transpose(x3d, (0, 2, 1, 3, 4, 5)) # (bc)
    x3d -= np.transpose(x3d, (2, 1, 0, 3, 4, 5)) + np.transpose(x3d, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return x1a, x1b, x2a, x2b, x2c, x3a, x3b, x3c, x3d

def contract_vt3_fly(H, H0, T, T3_excitations, T3_amplitudes):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # Residual containers
    x3a = np.zeros((nua, nua, nua, noa, noa, noa))
    x3b = np.zeros((nua, nua, nub, noa, noa, nob))
    x3c = np.zeros((nua, nub, nub, noa, nob, nob))
    x3d = np.zeros((nub, nub, nub, nob, nob, nob))

    # Intermediate containers
    I2A_vooo = 0.5 * (H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True))
    I2A_vvov = 0.5 * H.aa.vvov
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vvov = H.ab.vvov.copy()
    I2B_vvvo = H.ab.vvvo.copy()
    I2C_vooo = 0.5 * (H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True))
    I2C_vvov = 0.5 * H.bb.vvov

    # Loop over aaa determinants
    for idet in range(len(T3_amplitudes["aaa"])):

        # Get the particular aaa T3 amplitude
        t_amp = T3_amplitudes["aaa"][idet]

        # x3a(abcijk) <- -A(abc)A(i/jk)A(jk)A(m/jk) h1a(mi) * t3a(abcmjk)
        #              = -A(abc)A(ijk)[ A(m/jk) h1a(mi) * t3a(abcmjk) ]
        a, b, c, m, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3a[a, b, c, :, j, k] -= H.a.oo[m, :] * t_amp # (1)
        x3a[a, b, c, :, m, k] += H.a.oo[j, :] * t_amp # (mj)
        x3a[a, b, c, :, j, m] += H.a.oo[k, :] * t_amp # (mk)

        # x3a(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1a(ae) * t3a(ebcijk)
        #              = A(abc)A(ijk)[ A(e/bc) h1a(ae) * t3a(ebcijk) ]
        e, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3a[:, b, c, i, j, k] += H.a.vv[:, e] * t_amp # (1)
        x3a[:, e, c, i, j, k] -= H.a.vv[:, b] * t_amp # (be)
        x3a[:, b, e, i, j, k] -= H.a.vv[:, c] * t_amp # (ce)

        # x3a(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2a(mnij) * t3a(abcmnk) ]
        #             = A(abc)A(ijk)[ 1/2 A(k/mn) h2a(mnij) * t3a(abcmnk) ]
        a, b, c, m, n, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3a[a, b, c, :, :, k] += 0.5 * H.aa.oooo[m, n, :, :] * t_amp # (1)
        x3a[a, b, c, :, :, m] -= 0.5 * H.aa.oooo[k, n, :, :] * t_amp # (mk)
        x3a[a, b, c, :, :, n] -= 0.5 * H.aa.oooo[m, k, :, :] * t_amp # (nk)

        # x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
        #              = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
        e, f, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3a[:, :, c, i, j, k] += 0.5 * H.aa.vvvv[:, :, e, f] * t_amp # (1)
        x3a[:, :, e, i, j, k] -= 0.5 * H.aa.vvvv[:, :, c, f] * t_amp # (ec)
        x3a[:, :, f, i, j, k] -= 0.5 * H.aa.vvvv[:, :, e, c] * t_amp # (fc)

        # x3a(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
        #              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3a[:, b, c, :, j, k] += H.aa.voov[:, m, :, e] * t_amp # (1)
        x3a[:, b, c, :, m, k] -= H.aa.voov[:, j, :, e] * t_amp # (mj)
        x3a[:, b, c, :, j, m] -= H.aa.voov[:, k, :, e] * t_amp # (mk)
        x3a[:, e, c, :, j, k] -= H.aa.voov[:, m, :, b] * t_amp # (eb)
        x3a[:, e, c, :, m, k] += H.aa.voov[:, j, :, b] * t_amp # (eb)(mj)
        x3a[:, e, c, :, j, m] += H.aa.voov[:, k, :, b] * t_amp # (eb)(mk)
        x3a[:, b, e, :, j, k] -= H.aa.voov[:, m, :, c] * t_amp # (ec)
        x3a[:, b, e, :, m, k] += H.aa.voov[:, j, :, c] * t_amp # (ec)(mj)
        x3a[:, b, e, :, j, m] += H.aa.voov[:, k, :, c] * t_amp # (ec)(mk)

        # x3b[abcijk) < - A(ij)A(ab)[A(m/ij)A(e/ab) h2b(mcek) * t3a(abeijm)]
        a, b, e, i, j, m = [x - 1 for x in T3_excitations["aaa"][idet]]
        x3b[a,b,:,i,j,:] = x3b[a,b,:,i,j,:] + H.ab.ovvo[m,:,e,:] * t_amp # (1)
        x3b[a,b,:,m,j,:] = x3b[a,b,:,m,j,:] - H.ab.ovvo[i,:,e,:] * t_amp # (im)
        x3b[a,b,:,i,m,:] = x3b[a,b,:,i,m,:] - H.ab.ovvo[j,:,e,:] * t_amp # (jm)
        x3b[e,b,:,i,j,:] = x3b[e,b,:,i,j,:] - H.ab.ovvo[m,:,a,:] * t_amp # (ae)
        x3b[e,b,:,m,j,:] = x3b[e,b,:,m,j,:] + H.ab.ovvo[i,:,a,:] * t_amp # (im)(ae)
        x3b[e,b,:,i,m,:] = x3b[e,b,:,i,m,:] + H.ab.ovvo[j,:,a,:] * t_amp # (jm)(ae)
        x3b[a,e,:,i,j,:] = x3b[a,e,:,i,j,:] - H.ab.ovvo[m,:,b,:] * t_amp # (be)
        x3b[a,e,:,m,j,:] = x3b[a,e,:,m,j,:] + H.ab.ovvo[i,:,b,:] * t_amp # (im)(be)
        x3b[a,e,:,i,m,:] = x3b[a,e,:,i,m,:] + H.ab.ovvo[j,:,b,:] * t_amp # (jm)(be)

        # I2A(amij) <- A(ij) [A(a/ef)A(n/ij) h2a(mnef) * t3a(aefijn)]
        a, e, f, i, j, n = [x - 1 for x in T3_excitations["aaa"][idet]]
        I2A_vooo[a, :, i, j] += H.aa.oovv[:, n, e, f] * t_amp # (1)
        I2A_vooo[e, :, i, j] -= H.aa.oovv[:, n, a, f] * t_amp # (ae)
        I2A_vooo[f, :, i, j] -= H.aa.oovv[:, n, e, a] * t_amp # (af)
        I2A_vooo[a, :, n, j] -= H.aa.oovv[:, i, e, f] * t_amp # (in)
        I2A_vooo[e, :, n, j] += H.aa.oovv[:, i, a, f] * t_amp # (in)(ae)
        I2A_vooo[f, :, n, j] += H.aa.oovv[:, i, e, a] * t_amp # (in)(af)
        I2A_vooo[a, :, i, n] -= H.aa.oovv[:, j, e, f] * t_amp # (jn)
        I2A_vooo[e, :, i, n] += H.aa.oovv[:, j, a, f] * t_amp # (jn)(ae)
        I2A_vooo[f, :, i, n] += H.aa.oovv[:, j, e, a] * t_amp # (jn)(af)

        # I2A(abie) < - A(ab)[A(i/mn)A(f/ab) - h2a(mnef) * t3a(abfimn)]
        a, b, f, i, m, n = [x - 1 for x in T3_excitations["aaa"][idet]]
        I2A_vvov[a,b,i,:] = I2A_vvov[a,b,i,:] - H.aa.oovv[m,n,:,f] * t_amp # (1)
        I2A_vvov[a,b,m,:] = I2A_vvov[a,b,m,:] + H.aa.oovv[i,n,:,f] * t_amp # (im)
        I2A_vvov[a,b,n,:] = I2A_vvov[a,b,n,:] + H.aa.oovv[m,i,:,f] * t_amp # (in)
        I2A_vvov[f,b,i,:] = I2A_vvov[f,b,i,:] + H.aa.oovv[m,n,:,a] * t_amp # (af)
        I2A_vvov[f,b,m,:] = I2A_vvov[f,b,m,:] - H.aa.oovv[i,n,:,a] * t_amp # (im)(af)
        I2A_vvov[f,b,n,:] = I2A_vvov[f,b,n,:] - H.aa.oovv[m,i,:,a] * t_amp # (in)(af)
        I2A_vvov[a,f,i,:] = I2A_vvov[a,f,i,:] + H.aa.oovv[m,n,:,b] * t_amp # (bf)
        I2A_vvov[a,f,m,:] = I2A_vvov[a,f,m,:] - H.aa.oovv[i,n,:,b] * t_amp # (im)(bf)
        I2A_vvov[a,f,n,:] = I2A_vvov[a,f,n,:] - H.aa.oovv[m,i,:,b] * t_amp # (in)(bf)

    # Loop over aab determinants
    for idet in range(len(T3_amplitudes["aab"])):

        # Get the particular aab T3 amplitude
        t_amp = T3_amplitudes["aab"][idet]

        # x3a(abcijk) <- A(c/ab)A(ab)A(ij)A(k/ij)[ h2b(cmke) * t3b(abeijm) ]
        #             = A(abc)A(ijk)[ h2b(cmke) * t3b(abeijm) ]
        a, b, e, i, j, m = [x - 1 for x in T3_excitations["aab"][idet]]
        x3a[a, b, :, i, j, :] += H.ab.voov[:, m, :, e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(mj) -h1a(mi) * t3b(abcmjk)]
        a, b, c, m, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,c,:,j,k] = x3b[a,b,c,:,j,k] - H.a.oo[m,:] * t_amp # (1)
        x3b[a,b,c,:,m,k] = x3b[a,b,c,:,m,k] + H.a.oo[j,:] * t_amp # (mj)

        # x3b[abcijk) <- A(ij)A(ab) [-h1b(mk) * t3b(abcijm)]
        a, b, c, i, j, m = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,c,i,j,:] = x3b[a,b,c,i,j,:] - H.b.oo[m,:] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(be) h1a(ae) * t3b(ebcijk)]
        e, b, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[:,b,c,i,j,k] = x3b[:,b,c,i,j,k] + H.a.vv[:,e] * t_amp # (1)
        x3b[:,e,c,i,j,k] = x3b[:,e,c,i,j,k] - H.a.vv[:,b] * t_amp # (eb)

        # x3b[abcijk) <- A(ij)A(ab) [h1b(ce) * t3b(abeijk)]
        a, b, e, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,:,i,j,k] = x3b[a,b,:,i,j,k] + H.b.vv[:,e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [1/2 h2a(mnij) * t3b(abcmnk)]
        a, b, c, m, n, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,c,:,:,k] = x3b[a,b,c,:,:,k] + 0.5 * H.aa.oooo[m,n,:,:] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(im) h2b(mnjk) * t3b(abcimn)]
        a, b, c, i, m, n = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,c,i,:,:] = x3b[a,b,c,i,:,:] + H.ab.oooo[m,n,:,:] * t_amp # (1)
        x3b[a,b,c,m,:,:] = x3b[a,b,c,m,:,:] - H.ab.oooo[i,n,:,:] * t_amp # (im)

        # x3b[abcijk) <- A(ij)A(ab) [1/2 h2a(abef) * t3b(efcijk)]
        e, f, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[:,:,c,i,j,k] = x3b[:,:,c,i,j,k] + 0.5 * H.aa.vvvv[:,:,e,f] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(ae) h2b(bcef) * t3b(aefijk)]
        a, e, f, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,:,:,i,j,k] = x3b[a,:,:,i,j,k] + H.ab.vvvv[:,:,e,f] * t_amp # (1)
        x3b[e,:,:,i,j,k] = x3b[e,:,:,i,j,k] - H.ab.vvvv[:,:,a,f] * t_amp # (ae)

        # x3b[abcijk) <- A(ij)A(ab) [A(be)A(jm) h2a(amie) * t3b(ebcmjk)]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[:,b,c,:,j,k] = x3b[:,b,c,:,j,k] + H.aa.voov[:,m,:,e] * t_amp # (1)
        x3b[:,e,c,:,j,k] = x3b[:,e,c,:,j,k] - H.aa.voov[:,m,:,b] * t_amp # (be)
        x3b[:,b,c,:,m,k] = x3b[:,b,c,:,m,k] - H.aa.voov[:,j,:,e] * t_amp # (jm)
        x3b[:,e,c,:,m,k] = x3b[:,e,c,:,m,k] + H.aa.voov[:,j,:,b] * t_amp # (be)(jm)

        # x3b[abcijk) <- A(ij)A(ab) [h2c(cmke) * t3b(abeijm)]
        a, b, e, i, j, m = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,:,i,j,:] = x3b[a,b,:,i,j,:] + H.bb.voov[:,m,:,e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(be) -h2b(amek) * t3b(ebcijm)]
        e, b, c, i, j, m = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[:,b,c,i,j,:] = x3b[:,b,c,i,j,:] - H.ab.vovo[:,m,e,:] * t_amp # (1)
        x3b[:,e,c,i,j,:] = x3b[:,e,c,i,j,:] + H.ab.vovo[:,m,b,:] * t_amp # (be)

        # x3b[abcijk) <- A(ij)A(ab) [A(jm) -h2b(mcie) * t3b(abemjk)]
        a, b, e, m, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3b[a,b,:,:,j,k] = x3b[a,b,:,:,j,k] - H.ab.ovov[m,:,:,e] * t_amp # (1)
        x3b[a,b,:,:,m,k] = x3b[a,b,:,:,m,k] + H.ab.ovov[j,:,:,e] * t_amp # (jm)

        # x3c(abcijk) <- A(jk)A(bc) [A(im)A(ae) h2b(mbej) * t3b(aecimk)]
        a, e, c, i, m, k = [x - 1 for x in T3_excitations["aab"][idet]]
        x3c[a,:,c,i,:,k] = x3c[a,:,c,i,:,k] + H.ab.ovvo[m,:,e,:] * t_amp # (1)
        x3c[a,:,c,m,:,k] = x3c[a,:,c,m,:,k] - H.ab.ovvo[i,:,e,:] * t_amp # (im)
        x3c[e,:,c,i,:,k] = x3c[e,:,c,i,:,k] - H.ab.ovvo[m,:,a,:] * t_amp # (ae)
        x3c[e,:,c,m,:,k] = x3c[e,:,c,m,:,k] + H.ab.ovvo[i,:,a,:] * t_amp # (im)(ae)

        # I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
        a, e, f, i, j, n = [x - 1 for x in T3_excitations["aab"][idet]]
        I2A_vooo[a, :, i, j] += H.ab.oovv[:, n, e, f] * t_amp # (1)
        I2A_vooo[e, :, i, j] -= H.ab.oovv[:, n, a, f] * t_amp # (ae)

        # I2A(abie) <- A(ab) [A(im) -h2b(mnef) * t3b(abfimn)]
        a, b, f, i, m, n = [x - 1 for x in T3_excitations["aab"][idet]]
        I2A_vvov[a,b,i,:] = I2A_vvov[a,b,i,:] - H.ab.oovv[m,n,:,f] * t_amp # (1)
        I2A_vvov[a,b,m,:] = I2A_vvov[a,b,m,:] + H.ab.oovv[i,n,:,f] * t_amp # (im)

        # I2B(abej) <- [A(af) -h2a(mnef) * t3b(afbmnj)]
        a, f, b, m, n, j = [x - 1 for x in T3_excitations["aab"][idet]]
        I2B_vvvo[a, b, :, j] = I2B_vvvo[a, b, :, j] - H.aa.oovv[m, n, :, f] * t_amp # (1)
        I2B_vvvo[f, b, :, j] = I2B_vvvo[f, b, :, j] + H.aa.oovv[m, n, :, a] * t_amp # (af)

        # I2B(mbij) <- [A(in) h2a(mnef) * t3b(efbinj)]
        e, f, b, i, n, j = [x - 1 for x in T3_excitations["aab"][idet]]
        I2B_ovoo[:, b, i, j] = I2B_ovoo[:, b, i, j] + H.aa.oovv[:, n, e, f] * t_amp # (1)
        I2B_ovoo[:, b, n, j] = I2B_ovoo[:, b, n, j] - H.aa.oovv[:, i, e, f] * t_amp # (in)

        # I2B(abie) <- [A(af)A(in) -h2b(nmfe) * t3b(afbinm)]
        a, f, b, i, n, m = [x - 1 for x in T3_excitations["aab"][idet]]
        I2B_vvov[a, b, i, :] = I2B_vvov[a, b, i, :] - H.ab.oovv[n, m, f, :] * t_amp # (1)
        I2B_vvov[f, b, i, :] = I2B_vvov[f, b, i, :] + H.ab.oovv[n, m, a, :] * t_amp # (af)
        I2B_vvov[a, b, n, :] = I2B_vvov[a, b, n, :] + H.ab.oovv[i, m, f, :] * t_amp # (in)
        I2B_vvov[f, b, n, :] = I2B_vvov[f, b, n, :] - H.ab.oovv[i, m, a, :] * t_amp # (af)(in)

        # I2B_vooo(amij) <- [A(in)A(af) h2b(nmfe) * t3b(afeinj)]
        a, f, e, i, n, j = [x - 1 for x in T3_excitations["aab"][idet]]
        I2B_vooo[a, :, i, j] = I2B_vooo[a, :, i, j] + H.ab.oovv[n, :, f, e] * t_amp # (1)
        I2B_vooo[a, :, n, j] = I2B_vooo[a, :, n, j] - H.ab.oovv[i, :, f, e] * t_amp # (in)
        I2B_vooo[f, :, i, j] = I2B_vooo[f, :, i, j] - H.ab.oovv[n, :, a, e] * t_amp # (af)
        I2B_vooo[f, :, n, j] = I2B_vooo[f, :, n, j] + H.ab.oovv[i, :, a, e] * t_amp # (in)(af)

    # Loop over abb determinants
    for idet in range(len(T3_amplitudes["abb"])):
    
        # Get the particular abb T3 amplitude
        t_amp = T3_amplitudes["abb"][idet]

        #  x3b(abcijk) < - A(ij)A(ab)[A(ec)A(mk) h2b(amie) * t3c(becjmk)]
        b, e, c, j, m, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3b[:,b,c,:,j,k] = x3b[:,b,c,:,j,k] + H.ab.voov[:,m,:,e] * t_amp # (1)
        x3b[:,b,e,:,j,k] = x3b[:,b,e,:,j,k] - H.ab.voov[:,m,:,c] * t_amp # (ec)
        x3b[:,b,c,:,j,m] = x3b[:,b,c,:,j,m] - H.ab.voov[:,k,:,e] * t_amp # (mk)
        x3b[:,b,e,:,j,m] = x3b[:,b,e,:,j,m] + H.ab.voov[:,k,:,c] * t_amp # (ec)(mk)

        # x3c(abcijk) < - A(jk)A(bc)[-h1a(mi) * t3c(abcmjk)]
        a, b, c, m, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,b,c,:,j,k] = x3c[a,b,c,:,j,k] - H.a.oo[m,:] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(mk) -h1b(mj) * t3c(abcimk)]
        a, b, c, i, m, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,b,c,i,:,k] = x3c[a,b,c,i,:,k] - H.b.oo[m,:] * t_amp # (1)
        x3c[a,b,c,i,:,m] = x3c[a,b,c,i,:,m] + H.b.oo[k,:] * t_amp # (mk)

        # x3c(abcijk) < - A(jk)A(bc)[h1a(ae) * t3c(ebcijk)]
        e, b, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[:,b,c,i,j,k] = x3c[:,b,c,i,j,k] + H.a.vv[:,e] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec) h1b(be) * t3c(aecijk)]
        a, e, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,:,c,i,j,k] = x3c[a,:,c,i,j,k] + H.b.vv[:,e] * t_amp # (1)
        x3c[a,:,e,i,j,k] = x3c[a,:,e,i,j,k] - H.b.vv[:,c] * t_amp # (ec)

        # x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(mnjk) * t3c(abcimn)]
        a, b, c, i, m, n = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,b,c,i,:,:] = x3c[a,b,c,i,:,:] + 0.5 * H.bb.oooo[m,n,:,:] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(kn) h2b(mnij) * t3c(abcmnk)]
        a, b, c, m, n, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,b,c,:,:,k] = x3c[a,b,c,:,:,k] + H.ab.oooo[m,n,:,:] * t_amp # (1)
        x3c[a,b,c,:,:,n] = x3c[a,b,c,:,:,n] - H.ab.oooo[m,k,:,:] * t_amp # (kn)

        # x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(bcef) * t3c(aefijk)]
        a, e, f, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,:,:,i,j,k] = x3c[a,:,:,i,j,k] + 0.5 * H.bb.vvvv[:,:,e,f] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(fc) h2b(abef) * t3c(efcijk)]
        e, f, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[:,:,c,i,j,k] = x3c[:,:,c,i,j,k] + H.ab.vvvv[:,:,e,f] * t_amp # (1)
        x3c[:,:,f,i,j,k] = x3c[:,:,f,i,j,k] - H.ab.vvvv[:,:,e,c] * t_amp # (fc)

        # x3c(abcijk) <- A(jk)A(bc) [h2a(amie) * t3c(ebcmjk)]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[:,b,c,:,j,k] = x3c[:,b,c,:,j,k] + H.aa.voov[:,m,:,e] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec)(mk) h2c(bmje) * t3c(aecimk)]
        a, e, c, i, m, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,:,c,i,:,k] = x3c[a,:,c,i,:,k] + H.bb.voov[:,m,:,e] * t_amp # (1)
        x3c[a,:,e,i,:,k] = x3c[a,:,e,i,:,k] - H.bb.voov[:,m,:,c] * t_amp # (ec)
        x3c[a,:,c,i,:,m] = x3c[a,:,c,i,:,m] - H.bb.voov[:,k,:,e] * t_amp # (mk)
        x3c[a,:,e,i,:,m] = x3c[a,:,e,i,:,m] + H.bb.voov[:,k,:,c] * t_amp # (ec)(mk)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec) -h2b(mbie) * t3c(aecmjk)]
        a, e, c, m, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[a,:,c,:,j,k] = x3c[a,:,c,:,j,k] - H.ab.ovov[m,:,:,e] * t_amp # (1)
        x3c[a,:,e,:,j,k] = x3c[a,:,e,:,j,k] + H.ab.ovov[m,:,:,c] * t_amp # (ec)

        # x3c(abcijk) <- A(jk)A(bc) [A(km) -h2b(amej) * t3c(ebcimk)]
        e, b, c, i, m, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3c[:,b,c,i,:,k] = x3c[:,b,c,i,:,k] - H.ab.vovo[:,m,e,:] * t_amp # (1)
        x3c[:,b,c,i,:,m] = x3c[:,b,c,i,:,m] + H.ab.vovo[:,k,e,:] * t_amp # (km)
    
        # x3d(abcijk) < - A(ijk)A(abc)[h2b(maei) * t3c(ebcmjk)]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        x3d[:,b,c,:,j,k] = x3d[:,b,c,:,j,k] + H.ab.ovvo[m,:,e,:] * t_amp # (1)

        # I2B(abej) <- [A(fb)A(nj) -h2b(mnef) * t3c(afbmnj)]
        a, f, b, m, n, j = [x - 1 for x in T3_excitations["abb"][idet]]
        I2B_vvvo[a, b, :, j] = I2B_vvvo[a, b, :, j] - H.ab.oovv[m, n, :, f] * t_amp # (1)
        I2B_vvvo[a, f, :, j] = I2B_vvvo[a, f, :, j] + H.ab.oovv[m, n, :, b] * t_amp # (fb)
        I2B_vvvo[a, b, :, n] = I2B_vvvo[a, b, :, n] + H.ab.oovv[m, j, :, f] * t_amp # (nj)
        I2B_vvvo[a, f, :, n] = I2B_vvvo[a, f, :, n] - H.ab.oovv[m, j, :, b] * t_amp # (fb)(nj)

        # I2B(mbij) <- [A(jn)A(fb) h2b(mnef) * t3c(efbinj)]
        I2B_ovoo[:, b, i, j] = I2B_ovoo[:, b, i, j] + H.ab.oovv[:, n, e, f] * t_amp # (1)
        I2B_ovoo[:, b, i, n] = I2B_ovoo[:, b, i, n] - H.ab.oovv[:, j, e, f] * t_amp # (jn)
        I2B_ovoo[:, f, i, j] = I2B_ovoo[:, f, i, j] - H.ab.oovv[:, n, e, b] * t_amp # (fb)
        I2B_ovoo[:, f, i, n] = I2B_ovoo[:, f, i, n] + H.ab.oovv[:, j, e, b] * t_amp # (jn)(fb)

        # I2B(abie) <- [A(fb) -h2c(nmfe) * t3c(afbinm)]
        a, f, b, i, n, m = [x - 1 for x in T3_excitations["abb"][idet]]
        I2B_vvov[a, b, i, :] = I2B_vvov[a, b, i, :] - H.bb.oovv[n, m, f, :] * t_amp # (1)
        I2B_vvov[a, f, i, :] = I2B_vvov[a, f, i, :] + H.bb.oovv[n, m, b, :] * t_amp # (fb)

        # I2B(amij) <- [A(jn) h2c(nmfe) * t3c(afeinj)]
        a, f, e, i, n, j = [x - 1 for x in T3_excitations["abb"][idet]]
        I2B_vooo[a, :, i, j] = I2B_vooo[a, :, i, j] + H.bb.oovv[n, :, f, e] * t_amp # (1)
        I2B_vooo[a, :, i, n] = I2B_vooo[a, :, i, n] - H.bb.oovv[j, :, f, e] * t_amp # (jn)

        # I2C(abie) <- A(ab) [A(im) -h2b(nmfe) * t3c(fbanmi)]
        f, b, a, n, m, i  = [x - 1 for x in T3_excitations["abb"][idet]]
        I2C_vvov[a, b, i, :] = I2C_vvov[a, b, i, :] - H.ab.oovv[n, m, f, :] * t_amp # (1)
        I2C_vvov[a, b, m, :] = I2C_vvov[a, b, m, :] + H.ab.oovv[n, i, f, :] * t_amp # (im)

        # I2C(amij) <- A(ij) [A(ae) h2b(nmfe) * t3c(feanji)]
        f, e, a, n, j, i  = [x - 1 for x in T3_excitations["abb"][idet]]
        I2C_vooo[a, :, i, j] = I2C_vooo[a, :, i, j] + H.ab.oovv[n, :, f, e] * t_amp # (1)
        I2C_vooo[e, :, i, j] = I2C_vooo[e, :, i, j] - H.ab.oovv[n, :, f, a] * t_amp # (ae)

    # Loop over bbb determinants
    for idet in range(len(T3_amplitudes["bbb"])):
    
        # Get the particular bbb T3 amplitude
        t_amp = T3_amplitudes["bbb"][idet]

        # x3c(abcijk) <- A(jk)A(bc) [A(m/jk)A(e/bc) h2b(amie) * t3d(ebcmjk)]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3c[:,b,c,:,j,k] = x3c[:,b,c,:,j,k] + H.ab.voov[:,m,:,e] * t_amp # (1)
        x3c[:,b,c,:,m,k] = x3c[:,b,c,:,m,k] - H.ab.voov[:,j,:,e] * t_amp # (jm)
        x3c[:,b,c,:,j,m] = x3c[:,b,c,:,j,m] - H.ab.voov[:,k,:,e] * t_amp # (km)
        x3c[:,e,c,:,j,k] = x3c[:,e,c,:,j,k] - H.ab.voov[:,m,:,b] * t_amp # (eb)
        x3c[:,e,c,:,m,k] = x3c[:,e,c,:,m,k] + H.ab.voov[:,j,:,b] * t_amp # (jm)(eb)
        x3c[:,e,c,:,j,m] = x3c[:,e,c,:,j,m] + H.ab.voov[:,k,:,b] * t_amp # (km)(eb)
        x3c[:,b,e,:,j,k] = x3c[:,b,e,:,j,k] - H.ab.voov[:,m,:,c] * t_amp # (ec)
        x3c[:,b,e,:,m,k] = x3c[:,b,e,:,m,k] + H.ab.voov[:,j,:,c] * t_amp # (jm)(ec)
        x3c[:,b,e,:,j,m] = x3c[:,b,e,:,j,m] + H.ab.voov[:,k,:,c] * t_amp # (km)(ec)

        # x3d(abcijk) <- -A(abc)A(i/jk)A(jk)A(m/jk) h1b(mi) * t3d(abcmjk)
        #               = -A(abc)A(ijk)[ A(m/jk) h1b(mi) * t3d(abcmjk) ]
        a, b, c, m, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3d[a,b,c,:,j,k] = x3d[a,b,c,:,j,k] - H.b.oo[m,:] * t_amp # (1)
        x3d[a,b,c,:,m,k] = x3d[a,b,c,:,m,k] + H.b.oo[j,:] * t_amp # (mj)
        x3d[a,b,c,:,j,m] = x3d[a,b,c,:,j,m] + H.b.oo[k,:] * t_amp # (mk)

        # x3d(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1b(ae) * t3d(ebcijk)
        #              = A(abc)A(ijk)[ A(e/bc) h1b(ae) * t3d(ebcijk) ]
        e, b, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3d[:,b,c,i,j,k] = x3d[:,b,c,i,j,k] + H.b.vv[:, e] * t_amp # (1)
        x3d[:,e,c,i,j,k] = x3d[:,e,c,i,j,k] - H.b.vv[:, b] * t_amp # (be)
        x3d[:,b,e,i,j,k] = x3d[:,b,e,i,j,k] - H.b.vv[:, c] * t_amp # (ce)

        # x3d(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2c(mnij) * t3d(abcmnk) ]
        #             = A(abc)A(ijk)[ 1/2 A(k/mn) h2c(mnij) * t3d(abcmnk) ]
        a, b, c, m, n, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3d[a,b,c,:,:,k] = x3d[a,b,c,:,:,k] + 0.5 * H.bb.oooo[m,n,:,:] * t_amp # (1)
        x3d[a,b,c,:,:,m] = x3d[a,b,c,:,:,m] - 0.5 * H.bb.oooo[k,n,:,:] * t_amp # (mk)
        x3d[a,b,c,:,:,n] = x3d[a,b,c,:,:,n] - 0.5 * H.bb.oooo[m,k,:,:] * t_amp # (nk)

        # x3d(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2c(abef) * t3d(efcijk) ]
        #              = A(abc)A(ijk)[ 1/2 A(c/ef) h2c(abef) * t3d(efcijk) ]
        e, f, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3d[:,:,c,i,j,k] = x3d[:,:,c,i,j,k] + 0.5 * H.bb.vvvv[:,:,e,f] * t_amp # (1)
        x3d[:,:,e,i,j,k] = x3d[:,:,e,i,j,k] - 0.5 * H.bb.vvvv[:,:,c,f] * t_amp # (ec)
        x3d[:,:,f,i,j,k] = x3d[:,:,f,i,j,k] - 0.5 * H.bb.vvvv[:,:,e,c] * t_amp # (fc)

        # x3d(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
        #              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
        e, b, c, m, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        x3d[:,b,c,:,j,k] = x3d[:,b,c,:,j,k] + H.bb.voov[:,m,:,e] * t_amp # (1)
        x3d[:,b,c,:,m,k] = x3d[:,b,c,:,m,k] - H.bb.voov[:,j,:,e] * t_amp # (mj)
        x3d[:,b,c,:,j,m] = x3d[:,b,c,:,j,m] - H.bb.voov[:,k,:,e] * t_amp # (mk)
        x3d[:,e,c,:,j,k] = x3d[:,e,c,:,j,k] - H.bb.voov[:,m,:,b] * t_amp # (eb)
        x3d[:,e,c,:,m,k] = x3d[:,e,c,:,m,k] + H.bb.voov[:,j,:,b] * t_amp # (eb)(mj)
        x3d[:,e,c,:,j,m] = x3d[:,e,c,:,j,m] + H.bb.voov[:,k,:,b] * t_amp # (eb)(mk)
        x3d[:,b,e,:,j,k] = x3d[:,b,e,:,j,k] - H.bb.voov[:,m,:,c] * t_amp # (ec)
        x3d[:,b,e,:,m,k] = x3d[:,b,e,:,m,k] + H.bb.voov[:,j,:,c] * t_amp # (ec)(mj)
        x3d[:,b,e,:,j,m] = x3d[:,b,e,:,j,m] + H.bb.voov[:,k,:,c] * t_amp # (ec)(mk)

        # I2C(amij) <- A(ij) [A(a/ef)A(n/ij) h2c(mnef) * t3d(aefijn)]
        a, e, f, i, j, n = [x - 1 for x in T3_excitations["bbb"][idet]]
        I2C_vooo[a, :, i, j] += H.bb.oovv[:, n, e, f] * t_amp # (1)
        I2C_vooo[e, :, i, j] -= H.bb.oovv[:, n, a, f] * t_amp # (ae)
        I2C_vooo[f, :, i, j] -= H.bb.oovv[:, n, e, a] * t_amp # (af)
        I2C_vooo[a, :, n, j] -= H.bb.oovv[:, i, e, f] * t_amp # (in)
        I2C_vooo[e, :, n, j] += H.bb.oovv[:, i, a, f] * t_amp # (in)(ae)
        I2C_vooo[f, :, n, j] += H.bb.oovv[:, i, e, a] * t_amp # (in)(af)
        I2C_vooo[a, :, i, n] -= H.bb.oovv[:, j, e, f] * t_amp # (jn)
        I2C_vooo[e, :, i, n] += H.bb.oovv[:, j, a, f] * t_amp # (jn)(ae)
        I2C_vooo[f, :, i, n] += H.bb.oovv[:, j, e, a] * t_amp # (jn)(af)

        # I2C(abie) < - A(ab)[A(i/mn)A(f/ab) - h2c(mnef) * t3d(abfimn)]
        a, b, f, i, m, n = [x - 1 for x in T3_excitations["bbb"][idet]]
        I2C_vvov[a,b,i,:] = I2C_vvov[a,b,i,:] - H.bb.oovv[m,n,:,f] * t_amp # (1)
        I2C_vvov[a,b,m,:] = I2C_vvov[a,b,m,:] + H.bb.oovv[i,n,:,f] * t_amp # (im)
        I2C_vvov[a,b,n,:] = I2C_vvov[a,b,n,:] + H.bb.oovv[m,i,:,f] * t_amp # (in)
        I2C_vvov[f,b,i,:] = I2C_vvov[f,b,i,:] + H.bb.oovv[m,n,:,a] * t_amp # (af)
        I2C_vvov[f,b,m,:] = I2C_vvov[f,b,m,:] - H.bb.oovv[i,n,:,a] * t_amp # (im)(af)
        I2C_vvov[f,b,n,:] = I2C_vvov[f,b,n,:] - H.bb.oovv[m,i,:,a] * t_amp # (in)(af)
        I2C_vvov[a,f,i,:] = I2C_vvov[a,f,i,:] + H.bb.oovv[m,n,:,b] * t_amp # (bf)
        I2C_vvov[a,f,m,:] = I2C_vvov[a,f,m,:] - H.bb.oovv[i,n,:,b] * t_amp # (im)(bf)
        I2C_vvov[a,f,n,:] = I2C_vvov[a,f,n,:] - H.bb.oovv[m,i,:,b] * t_amp # (in)(bf)

    # Update loop
    resid_aaa = np.zeros(len(T3_amplitudes["aaa"]))
    for idet in range(len(T3_amplitudes["aaa"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]

        res_mm23 = 0.0
        for m in range(noa):
            # -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
            res_mm23 = res_mm23 - (I2A_vooo[a, m, i, j] - I2A_vooo[a, m, j, i]) * T.aa[b, c, m, k]
            res_mm23 = res_mm23 + (I2A_vooo[b, m, i, j] - I2A_vooo[b, m, j, i]) * T.aa[a, c, m, k]
            res_mm23 = res_mm23 + (I2A_vooo[c, m, i, j] - I2A_vooo[c, m, j, i]) * T.aa[b, a, m, k]
            res_mm23 = res_mm23 + (I2A_vooo[a, m, k, j] - I2A_vooo[a, m, j, k]) * T.aa[b, c, m, i]
            res_mm23 = res_mm23 - (I2A_vooo[b, m, k, j] - I2A_vooo[b, m, j, k]) * T.aa[a, c, m, i]
            res_mm23 = res_mm23 - (I2A_vooo[c, m, k, j] - I2A_vooo[c, m, j, k]) * T.aa[b, a, m, i]
            res_mm23 = res_mm23 + (I2A_vooo[a, m, i, k] - I2A_vooo[a, m, k, i]) * T.aa[b, c, m, j]
            res_mm23 = res_mm23 - (I2A_vooo[b, m, i, k] - I2A_vooo[b, m, k, i]) * T.aa[a, c, m, j]
            res_mm23 = res_mm23 - (I2A_vooo[c, m, i, k] - I2A_vooo[c, m, k, i]) * T.aa[b, a, m, j]
        for e in range(nua):
            # A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
            res_mm23 = res_mm23 + (I2A_vvov[a, b, i, e] - I2A_vvov[b, a, i, e]) * T.aa[e, c, j, k]
            res_mm23 = res_mm23 - (I2A_vvov[c, b, i, e] - I2A_vvov[b, c, i, e]) * T.aa[e, a, j, k]
            res_mm23 = res_mm23 - (I2A_vvov[a, c, i, e] - I2A_vvov[c, a, i, e]) * T.aa[e, b, j, k]
            res_mm23 = res_mm23 - (I2A_vvov[a, b, j, e] - I2A_vvov[b, a, j, e]) * T.aa[e, c, i, k]
            res_mm23 = res_mm23 + (I2A_vvov[c, b, j, e] - I2A_vvov[b, c, j, e]) * T.aa[e, a, i, k]
            res_mm23 = res_mm23 + (I2A_vvov[a, c, j, e] - I2A_vvov[c, a, j, e]) * T.aa[e, b, i, k]
            res_mm23 = res_mm23 - (I2A_vvov[a, b, k, e] - I2A_vvov[b, a, k, e]) * T.aa[e, c, j, i]
            res_mm23 = res_mm23 + (I2A_vvov[c, b, k, e] - I2A_vvov[b, c, k, e]) * T.aa[e, a, j, i]
            res_mm23 = res_mm23 + (I2A_vvov[a, c, k, e] - I2A_vvov[c, a, k, e]) * T.aa[e, b, j, i]

        denom = (
                    H0.a.oo[i, i] + H0.a.oo[j, j] + H0.a.oo[k, k]
                   -H0.a.vv[a, a] - H0.a.vv[b, b] - H0.a.vv[c, c]
        )
        val = (
                 x3a[a,b,c,i,j,k] - x3a[a,c,b,i,j,k] + x3a[b,c,a,i,j,k] - x3a[b,a,c,i,j,k] + x3a[c,a,b,i,j,k] - x3a[c,b,a,i,j,k]
                -x3a[a,b,c,i,k,j] + x3a[a,c,b,i,k,j] - x3a[b,c,a,i,k,j] + x3a[b,a,c,i,k,j] - x3a[c,a,b,i,k,j] + x3a[c,b,a,i,k,j]
                +x3a[a,b,c,j,k,i] - x3a[a,c,b,j,k,i] + x3a[b,c,a,j,k,i] - x3a[b,a,c,j,k,i] + x3a[c,a,b,j,k,i] - x3a[c,b,a,j,k,i]
                -x3a[a,b,c,j,i,k] + x3a[a,c,b,j,i,k] - x3a[b,c,a,j,i,k] + x3a[b,a,c,j,i,k] - x3a[c,a,b,j,i,k] + x3a[c,b,a,j,i,k]
                +x3a[a,b,c,k,i,j] - x3a[a,c,b,k,i,j] + x3a[b,c,a,k,i,j] - x3a[b,a,c,k,i,j] + x3a[c,a,b,k,i,j] - x3a[c,b,a,k,i,j]
                -x3a[a,b,c,k,j,i] + x3a[a,c,b,k,j,i] - x3a[b,c,a,k,j,i] + x3a[b,a,c,k,j,i] - x3a[c,a,b,k,j,i] + x3a[c,b,a,k,j,i]
        )
        val = (val + res_mm23)/denom
        resid_aaa[idet] = val

    resid_aab = np.zeros(len(T3_amplitudes["aab"]))
    for idet in range(len(T3_amplitudes["aab"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]

        res_mm23 = 0.0
        for e in range(nua):
            # A(ab) I2B(bcek) * t2a(aeij)
            res_mm23 = res_mm23 + I2B_vvvo[b, c, e, k] * T.aa[a, e, i, j]
            res_mm23 = res_mm23 - I2B_vvvo[a, c, e, k] * T.aa[b, e, i, j]
            # A(ij) I2A(abie) * t2b(ecjk)
            res_mm23 = res_mm23 + (I2A_vvov[a, b, i, e] - I2A_vvov[b, a, i, e]) * T.ab[e, c, j, k]
            res_mm23 = res_mm23 - (I2A_vvov[a, b, j, e] - I2A_vvov[b, a, j, e]) * T.ab[e, c, i, k]
        for e in range(nub):
            # A(ij)A(ab) I2B(acie) * t2b(bejk)
            res_mm23 = res_mm23 + I2B_vvov[a, c, i, e] * T.ab[b, e, j, k]
            res_mm23 = res_mm23 - I2B_vvov[a, c, j, e] * T.ab[b, e, i, k]
            res_mm23 = res_mm23 - I2B_vvov[b, c, i, e] * T.ab[a, e, j, k]
            res_mm23 = res_mm23 + I2B_vvov[b, c, j, e] * T.ab[a, e, i, k]
        for m in range(noa):
            # -A(ij) h2b(mcjk) * t2a(abim) 
            res_mm23 = res_mm23 - I2B_ovoo[m, c, j, k] * T.aa[a, b, i, m]
            res_mm23 = res_mm23 + I2B_ovoo[m, c, i, k] * T.aa[a, b, j, m]
            # -A(ab) h2a(amij) * t2b(bcmk)
            res_mm23 = res_mm23 - (I2A_vooo[a, m, i, j] - I2A_vooo[a, m, j, i]) * T.ab[b, c, m, k]
            res_mm23 = res_mm23 + (I2A_vooo[b, m, i, j] - I2A_vooo[b, m, j, i]) * T.ab[a, c, m, k]
        for m in range(nob):
            # -A(ij)A(ab) h2b(amik) * t2b(bcjm)
            res_mm23 = res_mm23 - I2B_vooo[a, m, i, k] * T.ab[b, c, j, m]
            res_mm23 = res_mm23 + I2B_vooo[b, m, i, k] * T.ab[a, c, j, m]
            res_mm23 = res_mm23 + I2B_vooo[a, m, j, k] * T.ab[b, c, i, m]
            res_mm23 = res_mm23 - I2B_vooo[b, m, j, k] * T.ab[a, c, i, m]

        denom = (
                    H0.a.oo[i, i] + H0.a.oo[j, j] + H0.b.oo[k, k]
                   -H0.a.vv[a, a] - H0.a.vv[b, b] - H0.b.vv[c, c]
        )
        val = (
                 x3b[a,b,c,i,j,k] - x3b[b,a,c,i,j,k]
                -x3b[a,b,c,j,i,k] + x3b[b,a,c,j,i,k]
        )
        val = (val + res_mm23)/denom
        resid_aab[idet] = val

    resid_abb = np.zeros(len(T3_amplitudes["abb"]))
    for idet in range(len(T3_amplitudes["abb"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]

        res_mm23 = 0.0
        for e in range(nua):
            # A(jk)A(bc) h2B(abej) * t2b(ecik)
            res_mm23 = res_mm23 + I2B_vvvo[a, b, e, j] * T.ab[e, c, i, k]
            res_mm23 = res_mm23 - I2B_vvvo[a, b, e, k] * T.ab[e, c, i, j]
            res_mm23 = res_mm23 - I2B_vvvo[a, c, e, j] * T.ab[e, b, i, k]
            res_mm23 = res_mm23 + I2B_vvvo[a, c, e, k] * T.ab[e, b, i, j]
        for e in range(nub):
            # A(bc) h2B(abie) * t2c(ecjk)
            res_mm23 = res_mm23 + I2B_vvov[a, b, i, e] * T.bb[e, c, j, k]
            res_mm23 = res_mm23 - I2B_vvov[a, c, i, e] * T.bb[e, b, j, k]
            # A(jk) h2C(cbke) * t2b(aeij)
            res_mm23 = res_mm23 + (I2C_vvov[c, b, k, e] - I2C_vvov[b, c, k, e]) * T.ab[a, e, i, j]
            res_mm23 = res_mm23 - (I2C_vvov[c, b, j, e] - I2C_vvov[b, c, j, e])* T.ab[a, e, i, k]
        for m in range(noa):
            # -A(kj)A(bc) h2b(mbij) * t2b(acmk)
            res_mm23 = res_mm23 - I2B_ovoo[m, b, i, j] * T.ab[a, c, m, k]
            res_mm23 = res_mm23 + I2B_ovoo[m, c, i, j] * T.ab[a, b, m, k]
            res_mm23 = res_mm23 + I2B_ovoo[m, b, i, k] * T.ab[a, c, m, j]
            res_mm23 = res_mm23 - I2B_ovoo[m, c, i, k] * T.ab[a, b, m, j]
        for m in range(nob):
            # -A(jk) h2b(amij) * t2c(bcmk)
            res_mm23 = res_mm23 - I2B_vooo[a, m, i, j] * T.bb[b, c, m, k]
            res_mm23 = res_mm23 + I2B_vooo[a, m, i, k] * T.bb[b, c, m, j]
            # -A(bc) h2c(cmkj) * t2b(abim)
            res_mm23 = res_mm23 - (I2C_vooo[c, m, k, j] - I2C_vooo[c, m, j, k]) * T.ab[a, b, i, m]
            res_mm23 = res_mm23 + (I2C_vooo[b, m, k, j] - I2C_vooo[b, m, j, k]) * T.ab[a, c, i, m]

        denom = (
                    H0.a.oo[i, i] + H0.b.oo[j, j] + H0.b.oo[k, k]
                   -H0.a.vv[a, a] - H0.b.vv[b, b] - H0.b.vv[c, c]
        )
        val = (
                 x3c[a,b,c,i,j,k] - x3c[a,c,b,i,j,k]
                -x3c[a,b,c,i,k,j] + x3c[a,c,b,i,k,j]
        )
        val = (val + res_mm23)/denom
        resid_abb[idet] = val

    resid_bbb = np.zeros(len(T3_amplitudes["bbb"]))
    for idet in range(len(T3_amplitudes["bbb"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]

        res_mm23 = 0.0
        for m in range(nob):
            # -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
            res_mm23 = res_mm23 - (I2C_vooo[a, m, i, j] - I2C_vooo[a, m, j, i]) * T.bb[b, c, m, k]
            res_mm23 = res_mm23 + (I2C_vooo[b, m, i, j] - I2C_vooo[b, m, j, i]) * T.bb[a, c, m, k]
            res_mm23 = res_mm23 + (I2C_vooo[c, m, i, j] - I2C_vooo[c, m, j, i]) * T.bb[b, a, m, k]
            res_mm23 = res_mm23 + (I2C_vooo[a, m, k, j] - I2C_vooo[a, m, j, k]) * T.bb[b, c, m, i]
            res_mm23 = res_mm23 - (I2C_vooo[b, m, k, j] - I2C_vooo[b, m, j, k]) * T.bb[a, c, m, i]
            res_mm23 = res_mm23 - (I2C_vooo[c, m, k, j] - I2C_vooo[c, m, j, k]) * T.bb[b, a, m, i]
            res_mm23 = res_mm23 + (I2C_vooo[a, m, i, k] - I2C_vooo[a, m, k, i]) * T.bb[b, c, m, j]
            res_mm23 = res_mm23 - (I2C_vooo[b, m, i, k] - I2C_vooo[b, m, k, i]) * T.bb[a, c, m, j]
            res_mm23 = res_mm23 - (I2C_vooo[c, m, i, k] - I2C_vooo[c, m, k, i]) * T.bb[b, a, m, j]
        for e in range(nub):
            # A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)
            res_mm23 = res_mm23 + (I2C_vvov[a, b, i, e] - I2C_vvov[b, a, i, e]) * T.bb[e, c, j, k]
            res_mm23 = res_mm23 - (I2C_vvov[c, b, i, e] - I2C_vvov[b, c, i, e]) * T.bb[e, a, j, k]
            res_mm23 = res_mm23 - (I2C_vvov[a, c, i, e] - I2C_vvov[c, a, i, e]) * T.bb[e, b, j, k]
            res_mm23 = res_mm23 - (I2C_vvov[a, b, j, e] - I2C_vvov[b, a, j, e]) * T.bb[e, c, i, k]
            res_mm23 = res_mm23 + (I2C_vvov[c, b, j, e] - I2C_vvov[b, c, j, e]) * T.bb[e, a, i, k]
            res_mm23 = res_mm23 + (I2C_vvov[a, c, j, e] - I2C_vvov[c, a, j, e]) * T.bb[e, b, i, k]
            res_mm23 = res_mm23 - (I2C_vvov[a, b, k, e] - I2C_vvov[b, a, k, e]) * T.bb[e, c, j, i]
            res_mm23 = res_mm23 + (I2C_vvov[c, b, k, e] - I2C_vvov[b, c, k, e]) * T.bb[e, a, j, i]
            res_mm23 = res_mm23 + (I2C_vvov[a, c, k, e] - I2C_vvov[c, a, k, e]) * T.bb[e, b, j, i]

        denom = (
                    H0.b.oo[i, i] + H0.b.oo[j, j] + H0.b.oo[k, k]
                   -H0.b.vv[a, a] - H0.b.vv[b, b] - H0.b.vv[c, c]
        )
        val = (
                 x3d[a,b,c,i,j,k] - x3d[a,c,b,i,j,k] + x3d[b,c,a,i,j,k] - x3d[b,a,c,i,j,k] + x3d[c,a,b,i,j,k] - x3d[c,b,a,i,j,k]
                -x3d[a,b,c,i,k,j] + x3d[a,c,b,i,k,j] - x3d[b,c,a,i,k,j] + x3d[b,a,c,i,k,j] - x3d[c,a,b,i,k,j] + x3d[c,b,a,i,k,j]
                +x3d[a,b,c,j,k,i] - x3d[a,c,b,j,k,i] + x3d[b,c,a,j,k,i] - x3d[b,a,c,j,k,i] + x3d[c,a,b,j,k,i] - x3d[c,b,a,j,k,i]
                -x3d[a,b,c,j,i,k] + x3d[a,c,b,j,i,k] - x3d[b,c,a,j,i,k] + x3d[b,a,c,j,i,k] - x3d[c,a,b,j,i,k] + x3d[c,b,a,j,i,k]
                +x3d[a,b,c,k,i,j] - x3d[a,c,b,k,i,j] + x3d[b,c,a,k,i,j] - x3d[b,a,c,k,i,j] + x3d[c,a,b,k,i,j] - x3d[c,b,a,k,i,j]
                -x3d[a,b,c,k,j,i] + x3d[a,c,b,k,j,i] - x3d[b,c,a,k,j,i] + x3d[b,a,c,k,j,i] - x3d[c,a,b,k,j,i] + x3d[c,b,a,k,j,i]
        )
        val = (val + res_mm23)/denom
        resid_bbb[idet] = val

    return resid_aaa, resid_aab, resid_abb, resid_bbb


def contract_vt3_fly_fortran(H, H0, T, T3_excitations, T3_amplitudes):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # build adjusted intermediates
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)

    # save the input T vectors; these get modififed by Fortran calls even if output is not set
    t1a_amps = T.a.copy()
    t1b_amps = T.b.copy()
    t2a_amps = T.aa.copy()
    t2b_amps = T.ab.copy()
    t2c_amps = T.bb.copy()
    t3a_amps = T3_amplitudes["aaa"].copy()
    t3b_amps = T3_amplitudes["aab"].copy()
    t3c_amps = T3_amplitudes["abb"].copy()
    t3d_amps = T3_amplitudes["bbb"].copy()

    _, x1a = ccp_linear_loops.ccp_linear_loops.update_t1a(
        T.a, 
        np.zeros((nua, noa)),
        T3_excitations["aaa"], T3_excitations["aab"], T3_excitations["abb"],
        T3_amplitudes["aaa"], T3_amplitudes["aab"], T3_amplitudes["abb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv,
        0.0
    )
    T.a = t1a_amps.copy()

    _, x1b = ccp_linear_loops.ccp_linear_loops.update_t1b(
        T.b, 
        np.zeros((nub, nob)),
        T3_excitations["aab"], T3_excitations["abb"], T3_excitations["bbb"],
        T3_amplitudes["aab"], T3_amplitudes["abb"], T3_amplitudes["bbb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.b.oo, H0.b.vv,
        0.0
    )
    T.b = t1b_amps.copy()

    _, x2a = ccp_linear_loops.ccp_linear_loops.update_t2a(
        T.aa,
        np.zeros((nua,nua,noa,noa)),
        T3_excitations["aaa"], T3_excitations["aab"],
        T3_amplitudes["aaa"], T3_amplitudes["aab"],
        H.a.ov, H.b.ov,
        H0.aa.ooov + H.aa.ooov, H0.aa.vovv + H.aa.vovv,
        H0.ab.ooov + H.ab.ooov, H0.ab.vovv + H.ab.vovv,
        H0.a.oo, H0.a.vv,
        0.0
    )
    T.aa = t2a_amps.copy()

    _, x2b = ccp_linear_loops.ccp_linear_loops.update_t2b(
        T.ab,
        np.zeros((nua,nub,noa,nob)),
        T3_excitations["aab"], T3_excitations["abb"],
        T3_amplitudes["aab"], T3_amplitudes["abb"],
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.oovo + H0.ab.oovo, H.ab.vovv + H0.ab.vovv, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        0.0
    )
    T.ab = t2b_amps.copy()

    _, x2c = ccp_linear_loops.ccp_linear_loops.update_t2c(
        T.bb,
        np.zeros((nub,nub,nob,nob)),
        T3_excitations["abb"], T3_excitations["bbb"],
        T3_amplitudes["abb"], T3_amplitudes["bbb"],
        H.a.ov, H.b.ov,
        H0.ab.oovo + H.ab.oovo, H0.ab.ovvv + H.ab.ovvv,
        H0.bb.ooov + H.bb.ooov, H0.bb.vovv + H.bb.vovv,
        H0.b.oo, H0.b.vv,
        0.0
    )
    T.bb = t2c_amps.copy()

    _, x3a = ccp_linear_loops.ccp_linear_loops.update_t3a_p(
        T3_amplitudes["aaa"],
        T3_excitations["aaa"], T3_excitations["aab"],
        T.aa,
        T3_amplitudes["aab"],
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        0.0
    )
    T3_amplitudes["aaa"] = t3a_amps.copy()

    _, x3b = ccp_linear_loops.ccp_linear_loops.update_t3b_p(
        T3_amplitudes["aab"],
        T3_excitations["aaa"], T3_excitations["aab"], T3_excitations["abb"],
        T.aa, T.ab,
        T3_amplitudes["aaa"], T3_amplitudes["abb"],
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo, 
        H.ab.oooo, H.ab.voov,H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        0.0
    )
    T3_amplitudes["aab"] = t3b_amps.copy()

    _, x3c = ccp_linear_loops.ccp_linear_loops.update_t3c_p(
        T3_amplitudes["abb"],
        T3_excitations["aab"], T3_excitations["abb"], T3_excitations["bbb"],
        T.ab, T.bb,
        T3_amplitudes["aab"], T3_amplitudes["bbb"],
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        0.0
    )
    T3_amplitudes["abb"] = t3c_amps.copy()

    _, x3d = ccp_linear_loops.ccp_linear_loops.update_t3d_p(
        T3_amplitudes["bbb"],
        T3_excitations["abb"], T3_excitations["bbb"],
        T.bb,
        T3_amplitudes["abb"],
        H.b.oo, H.b.vv,
        H0.ab.oovv, H.ab.ovvo,
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.b.oo, H0.b.vv,
        0.0
    )
    T3_amplitudes["bbb"] = t3d_amps.copy()

    return x1a, x1b, x2a, x2b, x2c, x3a, x3b, x3c, x3d

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.Mole()

    methylene = """
                    C 0.0000000000 0.0000000000 -0.1160863568
                    H -1.8693479331 0.0000000000 0.6911102033
                    H 1.8693479331 0.0000000000  0.6911102033
     """

    fluorine = """
                    F 0.0000000000 0.0000000000 -2.66816
                    F 0.0000000000 0.0000000000  2.66816
    """

    mol.build(
        atom=fluorine,
        basis="ccpvdz",
        symmetry="C2V",
        spin=0, 
        charge=0,
        unit="Bohr",
        cart=False,
    )
    mf = scf.ROHF(mol).run()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(calculation_type="ccsdt")
    T, cc_energy, converged = cc_driver(calculation, system, H)
    hbar = get_ccsd_intermediates(T, H)

    T3_excitations, T3_amplitudes = get_T3_list(T)

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T3 contraction", end="")
    t1 = time.time()
    x1_a_exact, x1_b_exact, x2_aa_exact, x2_ab_exact, x2_bb_exact, x3_aaa_exact, x3_aab_exact, x3_abb_exact, x3_bbb_exact = contract_vt3_exact(H, hbar, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # # Get the on-the-fly contraction result
    # print("   On-the-fly H*T3 contraction", end="")
    # t1 = time.time()
    # x3_aaa, x3_aab, x3_abb, x3_bbb = contract_vt3_fly(hbar, H, T, T3_excitations, T3_amplitudes)
    # print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction (Fortran)", end="")
    t1 = time.time()
    x1_a, x1_b, x2_aa, x2_ab, x2_bb, x3_aaa, x3_aab, x3_abb, x3_bbb = contract_vt3_fly_fortran(hbar, H, T, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    nua, noa = T.a.shape
    nub, nob = T.b.shape

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            denom = (
                        H.a.oo[i, i] - H.a.vv[a, a]
            )
            error = x1_a[a, i] - x1_a_exact[a, i]/denom
            err_cum += abs(error)
            if abs(error) > 1.0e-012:
                flag = False
    if flag:
        print("T1A update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T1A update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            denom = (
                        H.b.oo[i, i] - H.b.vv[a, a]
            )
            error = x1_b[a, i] - x1_b_exact[a, i]/denom
            err_cum += abs(error)
            if abs(error) > 1.0e-012:
                flag = False
    if flag:
        print("T1B update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T1B update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    denom = (
                                H.a.oo[i, i] + H.a.oo[j, j] 
                              - H.a.vv[a, a] - H.a.vv[b, b]
                    )
                    error = x2_aa[a, b, i, j] - x2_aa_exact[a, b, i, j]/denom
                    err_cum += abs(error)
                    if abs(error) > 1.0e-012:
                        flag = False
    if flag:
        print("T2A update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T2A update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    denom = (
                                H.a.oo[i, i] + H.b.oo[j, j] 
                              - H.a.vv[a, a] - H.b.vv[b, b]
                    )
                    error = x2_ab[a, b, i, j] - x2_ab_exact[a, b, i, j]/denom
                    err_cum += abs(error)
                    if abs(error) > 1.0e-012:
                        flag = False
    if flag:
        print("T2B update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T2B update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    denom = (
                                H.b.oo[i, i] + H.b.oo[j, j] 
                              - H.b.vv[a, a] - H.b.vv[b, b]
                    )
                    error = x2_bb[a, b, i, j] - x2_bb_exact[a, b, i, j]/denom
                    err_cum += abs(error)
                    if abs(error) > 1.0e-012:
                        flag = False
    if flag:
        print("T2C update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T2C update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aaa"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        denom = (
                    H.a.oo[i, i] + H.a.oo[j, j] + H.a.oo[k, k]
                   -H.a.vv[a, a] - H.a.vv[b, b] - H.a.vv[c, c]
        )
        error = x3_aaa[idet] - x3_aaa_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-012:
            flag = False
    if flag:
        print("T3A update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3A update FAILED!", "Cumulative Error = ", err_cum)
    

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aab"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        denom = (
                    H.a.oo[i, i] + H.a.oo[j, j] + H.b.oo[k, k]
                   -H.a.vv[a, a] - H.a.vv[b, b] - H.b.vv[c, c]
        )
        error = x3_aab[idet] - x3_aab_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3B update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3B update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["abb"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
        denom = (
                    H.a.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
                   -H.a.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
        )
        error = x3_abb[idet] - x3_abb_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3C update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3C update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["bbb"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
        denom = (
                    H.b.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
                   -H.b.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
        )
        error = x3_bbb[idet] - x3_bbb_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3D update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3D update FAILED!", "Cumulative Error = ", err_cum)

