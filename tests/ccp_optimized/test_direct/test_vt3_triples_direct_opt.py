"""In this script, we test out the idea of using the 'amplitude-driven' approach
to constructing the sparse triples projection < ijkabc | (H(2) * T3)_C | 0 >, where
T3 is sparse and defined over a given list of triples."""
import time

import numpy as np
from itertools import permutations
import random

from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import ccp_quadratic_loops_direct_opt

#print(ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3a_p.__doc__)
#print(ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3b_p.__doc__)
#print(ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3c_p.__doc__)
#print(ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3d_p.__doc__)

def get_T3_list_fraction(T, fraction):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    n3a = int(nua*(nua - 1)*(nua - 2)/6 * noa*(noa - 1)*(noa - 2)/6); n3a_p = int(fraction[0] * n3a);
    n3b = int(nua*(nua - 1)/2 * nub * noa*(noa - 1)/2 * nob); n3b_p = int(fraction[1] * n3b);
    n3c = int(nua * nub*(nub - 1)/2 * noa * nob*(nob - 1)/2); n3c_p = int(fraction[2] * n3c);
    n3d = int(nub*(nub - 1)*(nub - 2)/6 * nob*(nob - 1)*(nob - 2)/6); n3d_p = int(fraction[3] * n3d);

    T3_excitations = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    T3_amplitudes = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}

    num_triples = 0

    rand_arr = np.asarray(random.sample(range(n3a), n3a_p))
    idx = np.argsort(rand_arr)
    rand_arr = rand_arr[idx]
    nct1 = 0
    nct2 = 0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(j + 1, noa):

                            if nct2 == n3a_p:
                                for ip, jp, kp in permutations((i, j, k)):
                                    for ap, bp, cp in permutations((a, b, c)):
                                        T.aaa[ap, bp, cp, ip, jp, kp] *= 0.0
                                continue

                            if nct1 == rand_arr[nct2]:
                                num_triples += 1
                                nct2 += 1
                                T3_excitations["aaa"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                                T3_amplitudes["aaa"].append(T.aaa[a, b, c, i, j, k])
                            else:
                                for ip, jp, kp in permutations((i, j, k)):
                                    for ap, bp, cp in permutations((a, b, c)):
                                        T.aaa[ap, bp, cp, ip, jp, kp] *= 0.0
                            nct1 += 1

    rand_arr = np.asarray(random.sample(range(n3b), n3b_p))
    idx = np.argsort(rand_arr)
    rand_arr = rand_arr[idx]
    nct1 = 0
    nct2 = 0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(nob):

                            if nct2 == n3b_p:
                                for ip, jp in permutations((i, j)):
                                    for ap, bp in permutations((a, b)):
                                        T.aab[ap, bp, c, ip, jp, k] *= 0.0
                                continue

                            if nct1 == rand_arr[nct2]:
                                num_triples += 1
                                nct2 += 1
                                T3_excitations["aab"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                                T3_amplitudes["aab"].append(T.aab[a, b, c, i, j, k])
                            else:
                                for ip, jp in permutations((i, j)):
                                    for ap, bp in permutations((a, b)):
                                        T.aab[ap, bp, c, ip, jp, k] *= 0.0
                            nct1 += 1

    rand_arr = np.asarray(random.sample(range(n3c), n3c_p))
    idx = np.argsort(rand_arr)
    rand_arr = rand_arr[idx]
    nct1 = 0
    nct2 = 0
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j + 1, nob):

                            if nct2 == n3c_p:
                                for jp, kp in permutations((j, k)):
                                    for bp, cp in permutations((b, c)):
                                        T.abb[a, bp, cp, i, jp, kp] *= 0.0
                                continue

                            if nct1 == rand_arr[nct2]:
                                num_triples += 1
                                nct2 += 1
                                T3_excitations["abb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                                T3_amplitudes["abb"].append(T.abb[a, b, c, i, j, k])
                            else:
                                for jp, kp in permutations((j, k)):
                                    for bp, cp in permutations((b, c)):
                                        T.abb[a, bp, cp, i, jp, kp] *= 0.0
                            nct1 += 1

    rand_arr = np.asarray(random.sample(range(n3d), n3d_p))
    idx = np.argsort(rand_arr)
    rand_arr = rand_arr[idx]
    nct1 = 0
    nct2 = 0
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        for k in range(j + 1, nob):

                            if nct2 == n3d_p:
                                for ip, jp, kp in permutations((i, j, k)):
                                    for ap, bp, cp in permutations((a, b, c)):
                                        T.bbb[ap, bp, cp, ip, jp, kp] *= 0.0
                                continue

                            if nct1 == rand_arr[nct2]:
                                num_triples += 1
                                nct2 += 1
                                T3_excitations["bbb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                                T3_amplitudes["bbb"].append(T.bbb[a, b, c, i, j, k])
                            else:
                                for ip, jp, kp in permutations((i, j, k)):
                                    for ap, bp, cp in permutations((a, b, c)):
                                        T.bbb[ap, bp, cp, ip, jp, kp] *= 0.0
                            nct1 += 1

    print("P space contains", num_triples, "triples.")

    for key in T3_excitations.keys():
        T3_excitations[key] = np.asarray(T3_excitations[key])
        T3_amplitudes[key] = np.asarray(T3_amplitudes[key])

    return T3_excitations, T3_amplitudes

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

    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)

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

    x3b -= np.transpose(x3b, (1, 0, 2, 3, 4, 5)) # (ab)
    x3b -= np.transpose(x3b, (0, 1, 2, 4, 3, 5)) # (ij)

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
    #x3c += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True)
    #x3c += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True)
    #x3c += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True)
    #x3c += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True)
    #x3c -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True)
    #x3c -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True)

    x3c -= np.transpose(x3c, (0, 2, 1, 3, 4, 5)) # (bc)
    x3c -= np.transpose(x3c, (0, 1, 2, 3, 5, 4)) # (jk)

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

    x3d -= np.transpose(x3d, (0, 1, 2, 3, 5, 4)) # (jk)
    x3d -= np.transpose(x3d, (0, 1, 2, 4, 3, 5)) + np.transpose(x3d, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3d -= np.transpose(x3d, (0, 2, 1, 3, 4, 5)) # (bc)
    x3d -= np.transpose(x3d, (2, 1, 0, 3, 4, 5)) + np.transpose(x3d, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return x3a, x3b, x3c, x3d

def contract_vt3_fly(H, H0, T, T3_excitations, T3_amplitudes):

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
    t3a_excits = T3_excitations["aaa"].copy()
    t3b_excits = T3_excitations["aab"].copy()
    t3c_excits = T3_excitations["abb"].copy()
    t3d_excits = T3_excitations["bbb"].copy()


    tic = time.time()
    resid_aaa, t3_aaa, t3_excits_aaa = ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3a_p(
        T3_amplitudes["aaa"], T3_excitations["aaa"], 
        T3_amplitudes["aab"], T3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        0.0,
    )
    print("t3a took", time.time() - tic)
    T3_amplitudes["aaa"] = t3a_amps.copy()
    T3_excitations["aaa"] = t3a_excits.copy()

    tic = time.time()
    resid_aab, t3_aab, t3_excits_aab = ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3b_p(
        T3_amplitudes["aaa"], T3_excitations["aaa"],
        T3_amplitudes["aab"], T3_excitations["aab"],
        T3_amplitudes["abb"], T3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo, 
        H.ab.oooo, H.ab.voov,H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        0.0
    )
    print("t3b took", time.time() - tic)
    T3_amplitudes["aab"] = t3b_amps.copy()
    T3_excitations["aab"] = t3b_excits.copy()

    tic = time.time()
    resid_abb, t3_abb, t3_excits_abb  = ccp_quadratic_loops_direct_opt.ccp_quadratic_loops_direct_opt.update_t3c_p(
        T3_amplitudes["aab"], T3_excitations["aab"],
        T3_amplitudes["abb"], T3_excitations["abb"],
        T3_amplitudes["bbb"], T3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        0.0
    )
    print("t3c took", time.time() - tic)
    T3_amplitudes["abb"] = t3c_amps.copy()
    T3_excitations["abb"] = t3c_excits.copy()

    #tic = time.time()
    #t3_bbb, resid_bbb = ccp_quadratic_loops_direct_h.ccp_quadratic_loops_direct_h.update_t3d_p(
    #    T3_amplitudes["bbb"],
    #    T3_excitations["abb"].T, T3_excitations["bbb"].T,
    #    T.bb,
    #    T3_amplitudes["abb"],
    #    H.b.oo, H.b.vv,
    #    H0.ab.oovv, H.ab.ovvo,
    #    H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
    #    H0.b.oo, H0.b.vv,
    #    0.0
    #)
    #print("t3d took", time.time() - tic)
    #T3_amplitudes["bbb"] = t3d_amps.copy()

    return t3_aaa, t3_excits_aaa, resid_aaa, t3_aab, t3_excits_aab, resid_aab, t3_abb, t3_excits_abb, resid_abb#, t3_bbb, resid_bbb

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
        basis="cc-pvdz",
        symmetry="D2H",
        spin=0, 
        charge=0,
        unit="Bohr",
        cart=True,
    )
    mf = scf.ROHF(mol).run()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(calculation_type="ccsdt")
    T, cc_energy, converged = cc_driver(calculation, system, H)
    hbar = get_ccsd_intermediates(T, H)

    #T3_excitations, T3_amplitudes = get_T3_list(T)
    T3_excitations, T3_amplitudes = get_T3_list_fraction(T, fraction=[1,1,1,1])

    T3_excitations["aaa"] = T3_excitations["aaa"].T
    T3_excitations["aab"] = T3_excitations["aab"].T
    T3_excitations["abb"] = T3_excitations["abb"].T
    T3_excitations["bbb"] = T3_excitations["bbb"].T

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T3 contraction")
    t1 = time.time()
    x3_aaa_exact, x3_aab_exact, x3_abb_exact, x3_bbb_exact = contract_vt3_exact(H, hbar, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction")
    t1 = time.time()
    # WARNING: When you sort and re-sort excitation lists on-the-fly, you have to make sure that the amps
    # come with a matching excitation list.
    t3_aaa, t3_excits_aaa, x3_aaa, t3_aab, t3_excits_aab, x3_aab, t3_abb, t3_excits_abb, x3_abb = contract_vt3_fly(hbar, H, T, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    nua, noa = T.a.shape
    nub, nob = T.b.shape

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aaa"])):
        a, b, c, i, j, k = [x - 1 for x in t3_excits_aaa[:, idet]]
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
        a, b, c, i, j, k = [x - 1 for x in t3_excits_aab[:, idet]]
        denom = (
                    H.a.oo[i, i] + H.a.oo[j, j] + H.b.oo[k, k]
                   -H.a.vv[a, a] - H.a.vv[b, b] - H.b.vv[c, c]
        )
        error = x3_aab[idet] - x3_aab_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-012:
            flag = False
    if flag:
        print("T3B update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3B update FAILED!", "Cumulative Error = ", err_cum)
    
    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["abb"])):
        a, b, c, i, j, k = [x - 1 for x in t3_excits_abb[:, idet]]
        denom = (
                    H.a.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
                   -H.a.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
        )
        error = x3_abb[idet] - x3_abb_exact[a, b, c, i, j, k]/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-012:
            flag = False
    if flag:
        print("T3C update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3C update FAILED!", "Cumulative Error = ", err_cum)

    #flag = True
    #err_cum = 0.0
    #for idet in range(len(T3_amplitudes["bbb"])):
    #    a, b, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
    #    denom = (
    #                H.b.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
    #               -H.b.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
    #    )
    #    error = x3_bbb[idet] - x3_bbb_exact[a, b, c, i, j, k]/denom
    #    err_cum += abs(error)
    #    if abs(error) > 1.0e-010:
    #        flag = False
    #if flag:
    #    print("T3D update passed!", "Cumulative Error = ", err_cum)
    #else:
    #    print("T3D update FAILED!", "Cumulative Error = ", err_cum)

