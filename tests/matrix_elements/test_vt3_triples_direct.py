"""In this script, we test out the idea of using the 'amplitude-driven' approach
to constructing the sparse triples projection < ijkabc | (H(2) * T3)_C | 0 >, where
T3 is sparse and defined over a given list of triples."""
import numpy as np
import time

from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import ccp_linear_loops

def kd(i, j):
    if i == j:
        return 1.0
    else:
        return 0.0

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

    # MM(2,3)
    # x3a = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    # x3a += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (Hbar*T3)
    x3a = -(3.0 / 36.0) * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aaa, optimize=True)
    # x3a += (3.0 / 36.0) * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aaa, optimize=True)
    # x3a += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True) # (k/ij) = 3
    # x3a += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) # (c/ab) = 3
    # x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) # (c/ij)(k/ij) = 9
    # x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) # (c/ij)(k/ij) = 9


    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)


    return x3a

def contract_vt3_fly(H, H0, T, T3_excitations, T3_amplitudes):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # Residual containers
    resid_aaa = np.zeros(len(T3_amplitudes["aaa"]))
    resid_aab = np.zeros(len(T3_amplitudes["aab"]))
    resid_abb = np.zeros(len(T3_amplitudes["abb"]))
    resid_bbb = np.zeros(len(T3_amplitudes["bbb"]))

    # Loop over aaa determinants
    for idet in range(len(T3_amplitudes["aaa"])):

        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]

        for m in range(noa):
            if m <= i: # m <-> i
                t_amp = T.aaa[a, b, c, m, j, k]
                resid_aaa[idet] -= H.a.oo[m, i] * t_amp
            elif i < m < k: # m <-> j
                t_amp = T.aaa[a, b, c, i, m, k]
                resid_aaa[idet] -= H.a.oo[m, j] * t_amp
            elif i >= k: # m <-> k
                t_amp = T.aaa[a, b, c, i, j, m]
                resid_aaa[idet] -= H.a.oo[m, k] * t_amp

        # for jdet in range(len(T3_amplitudes["aaa"])):

        #     d, e, f, l, m, n = [x - 1 for x in T3_excitations["aaa"][jdet]]

        #     # Get the particular aaa T3 amplitude
        #     t_amp = T3_amplitudes["aaa"][jdet]

        #     hmatel = 0.0

        #     # diagram 1: -A(abc)A(jk)A(i/jk)A(l/mn)[ d(j,m)d(k,n)d(b,e)d(c,f)d(a,d)h(l,i) ]
        #     #           -> -d(a,d)d(b,e)d(c,f)* [A(ijk)(A/lmn) d(j,m)d(k,n)h(l,i)]
        #     hmatel += kd(a, d) * kd(b, e) * kd(c, f) * (
        #         # (1)
        #         - kd(j, m) * kd(k, n) * H.a.oo[l, i]  # (1)
        #         + kd(i, m) * kd(k, n) * H.a.oo[l, j]  # (ij)
        #         + kd(j, m) * kd(i, n) * H.a.oo[l, k]  # (ik)
        #         + kd(k, m) * kd(j, n) * H.a.oo[l, i]  # (jk)
        #         - kd(i, m) * kd(j, n) * H.a.oo[l, k]  # (ij)(jk)
        #         - kd(k, m) * kd(i, n) * H.a.oo[l, j]  # (ik)(jk)
        #         # (lm)
        #         + kd(j, l) * kd(k, n) * H.a.oo[m, i]  # (1)
        #         - kd(i, l) * kd(k, n) * H.a.oo[m, j]  # (ij)
        #         - kd(j, l) * kd(i, n) * H.a.oo[m, k]  # (ik)
        #         - kd(k, l) * kd(j, n) * H.a.oo[m, i]  # (jk)
        #         + kd(i, l) * kd(j, n) * H.a.oo[m, k]  # (ij)(jk)
        #         + kd(k, l) * kd(i, n) * H.a.oo[m, j]  # (ik)(jk)
        #         # (ln)
        #         + kd(j, m) * kd(k, l) * H.a.oo[n, i]  # (1)
        #         - kd(i, m) * kd(k, l) * H.a.oo[n, j]  # (ij)
        #         - kd(j, m) * kd(i, l) * H.a.oo[n, k]  # (ik)
        #         - kd(k, m) * kd(j, l) * H.a.oo[n, i]  # (jk)
        #         + kd(i, m) * kd(j, l) * H.a.oo[n, k]  # (ij)(jk)
        #         + kd(k, m) * kd(i, l) * H.a.oo[n, j]  # (ik)(jk)
        #     )

            # diagram 2:


            # # diagram 5: A(jk)A(bc)A(i/jk)A(a/bc)A(l/mn)A(d/ef)[ d(k,n)d(b,e)d(j,m)d(c,f)h(a,l,i,d) ]
            # hmatel += (
            #     + kd(k, n) * kd(b, e) * kd(j, m) * kd(c, f) * H.aa.voov[a, l, i, d] # (1)
            #     - kd(k, n) * kd(b, e) * kd(i, m) * kd(c, f) * H.aa.voov[a, l, j, d] # (ij)
            #     - kd(i, n) * kd(b, e) * kd(j, m) * kd(c, f) * H.aa.voov[a, l, k, d] # (ik)
            #     - kd(k, n) * kd(a, e) * kd(j, m) * kd(c, f) * H.aa.voov[b, l, i, d]  # (ab)
            #     + kd(k, n) * kd(a, e) * kd(i, m) * kd(c, f) * H.aa.voov[b, l, j, d]  # (ij)(ab)
            #     + kd(i, n) * kd(a, e) * kd(j, m) * kd(c, f) * H.aa.voov[b, l, k, d]  # (ik)(ab)
            #     - kd(k, n) * kd(b, e) * kd(j, m) * kd(a, f) * H.aa.voov[c, l, i, d]  # (ac)
            #     + kd(k, n) * kd(b, e) * kd(i, m) * kd(a, f) * H.aa.voov[c, l, j, d]  # (ij)(ac)
            #     + kd(i, n) * kd(b, e) * kd(j, m) * kd(a, f) * H.aa.voov[c, l, k, d]  # (ik)(ac)
            #     - kd(k, n) * kd(b, e) * kd(j, l) * kd(c, f) * H.aa.voov[a, m, i, d]  # (lm)
            #     + kd(k, n) * kd(b, e) * kd(i, l) * kd(c, f) * H.aa.voov[a, m, j, d]  # (ij)(lm)
            #     + kd(i, n) * kd(b, e) * kd(j, l) * kd(c, f) * H.aa.voov[a, m, k, d]  # (ik)(lm)
            #     + kd(k, n) * kd(a, e) * kd(j, l) * kd(c, f) * H.aa.voov[b, m, i, d]  # (ab)(lm)
            #     - kd(k, n) * kd(a, e) * kd(i, l) * kd(c, f) * H.aa.voov[b, m, j, d]  # (ij)(ab)(lm)
            #     - kd(i, n) * kd(a, e) * kd(j, l) * kd(c, f) * H.aa.voov[b, m, k, d]  # (ik)(ab)(lm)
            #     + kd(k, n) * kd(b, e) * kd(j, l) * kd(a, f) * H.aa.voov[c, m, i, d]  # (ac)(lm)
            #     - kd(k, n) * kd(b, e) * kd(i, l) * kd(a, f) * H.aa.voov[c, m, j, d]  # (ij)(ac)(lm)
            #     - kd(i, n) * kd(b, e) * kd(j, l) * kd(a, f) * H.aa.voov[c, m, k, d]  # (ik)(ac)(lm)
            #     - kd(k, l) * kd(b, e) * kd(j, m) * kd(c, f) * H.aa.voov[a, n, i, d]  # (ln)
            #     + kd(k, l) * kd(b, e) * kd(i, m) * kd(c, f) * H.aa.voov[a, n, j, d]  # (ij)(ln)
            #     + kd(i, l) * kd(b, e) * kd(j, m) * kd(c, f) * H.aa.voov[a, n, k, d]  # (ik)(ln)
            #     + kd(k, l) * kd(a, e) * kd(j, m) * kd(c, f) * H.aa.voov[b, n, i, d]  # (ab)(ln)
            #     - kd(k, l) * kd(a, e) * kd(i, m) * kd(c, f) * H.aa.voov[b, n, j, d]  # (ij)(ab)(ln)
            #     - kd(i, l) * kd(a, e) * kd(j, m) * kd(c, f) * H.aa.voov[b, n, k, d]  # (ik)(ab)(ln)
            #     + kd(k, l) * kd(b, e) * kd(j, m) * kd(a, f) * H.aa.voov[c, n, i, d]  # (ac)(ln)
            #     - kd(k, l) * kd(b, e) * kd(i, m) * kd(a, f) * H.aa.voov[c, n, j, d]  # (ij)(ac)(ln)
            #     - kd(i, l) * kd(b, e) * kd(j, m) * kd(a, f) * H.aa.voov[c, n, k, d]  # (ik)(ac)(ln)
            #     - kd(k, n) * kd(b, d) * kd(j, m) * kd(c, f) * H.aa.voov[a, l, i, e]  # (de)
            #     + kd(k, n) * kd(b, d) * kd(i, m) * kd(c, f) * H.aa.voov[a, l, j, e]  # (ij)(de)
            #     + kd(i, n) * kd(b, d) * kd(j, m) * kd(c, f) * H.aa.voov[a, l, k, e]  # (ik)(de)
            #     + kd(k, n) * kd(a, d) * kd(j, m) * kd(c, f) * H.aa.voov[b, l, i, e]  # (ab)(de)
            #     - kd(k, n) * kd(a, d) * kd(i, m) * kd(c, f) * H.aa.voov[b, l, j, e]  # (ij)(ab)(de)
            #     - kd(i, n) * kd(a, d) * kd(j, m) * kd(c, f) * H.aa.voov[b, l, k, e]  # (ik)(ab)(de)
            #     + kd(k, n) * kd(b, d) * kd(j, m) * kd(a, f) * H.aa.voov[c, l, i, e]  # (ac)(de)
            #     - kd(k, n) * kd(b, d) * kd(i, m) * kd(a, f) * H.aa.voov[c, l, j, e]  # (ij)(ac)(de)
            #     - kd(i, n) * kd(b, d) * kd(j, m) * kd(a, f) * H.aa.voov[c, l, k, e]  # (ik)(ac)(de)
            #     + kd(k, n) * kd(b, d) * kd(j, l) * kd(c, f) * H.aa.voov[a, m, i, e]  # (lm)(de)
            #     - kd(k, n) * kd(b, d) * kd(i, l) * kd(c, f) * H.aa.voov[a, m, j, e]  # (ij)(lm)(de)
            #     - kd(i, n) * kd(b, d) * kd(j, l) * kd(c, f) * H.aa.voov[a, m, k, e]  # (ik)(lm)(de)
            #     - kd(k, n) * kd(a, d) * kd(j, l) * kd(c, f) * H.aa.voov[b, m, i, e]  # (ab)(lm)(de)
            #     + kd(k, n) * kd(a, d) * kd(i, l) * kd(c, f) * H.aa.voov[b, m, j, e]  # (ij)(ab)(lm)(de)
            #     + kd(i, n) * kd(a, d) * kd(j, l) * kd(c, f) * H.aa.voov[b, m, k, e]  # (ik)(ab)(lm)(de)
            #     - kd(k, n) * kd(b, d) * kd(j, l) * kd(a, f) * H.aa.voov[c, m, i, e]  # (ac)(lm)(de)
            #     + kd(k, n) * kd(b, d) * kd(i, l) * kd(a, f) * H.aa.voov[c, m, j, e]  # (ij)(ac)(lm)(de)
            #     + kd(i, n) * kd(b, d) * kd(j, l) * kd(a, f) * H.aa.voov[c, m, k, e]  # (ik)(ac)(lm)(de)
            #     + kd(k, l) * kd(b, d) * kd(j, m) * kd(c, f) * H.aa.voov[a, n, i, e]  # (ln)(de)
            #     - kd(k, l) * kd(b, d) * kd(i, m) * kd(c, f) * H.aa.voov[a, n, j, e]  # (ij)(ln)(de)
            #     - kd(i, l) * kd(b, d) * kd(j, m) * kd(c, f) * H.aa.voov[a, n, k, e]  # (ik)(ln)(de)
            #     - kd(k, l) * kd(a, d) * kd(j, m) * kd(c, f) * H.aa.voov[b, n, i, e]  # (ab)(ln)(de)
            #     + kd(k, l) * kd(a, d) * kd(i, m) * kd(c, f) * H.aa.voov[b, n, j, e]  # (ij)(ab)(ln)(de)
            #     + kd(i, l) * kd(a, d) * kd(j, m) * kd(c, f) * H.aa.voov[b, n, k, e]  # (ik)(ab)(ln)(de)
            #     - kd(k, l) * kd(b, d) * kd(j, m) * kd(a, f) * H.aa.voov[c, n, i, e]  # (ac)(ln)(de)
            #     + kd(k, l) * kd(b, d) * kd(i, m) * kd(a, f) * H.aa.voov[c, n, j, e]  # (ij)(ac)(ln)(de)
            #     + kd(i, l) * kd(b, d) * kd(j, m) * kd(a, f) * H.aa.voov[c, n, k, e]  # (ik)(ac)(ln)(de)
            #     - kd(k, n) * kd(b, e) * kd(j, m) * kd(c, d) * H.aa.voov[a, l, i, f]  # (df)
            #     + kd(k, n) * kd(b, e) * kd(i, m) * kd(c, d) * H.aa.voov[a, l, j, f]  # (ij)(df)
            #     + kd(i, n) * kd(b, e) * kd(j, m) * kd(c, d) * H.aa.voov[a, l, k, f]  # (ik)(df)
            #     + kd(k, n) * kd(a, e) * kd(j, m) * kd(c, d) * H.aa.voov[b, l, i, f]  # (ab)(df)
            #     - kd(k, n) * kd(a, e) * kd(i, m) * kd(c, d) * H.aa.voov[b, l, j, f]  # (ij)(ab)(df)
            #     - kd(i, n) * kd(a, e) * kd(j, m) * kd(c, d) * H.aa.voov[b, l, k, f]  # (ik)(ab)(df)
            #     + kd(k, n) * kd(b, e) * kd(j, m) * kd(a, d) * H.aa.voov[c, l, i, f]  # (ac)(df)
            #     - kd(k, n) * kd(b, e) * kd(i, m) * kd(a, d) * H.aa.voov[c, l, j, f]  # (ij)(ac)(df)
            #     - kd(i, n) * kd(b, e) * kd(j, m) * kd(a, d) * H.aa.voov[c, l, k, f]  # (ik)(ac)(df)
            #     + kd(k, n) * kd(b, e) * kd(j, l) * kd(c, d) * H.aa.voov[a, m, i, f]  # (lm)(df)
            #     - kd(k, n) * kd(b, e) * kd(i, l) * kd(c, d) * H.aa.voov[a, m, j, f]  # (ij)(lm)(df)
            #     - kd(i, n) * kd(b, e) * kd(j, l) * kd(c, d) * H.aa.voov[a, m, k, f]  # (ik)(lm)(df)
            #     - kd(k, n) * kd(a, e) * kd(j, l) * kd(c, d) * H.aa.voov[b, m, i, f]  # (ab)(lm)(df)
            #     + kd(k, n) * kd(a, e) * kd(i, l) * kd(c, d) * H.aa.voov[b, m, j, f]  # (ij)(ab)(lm)(df)
            #     + kd(i, n) * kd(a, e) * kd(j, l) * kd(c, d) * H.aa.voov[b, m, k, f]  # (ik)(ab)(lm)(df)
            #     - kd(k, n) * kd(b, e) * kd(j, l) * kd(a, d) * H.aa.voov[c, m, i, f]  # (ac)(lm)(df)
            #     + kd(k, n) * kd(b, e) * kd(i, l) * kd(a, d) * H.aa.voov[c, m, j, f]  # (ij)(ac)(lm)(df)
            #     + kd(i, n) * kd(b, e) * kd(j, l) * kd(a, d) * H.aa.voov[c, m, k, f]  # (ik)(ac)(lm)(df)
            #     + kd(k, l) * kd(b, e) * kd(j, m) * kd(c, d) * H.aa.voov[a, n, i, f]  # (ln)(df)
            #     - kd(k, l) * kd(b, e) * kd(i, m) * kd(c, d) * H.aa.voov[a, n, j, f]  # (ij)(ln)(df)
            #     - kd(i, l) * kd(b, e) * kd(j, m) * kd(c, d) * H.aa.voov[a, n, k, f]  # (ik)(ln)(df)
            #     - kd(k, l) * kd(a, e) * kd(j, m) * kd(c, d) * H.aa.voov[b, n, i, f]  # (ab)(ln)(df)
            #     + kd(k, l) * kd(a, e) * kd(i, m) * kd(c, d) * H.aa.voov[b, n, j, f]  # (ij)(ab)(ln)(df)
            #     + kd(i, l) * kd(a, e) * kd(j, m) * kd(c, d) * H.aa.voov[b, n, k, f]  # (ik)(ab)(ln)(df)
            #     - kd(k, l) * kd(b, e) * kd(j, m) * kd(a, d) * H.aa.voov[c, n, i, f]  # (ac)(ln)(df)
            #     + kd(k, l) * kd(b, e) * kd(i, m) * kd(a, d) * H.aa.voov[c, n, j, f]  # (ij)(ac)(ln)(df)
            #     + kd(i, l) * kd(b, e) * kd(j, m) * kd(a, d) * H.aa.voov[c, n, k, f]  # (ik)(ac)(ln)(df)
            # )

            #resid_aaa[idet] += hmatel * t_amp

    return resid_aaa

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
        atom=methylene,
        basis="ccpvdz",
        symmetry="C2V",
        spin=0, 
        charge=0,
        unit="Bohr",
        cart=False,
    )
    mf = scf.ROHF(mol).run()

    system, H = load_pyscf_integrals(mf, nfrozen=1)
    system.print_info()

    calculation = Calculation(calculation_type="ccsdt")
    T, cc_energy, converged = cc_driver(calculation, system, H)
    hbar = get_ccsd_intermediates(T, H)

    T3_excitations, T3_amplitudes = get_T3_list(T)

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T3 contraction", end="")
    t1 = time.time()
    x3_aaa_exact = contract_vt3_exact(H, hbar, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction", end="")
    t1 = time.time()
    x3_aaa = contract_vt3_fly(hbar, H, T, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    nua, noa = T.a.shape
    nub, nob = T.b.shape

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aaa"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        denom = (
                    H.a.oo[i, i] + H.a.oo[j, j] + H.a.oo[k, k]
                   -H.a.vv[a, a] - H.a.vv[b, b] - H.a.vv[c, c]
        )
        error = (x3_aaa[idet] - x3_aaa_exact[a, b, c, i, j, k])#/denom
        err_cum += abs(error)
        if abs(error) > 1.0e-012:
            #print(a, b, c, i, j, k, "Expected = ", x3_aaa_exact[a, b, c, i, j, k], "Got = ", x3_aaa[idet])
            flag = False
    if flag:
        print("T3A update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3A update FAILED!", "Cumulative Error = ", err_cum)
    

    # flag = True
    # err_cum = 0.0
    # for idet in range(len(T3_amplitudes["aab"])):
    #     a, b, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
    #     denom = (
    #                 H.a.oo[i, i] + H.a.oo[j, j] + H.b.oo[k, k]
    #                -H.a.vv[a, a] - H.a.vv[b, b] - H.b.vv[c, c]
    #     )
    #     error = (x3_aab[idet] - x3_aab_exact[a, b, c, i, j, k])/denom
    #     err_cum += abs(error)
    #     if abs(error) > 1.0e-010:
    #         flag = False
    # if flag:
    #     print("T3B update passed!", "Cumulative Error = ", err_cum)
    # else:
    #     print("T3B update FAILED!", "Cumulative Error = ", err_cum)
    #
    # flag = True
    # err_cum = 0.0
    # for idet in range(len(T3_amplitudes["abb"])):
    #     a, b, c, i, j, k = [x - 1 for x in T3_excitations["abb"][idet]]
    #     denom = (
    #                 H.a.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
    #                -H.a.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
    #     )
    #     error = (x3_abb[idet] - x3_abb_exact[a, b, c, i, j, k])/denom
    #     err_cum += abs(error)
    #     if abs(error) > 1.0e-010:
    #         flag = False
    # if flag:
    #     print("T3C update passed!", "Cumulative Error = ", err_cum)
    # else:
    #     print("T3C update FAILED!", "Cumulative Error = ", err_cum)
    #
    # flag = True
    # err_cum = 0.0
    # for idet in range(len(T3_amplitudes["bbb"])):
    #     a, b, c, i, j, k = [x - 1 for x in T3_excitations["bbb"][idet]]
    #     denom = (
    #                 H.b.oo[i, i] + H.b.oo[j, j] + H.b.oo[k, k]
    #                -H.b.vv[a, a] - H.b.vv[b, b] - H.b.vv[c, c]
    #     )
    #     error = (x3_bbb[idet] - x3_bbb_exact[a, b, c, i, j, k])/denom
    #     err_cum += abs(error)
    #     if abs(error) > 1.0e-010:
    #         flag = False
    # if flag:
    #     print("T3D update passed!", "Cumulative Error = ", err_cum)
    # else:
    #     print("T3D update FAILED!", "Cumulative Error = ", err_cum)

