"""In this script, we test out the idea of using the 'amplitude-driven' approach
to constructing the sparse triples projection < ijkabc | (H(2) * T3)_C | 0 >, where
T3 is sparse and defined over a given list of triples."""
import numpy as np
import time

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation

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
                            T3_excitations["aaa"].append([a, b, c, i, j, k])
                            T3_amplitudes["aaa"].append(T.aaa[a, b, c, i, j, k])
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(nob):
                            T3_excitations["aab"].append([a, b, c, i, j, k])
                            T3_amplitudes["aab"].append(T.aab[a, b, c, i, j, k])
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j + 1, nob):
                            T3_excitations["abb"].append([a, b, c, i, j, k])
                            T3_amplitudes["abb"].append(T.abb[a, b, c, i, j, k])
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        for k in range(j + 1, nob):
                            T3_excitations["bbb"].append([a, b, c, i, j, k])
                            T3_amplitudes["bbb"].append(T.bbb[a, b, c, i, j, k])

    for key in T3_excitations.keys():
        T3_excitations[key] = np.asarray(T3_excitations[key])
        T3_amplitudes[key] = np.asarray(T3_amplitudes[key])

    return T3_excitations, T3_amplitudes

def contract_vt3_exact(H, T):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # Residual containers
    x3a = np.zeros((nua, nua, nua, noa, noa, noa))
    x3b = np.zeros((nua, nua, nub, noa, noa, nob))
    x3c = np.zeros((nua, nub, nub, noa, nob, nob))
    x3d = np.zeros((nub, nub, nub, nob, nob, nob))

    x3a -= (3.0 / 36.0) * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aaa, optimize=True)
    x3a += (3.0 / 36.0) * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aaa, optimize=True)
    x3a += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True) # (k/ij) = 3
    x3a += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) # (c/ab) = 3
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) # (c/ij)(k/ij) = 9
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) # (c/ij)(k/ij) = 9

    I2A_vooo = (
                0.5 * np.einsum("mnef,aefijn->amij", H.aa.oovv, T.aaa, optimize=True)
                + np.einsum("mnef,aefijn->amij", H.ab.oovv, T.aab, optimize=True)
    )

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

    return x3a, x3b, x3c, x3d, I2A_vooo

def contract_vt3_fly(H, T3_excitations, T3_amplitudes):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # Residual containers
    x3a = np.zeros((nua, nua, nua, noa, noa, noa))
    x3b = np.zeros((nua, nua, nub, noa, noa, nob))
    x3c = np.zeros((nua, nub, nub, noa, nob, nob))
    x3d = np.zeros((nub, nub, nub, nob, nob, nob))

    # Intermediate containers
    I2A_vooo = np.zeros((nua, noa, noa, noa))
    I2A_vvov = np.zeros((nua, nua, noa, nua))
    I2B_vooo = np.zeros((nua, nob, noa, nob))
    I2B_ovoo = np.zeros((noa, nub, noa, nob))
    I2B_vvov = np.zeros((nua, nub, noa, nub))
    I2B_vvvo = np.zeros((nua, nub, nua, nob))
    I2C_vooo = np.zeros((nub, nob, nob, nob))
    I2C_vvov = np.zeros((nub, nub, nob, nub))

    # Loop over aaa determinants
    for idet in range(len(T3_amplitudes["aaa"])):

        # Get the particular aaa T3 amplitude
        t_amp = T3_amplitudes["aaa"][idet]

        # x3a(abcijk) <- -A(abc)A(i/jk)A(jk)A(m/jk) h1a(mi) * t3a(abcmjk)
        #              = -A(abc)A(ijk)[ A(m/jk) h1a(mi) * t3a(abcmjk) ]
        a, b, c, m, j, k = T3_excitations["aaa"][idet]
        x3a[a, b, c, :, j, k] -= H.a.oo[m, :] * t_amp # (1)
        x3a[a, b, c, :, m, k] += H.a.oo[j, :] * t_amp # (mj)
        x3a[a, b, c, :, j, m] += H.a.oo[k, :] * t_amp # (mk)
        # equivalent to...?
        # a, b, c, m, j, k = T3_excitations["aaa"][idet]
        # for i in range(j):
        #     x3a[a, b, c, i, j, k] -= H.a.oo[m, i] * t_amp # (1)
        #     x3a[a, b, c, i, m, k] += H.a.oo[j, i] * t_amp # (mj)
        #     x3a[a, b, c, i, j, m] += H.a.oo[k, i] * t_amp # (mk)

        # x3a(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1a(ae) * t3a(ebcijk)
        #              = A(abc)A(ijk)[ A(e/bc) h1a(ae) * t3a(ebcijk) ]
        e, b, c, i, j, k = T3_excitations["aaa"][idet]
        x3a[:, b, c, i, j, k] += H.a.vv[:, e] * t_amp # (1)
        x3a[:, e, c, i, j, k] -= H.a.vv[:, b] * t_amp # (be)
        x3a[:, b, e, i, j, k] -= H.a.vv[:, c] * t_amp # (ce)

        # x3a(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2a(mnij) * t3a(abcmnk) ]
        #             = A(abc)A(ijk)[ 1/2 A(k/mn) h2a(mnij) * t3a(abcmnk) ]
        a, b, c, m, n, k = T3_excitations["aaa"][idet]
        x3a[a, b, c, :, :, k] += 0.5 * H.aa.oooo[m, n, :, :] * t_amp # (1)
        x3a[a, b, c, :, :, m] -= 0.5 * H.aa.oooo[k, n, :, :] * t_amp # (mk)
        x3a[a, b, c, :, :, n] -= 0.5 * H.aa.oooo[m, k, :, :] * t_amp # (nk)

        # x3a(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2a(abef) * t3a(efcijk) ]
        #              = A(abc)A(ijk)[ 1/2 A(c/ef) h2a(abef) * t3a(efcijk) ]
        e, f, c, i, j, k = T3_excitations["aaa"][idet]
        x3a[:, :, c, i, j, k] += 0.5 * H.aa.vvvv[:, :, e, f] * t_amp # (1)
        x3a[:, :, e, i, j, k] -= 0.5 * H.aa.vvvv[:, :, c, f] * t_amp # (ec)
        x3a[:, :, f, i, j, k] -= 0.5 * H.aa.vvvv[:, :, e, c] * t_amp # (fc)

        # x3a(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
        #              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2a(amie) * t3a(ebcmjk) ]
        e, b, c, m, j, k = T3_excitations["aaa"][idet]
        x3a[:, b, c, :, j, k] += H.aa.voov[:, m, :, e] * t_amp # (1)
        x3a[:, b, c, :, m, k] -= H.aa.voov[:, j, :, e] * t_amp # (mj)
        x3a[:, b, c, :, j, m] -= H.aa.voov[:, k, :, e] * t_amp # (mk)
        x3a[:, e, c, :, j, k] -= H.aa.voov[:, m, :, b] * t_amp # (eb)
        x3a[:, e, c, :, m, k] += H.aa.voov[:, j, :, b] * t_amp # (eb)(mj)
        x3a[:, e, c, :, j, m] += H.aa.voov[:, k, :, b] * t_amp # (eb)(mk)
        x3a[:, b, e, :, j, k] -= H.aa.voov[:, m, :, c] * t_amp # (ec)
        x3a[:, b, e, :, m, k] += H.aa.voov[:, j, :, c] * t_amp # (ec)(mj)
        x3a[:, b, e, :, j, m] += H.aa.voov[:, k, :, c] * t_amp # (ec)(mk)

        # I2A(amij) <- A(ij) [A(a/ef)A(n/ij) h2a(mnef) * t3a(aefijn)]
        a, e, f, i, j, n = T3_excitations["aaa"][idet]
        I2A_vooo[a, :, i, j] += H.aa.oovv[:, n, e, f] * t_amp # (1)
        I2A_vooo[e, :, i, j] -= H.aa.oovv[:, n, a, f] * t_amp # (ae)
        I2A_vooo[f, :, i, j] -= H.aa.oovv[:, n, e, a] * t_amp # (af)
        I2A_vooo[a, :, n, j] -= H.aa.oovv[:, i, e, f] * t_amp # (in)
        I2A_vooo[e, :, n, j] += H.aa.oovv[:, i, a, f] * t_amp # (in)(ae)
        I2A_vooo[f, :, n, j] += H.aa.oovv[:, i, e, a] * t_amp # (in)(af)
        I2A_vooo[a, :, i, n] -= H.aa.oovv[:, j, e, f] * t_amp # (jn)
        I2A_vooo[e, :, i, n] += H.aa.oovv[:, j, a, f] * t_amp # (jn)(ae)
        I2A_vooo[f, :, i, n] += H.aa.oovv[:, j, e, a] * t_amp # (jn)(af)

    # Loop over aab determinants
    for idet in range(len(T3_amplitudes["aab"])):

        # Get the particular aab T3 amplitude
        t_amp = T3_amplitudes["aab"][idet]

        # x3a(abcijk) <- A(c/ab)A(ab)A(ij)A(k/ij)[ h2b(cmke) * t3b(abeijm) ]
        #              = A(abc)A(ijk)[ h2b(cmke) * t3b(abeijm) ]
        a, b, e, i, j, m = T3_excitations["aab"][idet]
        x3a[a, b, :, i, j, :] += H.ab.voov[:, m, :, e] * t_amp # (1)

        # I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
        a, e, f, i, j, n = T3_excitations["aab"][idet]
        I2A_vooo[a, :, i, j] += H.ab.oovv[:, n, e, f] * t_amp # (1)
        I2A_vooo[e, :, i, j] -= H.ab.oovv[:, n, a, f] * t_amp # (ae)


    # # Loop over abb determinants
    # for idet in range(len(T3_amplitudes["abb"])):
    #
    #     # Get the particular abb T3 amplitude
    #     t_amp = T3_amplitudes["abb"][idet]
    #
    #
    # # Loop over bbb determinants
    # for idet in range(len(T3_amplitudes["bbb"])):
    #
    #     # Get the particular bbb T3 amplitude
    #     t_amp = T3_amplitudes["bbb"][idet]


    # Antisymmetrize the final quantities
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

    I2A_vooo -= np.transpose(I2A_vooo, (0, 1, 3, 2))

    return x3a, x3b, x3c, x3d, I2A_vooo

if __name__ == "__main__":

    #ccpy_root = "/home2/gururang/ccpy"
    #ccpy_root = "/Users/harellab/Documents/ccpy"
    ccpy_root = "/Users/karthik/Documents/Python/ccpy"

    system, H = load_from_gamess(
            ccpy_root + "/examples/ext_corr/h2o-Re/h2o-Re.log",
            ccpy_root + "/examples/ext_corr/h2o-Re/onebody.inp",
            ccpy_root + "/examples/ext_corr/h2o-Re/twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    calculation = Calculation(calculation_type="ccsdt")
    T, cc_energy, converged = cc_driver(calculation, system, H)

    T3_excitations, T3_amplitudes = get_T3_list(T)

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T3 contraction", end="")
    t1 = time.time()
    x3_aaa_exact, x3_aab_exact, x3_abb_exact, x3_bbb_exact, I_aa_vooo_exact = contract_vt3_exact(H, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction", end="")
    t1 = time.time()
    x3_aaa, x3_aab, x3_abb, x3_bbb, I_aa_vooo = contract_vt3_fly(H, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    print("Error in x3a = ", np.linalg.norm(x3_aaa.flatten() - x3_aaa_exact.flatten()))
    print("Error in x3b = ", np.linalg.norm(x3_aab.flatten() - x3_aab_exact.flatten()))
    print("Error in x3c = ", np.linalg.norm(x3_abb.flatten() - x3_abb_exact.flatten()))
    print("Error in x3d = ", np.linalg.norm(x3_bbb.flatten() - x3_bbb_exact.flatten()))
    print("")
    print("Error in I2A_vooo = ", np.linalg.norm(I_aa_vooo.flatten() - I_aa_vooo_exact.flatten()))
