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

    x3a = -(3.0 / 36.0) * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aaa, optimize=True)
    x3a += (3.0 / 36.0) * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aaa, optimize=True)
    x3a += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True) # (k/ij) = 3
    x3a += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) # (c/ab) = 3
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) # (c/ij)(k/ij) = 9
    x3a += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) # (c/ij)(k/ij) = 9

    x3b = -0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
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

    x3c = -0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
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

    x3d = -(1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    x3d += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    x3d += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    x3d += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    x3d += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    x3d += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)

    # Should also check intermediates
    I2A_vooo = H.aa.vooo + (
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

    return x3a, x3b, x3c, x3d

def contract_vt3_fly(H, T3_excitations, T3_amplitudes):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    # Residual containers
    x3a = np.zeros((nua, nua, nua, noa, noa, noa))
    x3b = np.zeros((nua, nua, nub, noa, noa, nob))
    x3c = np.zeros((nua, nub, nub, noa, nob, nob))
    x3d = np.zeros((nub, nub, nub, nob, nob, nob))

    # Intermediate containers
    I2A_vooo = 0.5 * H.aa.vooo
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

        # x3b[abcijk) < - A(ij)A(ab)[A(m/ij)A(e/ab) h2b(mcek) * t3a(abeijm)]
        a, b, e, i, j, m = T3_excitations["aaa"][idet]
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
        #             = A(abc)A(ijk)[ h2b(cmke) * t3b(abeijm) ]
        a, b, e, i, j, m = T3_excitations["aab"][idet]
        x3a[a, b, :, i, j, :] += H.ab.voov[:, m, :, e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(mj) -h1a(mi) * t3b(abcmjk)]
        a, b, c, m, j, k = T3_excitations["aab"][idet]
        x3b[a,b,c,:,j,k] = x3b[a,b,c,:,j,k] - H.a.oo[m,:] * t_amp # (1)
        x3b[a,b,c,:,m,k] = x3b[a,b,c,:,m,k] + H.a.oo[j,:] * t_amp # (mj)

        # x3b[abcijk) <- A(ij)A(ab) [-h1b(mk) * t3b(abcijm)]
        a, b, c, i, j, m = T3_excitations["aab"][idet]
        x3b[a,b,c,i,j,:] = x3b[a,b,c,i,j,:] - H.b.oo[m,:] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(be) h1a(ae) * t3b(ebcijk)]
        e, b, c, i, j, k = T3_excitations["aab"][idet]
        x3b[:,b,c,i,j,k] = x3b[:,b,c,i,j,k] + H.a.vv[:,e] * t_amp # (1)
        x3b[:,e,c,i,j,k] = x3b[:,e,c,i,j,k] - H.a.vv[:,b] * t_amp # (eb)

        # x3b[abcijk) <- A(ij)A(ab) [h1b(ce) * t3b(abeijk)]
        a, b, e, i, j, k = T3_excitations["aab"][idet]
        x3b[a,b,:,i,j,k] = x3b[a,b,:,i,j,k] + H.b.vv[:,e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [1/2 h2a(mnij) * t3b(abcmnk)]
        a, b, c, m, n, k = T3_excitations["aab"][idet]
        x3b[a,b,c,:,:,k] = x3b[a,b,c,:,:,k] + 0.5 * H.aa.oooo[m,n,:,:] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(im) h2b(mnjk) * t3b(abcimn)]
        a, b, c, i, m, n = T3_excitations["aab"][idet]
        x3b[a,b,c,i,:,:] = x3b[a,b,c,i,:,:] + H.ab.oooo[m,n,:,:] * t_amp # (1)
        x3b[a,b,c,m,:,:] = x3b[a,b,c,m,:,:] - H.ab.oooo[i,n,:,:] * t_amp # (im)

        # x3b[abcijk) <- A(ij)A(ab) [1/2 h2a(abef) * t3b(efcijk)]
        e, f, c, i, j, k = T3_excitations["aab"][idet]
        x3b[:,:,c,i,j,k] = x3b[:,:,c,i,j,k] + 0.5 * H.aa.vvvv[:,:,e,f] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(ae) h2b(bcef) * t3b(aefijk)]
        a, e, f, i, j, k = T3_excitations["aab"][idet]
        x3b[a,:,:,i,j,k] = x3b[a,:,:,i,j,k] + H.ab.vvvv[:,:,e,f] * t_amp # (1)
        x3b[e,:,:,i,j,k] = x3b[e,:,:,i,j,k] - H.ab.vvvv[:,:,a,f] * t_amp # (ae)

        # x3b[abcijk) <- A(ij)A(ab) [A(be)A(jm) h2a(amie) * t3b(ebcmjk)]
        e, b, c, m, j, k = T3_excitations["aab"][idet]
        x3b[:,b,c,:,j,k] = x3b[:,b,c,:,j,k] + H.aa.voov[:,m,:,e] * t_amp # (1)
        x3b[:,e,c,:,j,k] = x3b[:,e,c,:,j,k] - H.aa.voov[:,m,:,b] * t_amp # (be)
        x3b[:,b,c,:,m,k] = x3b[:,b,c,:,m,k] - H.aa.voov[:,j,:,e] * t_amp # (jm)
        x3b[:,e,c,:,m,k] = x3b[:,e,c,:,m,k] + H.aa.voov[:,j,:,b] * t_amp # (be)(jm)

        # x3b[abcijk) <- A(ij)A(ab) [h2c(cmke) * t3b(abeijm)]
        a, b, e, i, j, m = T3_excitations["aab"][idet]
        x3b[a,b,:,i,j,:] = x3b[a,b,:,i,j,:] + H.bb.voov[:,m,:,e] * t_amp # (1)

        # x3b[abcijk) <- A(ij)A(ab) [A(be) -h2b(amek) * t3b(ebcijm)]
        e, b, c, i, j, m = T3_excitations["aab"][idet]
        x3b[:,b,c,i,j,:] = x3b[:,b,c,i,j,:] - H.ab.vovo[:,m,e,:] * t_amp # (1)
        x3b[:,e,c,i,j,:] = x3b[:,e,c,i,j,:] + H.ab.vovo[:,m,b,:] * t_amp # (be)

        # x3b[abcijk) <- A(ij)A(ab) [A(jm) -h2b(mcie) * t3b(abemjk)]
        a, b, e, m, j, k = T3_excitations["aab"][idet]
        x3b[a,b,:,:,j,k] = x3b[a,b,:,:,j,k] - H.ab.ovov[m,:,:,e] * t_amp # (1)
        x3b[a,b,:,:,m,k] = x3b[a,b,:,:,m,k] + H.ab.ovov[j,:,:,e] * t_amp # (jm)

        # x3c(abcijk) <- A(jk)A(bc) [A(im)A(ae) h2b(mbej) * t3b(aecimk)]
        a, e, c, i, m, k = T3_excitations["aab"][idet]
        x3c[a,:,c,i,:,k] = x3c[a,:,c,i,:,k] + H.ab.ovvo[m,:,e,:] * t_amp # (1)
        x3c[a,:,c,m,:,k] = x3c[a,:,c,m,:,k] - H.ab.ovvo[i,:,e,:] * t_amp # (im)
        x3c[e,:,c,i,:,k] = x3c[e,:,c,i,:,k] - H.ab.ovvo[m,:,a,:] * t_amp # (ae)
        x3c[e,:,c,m,:,k] = x3c[e,:,c,m,:,k] + H.ab.ovvo[i,:,a,:] * t_amp # (im)(ae)


        # # I2A(amij) <- A(ij) [A(ae) h2b(mnef) * t3b(aefijn)]
        # a, e, f, i, j, n = T3_excitations["aab"][idet]
        # I2A_vooo[a, :, i, j] += H.ab.oovv[:, n, e, f] * t_amp # (1)
        # I2A_vooo[e, :, i, j] -= H.ab.oovv[:, n, a, f] * t_amp # (ae)

    # Loop over abb determinants
    for idet in range(len(T3_amplitudes["abb"])):
    
        # Get the particular abb T3 amplitude
        t_amp = T3_amplitudes["abb"][idet]

        #  x3b(abcijk) < - A(ij)A(ab)[A(ec)A(mk) h2b(amie) * t3c(becjmk)]
        b, e, c, j, m, k = T3_excitations["abb"][idet]
        x3b[:,b,c,:,j,k] = x3b[:,b,c,:,j,k] + H.ab.voov[:,m,:,e] * t_amp # (1)
        x3b[:,b,e,:,j,k] = x3b[:,b,e,:,j,k] - H.ab.voov[:,m,:,c] * t_amp # (ec)
        x3b[:,b,c,:,j,m] = x3b[:,b,c,:,j,m] - H.ab.voov[:,k,:,e] * t_amp # (mk)
        x3b[:,b,e,:,j,m] = x3b[:,b,e,:,j,m] + H.ab.voov[:,k,:,c] * t_amp # (ec)(mk)

        # x3c(abcijk) < - A(jk)A(bc)[-h1a(mi) * t3c(abcmjk)]
        a, b, c, m, j, k = T3_excitations["abb"][idet]
        x3c[a,b,c,:,j,k] = x3c[a,b,c,:,j,k] - H.a.oo[m,:] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(mk) -h1b(mj) * t3c(abcimk)]
        a, b, c, i, m, k = T3_excitations["abb"][idet]
        x3c[a,b,c,i,:,k] = x3c[a,b,c,i,:,k] - H.b.oo[m,:] * t_amp # (1)
        x3c[a,b,c,i,:,m] = x3c[a,b,c,i,:,m] + H.b.oo[k,:] * t_amp # (mk)

        # x3c(abcijk) < - A(jk)A(bc)[h1a(ae) * t3c(ebcijk)]
        e, b, c, i, j, k = T3_excitations["abb"][idet]
        x3c[:,b,c,i,j,k] = x3c[:,b,c,i,j,k] + H.a.vv[:,e] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec) h1b(be) * t3c(aecijk)]
        a, e, c, i, j, k = T3_excitations["abb"][idet]
        x3c[a,:,c,i,j,k] = x3c[a,:,c,i,j,k] + H.b.vv[:,e] * t_amp # (1)
        x3c[a,:,e,i,j,k] = x3c[a,:,e,i,j,k] - H.b.vv[:,c] * t_amp # (ec)


        # x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(mnjk) * t3c(abcimn)]
        a, b, c, i, m, n = T3_excitations["abb"][idet]
        x3c[a,b,c,i,:,:] = x3c[a,b,c,i,:,:] + 0.5 * H.bb.oooo[m,n,:,:] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(kn) h2b(mnij) * t3c(abcmnk)]
        a, b, c, m, n, k = T3_excitations["abb"][idet]
        x3c[a,b,c,:,:,k] = x3c[a,b,c,:,:,k] + H.ab.oooo[m,n,:,:] * t_amp # (1)
        x3c[a,b,c,:,:,n] = x3c[a,b,c,:,:,n] - H.ab.oooo[m,k,:,:] * t_amp # (kn)

        # x3c(abcijk) <- A(jk)A(bc) [1/2 h2c(bcef) * t3c(aefijk)]
        a, e, f, i, j, k = T3_excitations["abb"][idet]
        x3c[a,:,:,i,j,k] = x3c[a,:,:,i,j,k] + 0.5 * H.bb.vvvv[:,:,e,f] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(fc) h2b(abef) * t3c(efcijk)]
        e, f, c, i, j, k = T3_excitations["abb"][idet]
        x3c[:,:,c,i,j,k] = x3c[:,:,c,i,j,k] + H.ab.vvvv[:,:,e,f] * t_amp # (1)
        x3c[:,:,f,i,j,k] = x3c[:,:,f,i,j,k] - H.ab.vvvv[:,:,e,c] * t_amp # (fc)

        # x3c(abcijk) <- A(jk)A(bc) [h2a(amie) * t3c(ebcmjk)]
        e, b, c, m, j, k = T3_excitations["abb"][idet]
        x3c[:,b,c,:,j,k] = x3c[:,b,c,:,j,k] + H.aa.voov[:,m,:,e] * t_amp # (1)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec)(mk) h2c(bmje) * t3c(aecimk)]
        a, e, c, i, m, k = T3_excitations["abb"][idet]
        x3c[a,:,c,i,:,k] = x3c[a,:,c,i,:,k] + H.bb.voov[:,m,:,e] * t_amp # (1)
        x3c[a,:,e,i,:,k] = x3c[a,:,e,i,:,k] - H.bb.voov[:,m,:,c] * t_amp # (ec)
        x3c[a,:,c,i,:,m] = x3c[a,:,c,i,:,m] - H.bb.voov[:,k,:,e] * t_amp # (mk)
        x3c[a,:,e,i,:,m] = x3c[a,:,e,i,:,m] + H.bb.voov[:,k,:,c] * t_amp # (ec)(mk)

        # x3c(abcijk) <- A(jk)A(bc) [A(ec) -h2b(mbie) * t3c(aecmjk)]
        a, e, c, m, j, k = T3_excitations["abb"][idet]
        x3c[a,:,c,:,j,k] = x3c[a,:,c,:,j,k] - H.ab.ovov[m,:,:,e] * t_amp # (1)
        x3c[a,:,e,:,j,k] = x3c[a,:,e,:,j,k] + H.ab.ovov[m,:,:,c] * t_amp # (ec)

        # x3c(abcijk) <- A(jk)A(bc) [A(km) -h2b(amej) * t3c(ebcimk)]
        e, b, c, i, m, k = T3_excitations["abb"][idet]
        x3c[:,b,c,i,:,k] = x3c[:,b,c,i,:,k] - H.ab.vovo[:,m,e,:] * t_amp # (1)
        x3c[:,b,c,i,:,m] = x3c[:,b,c,i,:,m] + H.ab.vovo[:,k,e,:] * t_amp # (km)
    
        # x3d(abcijk) < - A(ijk)A(abc)[h2b(maei) * t3c(ebcmjk)]
        e, b, c, m, j, k = T3_excitations["abb"][idet]
        x3d[:,b,c,:,j,k] = x3d[:,b,c,:,j,k] + H.ab.ovvo[m,:,e,:] * t_amp # (1)
    
    # Loop over bbb determinants
    for idet in range(len(T3_amplitudes["bbb"])):
    
        # Get the particular bbb T3 amplitude
        t_amp = T3_amplitudes["bbb"][idet]

        # x3c(abcijk) <- A(jk)A(bc) [A(m/jk)A(e/bc) h2b(amie) * t3d(ebcmjk)]
        e, b, c, m, j, k = T3_excitations["bbb"][idet]
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
        a, b, c, m, j, k = T3_excitations["bbb"][idet]
        x3d[a,b,c,:,j,k] = x3d[a,b,c,:,j,k] - H.b.oo[m,:] * t_amp # (1)
        x3d[a,b,c,:,m,k] = x3d[a,b,c,:,m,k] + H.b.oo[j,:] * t_amp # (mj)
        x3d[a,b,c,:,j,m] = x3d[a,b,c,:,j,m] + H.b.oo[k,:] * t_amp # (mk)

        # x3d(abcijk) <- A(abc)A(a/bc)A(bc)A(e/bc) h1b(ae) * t3d(ebcijk)
        #              = A(abc)A(ijk)[ A(e/bc) h1b(ae) * t3d(ebcijk) ]
        e, b, c, i, j, k = T3_excitations["bbb"][idet]
        x3d[:,b,c,i,j,k] = x3d[:,b,c,i,j,k] + H.b.vv[:, e] * t_amp # (1)
        x3d[:,e,c,i,j,k] = x3d[:,e,c,i,j,k] - H.b.vv[:, b] * t_amp # (be)
        x3d[:,b,e,i,j,k] = x3d[:,b,e,i,j,k] - H.b.vv[:, c] * t_amp # (ce)

        # x3d(abcijk) <- A(abc)A(k/ij)[ A(k/mn) h2c(mnij) * t3d(abcmnk) ]
        #             = A(abc)A(ijk)[ 1/2 A(k/mn) h2c(mnij) * t3d(abcmnk) ]
        a, b, c, m, n, k = T3_excitations["bbb"][idet]
        x3d[a,b,c,:,:,k] = x3d[a,b,c,:,:,k] + 0.5 * H.bb.oooo[m,n,:,:] * t_amp # (1)
        x3d[a,b,c,:,:,m] = x3d[a,b,c,:,:,m] - 0.5 * H.bb.oooo[k,n,:,:] * t_amp # (mk)
        x3d[a,b,c,:,:,n] = x3d[a,b,c,:,:,n] - 0.5 * H.bb.oooo[m,k,:,:] * t_amp # (nk)

        # x3d(abcijk) <- A(c/ab)A(ijk)[ A(c/ef) h2c(abef) * t3d(efcijk) ]
        #              = A(abc)A(ijk)[ 1/2 A(c/ef) h2c(abef) * t3d(efcijk) ]
        e, f, c, i, j, k = T3_excitations["bbb"][idet]
        x3d[:,:,c,i,j,k] = x3d[:,:,c,i,j,k] + 0.5 * H.bb.vvvv[:,:,e,f] * t_amp # (1)
        x3d[:,:,e,i,j,k] = x3d[:,:,e,i,j,k] - 0.5 * H.bb.vvvv[:,:,c,f] * t_amp # (ec)
        x3d[:,:,f,i,j,k] = x3d[:,:,f,i,j,k] - 0.5 * H.bb.vvvv[:,:,e,c] * t_amp # (fc)

        # x3d(abcijk) <- A(a/bc)A(bc)A(jk)A(i/jk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
        #              = A(abc)A(ijk)[ A(e/bc)A(m/jk) h2c(amie) * t3d(ebcmjk) ]
        e, b, c, m, j, k = T3_excitations["bbb"][idet]
        x3d[:,b,c,:,j,k] = x3d[:,b,c,:,j,k] + H.bb.voov[:,m,:,e] * t_amp # (1)
        x3d[:,b,c,:,m,k] = x3d[:,b,c,:,m,k] - H.bb.voov[:,j,:,e] * t_amp # (mj)
        x3d[:,b,c,:,j,m] = x3d[:,b,c,:,j,m] - H.bb.voov[:,k,:,e] * t_amp # (mk)
        x3d[:,e,c,:,j,k] = x3d[:,e,c,:,j,k] - H.bb.voov[:,m,:,b] * t_amp # (eb)
        x3d[:,e,c,:,m,k] = x3d[:,e,c,:,m,k] + H.bb.voov[:,j,:,b] * t_amp # (eb)(mj)
        x3d[:,e,c,:,j,m] = x3d[:,e,c,:,j,m] + H.bb.voov[:,k,:,b] * t_amp # (eb)(mk)
        x3d[:,b,e,:,j,k] = x3d[:,b,e,:,j,k] - H.bb.voov[:,m,:,c] * t_amp # (ec)
        x3d[:,b,e,:,m,k] = x3d[:,b,e,:,m,k] + H.bb.voov[:,j,:,c] * t_amp # (ec)(mj)
        x3d[:,b,e,:,j,m] = x3d[:,b,e,:,j,m] + H.bb.voov[:,k,:,c] * t_amp # (ec)(mk)

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

    return x3a, x3b, x3c, x3d

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.Mole()

    methylene = """
                    C 0.0000000000 0.0000000000 -0.1160863568
                    H -1.8693479331 0.0000000000 0.6911102033
                    H 1.8693479331 0.0000000000  0.6911102033
                 """

    mol.build(
        atom=methylene,
        basis="ccpvdz",
        symmetry="C2V",
        spin=0, 
        charge=0,
        unit="Bohr"
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
    x3_aaa_exact, x3_aab_exact, x3_abb_exact, x3_bbb_exact = contract_vt3_exact(hbar, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction", end="")
    t1 = time.time()
    x3_aaa, x3_aab, x3_abb, x3_bbb = contract_vt3_fly(hbar, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    print("Error in x3a = ", np.linalg.norm(x3_aaa.flatten() - x3_aaa_exact.flatten()))
    print("Error in x3b = ", np.linalg.norm(x3_aab.flatten() - x3_aab_exact.flatten()))
    print("Error in x3c = ", np.linalg.norm(x3_abb.flatten() - x3_abb_exact.flatten()))
    print("Error in x3d = ", np.linalg.norm(x3_bbb.flatten() - x3_bbb_exact.flatten()))
    #print("")
    #print("Error in I2A_vooo = ", np.linalg.norm(I_aa_vooo.flatten() - I_aa_vooo_exact.flatten()))
