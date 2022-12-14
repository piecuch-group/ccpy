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

def contract_vt3_exact(H, T):

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
    I2B_vvvo = (
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
                  + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    )
    I2C_vooo = (
        H.bb.vooo + 0.5 * np.einsum("mnef,aefijn->amij", H.bb.oovv, T.bbb, optimize=True)
                  + np.einsum("nmfe,feanji->amij", H.ab.oovv, T.abb, optimize=True)
    )

    x3a = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    x3a += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)

    x3b = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    #x3b -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    #x3b += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    #x3b -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    x3b += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    x3b -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)

    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)

    x3b -= np.transpose(x3b, (1, 0, 2, 3, 4, 5)) # (ab)
    x3b -= np.transpose(x3b, (0, 1, 2, 4, 3, 5)) # (ij)

    return x3a, x3b, I2A_vooo, I2A_vvov, I2B_vooo, I2B_ovoo, I2B_vvov, I2B_vvvo, I2C_vooo, I2C_vvov

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
    I2B_vooo = H.ab.vooo.copy()
    I2B_ovoo = H.ab.ovoo.copy()
    I2B_vvov = H.ab.vvov.copy()
    I2B_vvvo = H.ab.vvvo.copy()
    I2C_vooo = 0.5 * (H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True))
    I2C_vvov = 0.5 * H.bb.vvov

    # Loop over aaa determinants
    for idet in range(len(T3_amplitudes["aaa"])):

        # Get the particular aaa T3 amplitude
        t_amp = T3_amplitudes["aaa"][idet]

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

    # Loop over abb determinants
    for idet in range(len(T3_amplitudes["abb"])):
        # Get the particular abb T3 amplitude
        t_amp = T3_amplitudes["abb"][idet]

        # I2B(abej) <- [A(fb)A(nj) -h2b(mnef) * t3c(afbmnj)]
        a, f, b, m, n, j = [x - 1 for x in T3_excitations["abb"][idet]]
        I2B_vvvo[a, b, :, j] = I2B_vvvo[a, b, :, j] - H.ab.oovv[m, n, :, f] * t_amp # (1)
        I2B_vvvo[a, f, :, j] = I2B_vvvo[a, f, :, j] + H.ab.oovv[m, n, :, b] * t_amp # (fb)
        I2B_vvvo[a, b, :, n] = I2B_vvvo[a, b, :, n] + H.ab.oovv[m, j, :, f] * t_amp # (nj)
        I2B_vvvo[a, f, :, n] = I2B_vvvo[a, f, :, n] - H.ab.oovv[m, j, :, b] * t_amp # (fb)(nj)

    # Loop over bbb determinants
    for idet in range(len(T3_amplitudes["bbb"])):
    
        # Get the particular bbb T3 amplitude
        t_amp = T3_amplitudes["bbb"][idet]

    # antisymmetrize
    # for a in range(nua):
    #     for m in range(noa):
    #         I2A_vooo[a, m, :, :] -= I2A_vooo[a, m, :, :].T
    # for i in range(noa):
    #     for e in range(nua):
    #         I2A_vvov[:, :, i, e] -= I2A_vvov[:, :, i, e].T

    # Update loop
    print(np.linalg.norm(I2A_vooo.flatten()))
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

        resid_aaa[idet] = res_mm23

    print(np.linalg.norm(I2A_vooo.flatten()))
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
        # for e in range(nub):
        #     # A(ij)A(ab) I2B(acie) * t2b(bejk)
        #     res_mm23 = res_mm23 + I2B_vvov[a, c, i, e] * T.ab[b, e, j, k]
        #     res_mm23 = res_mm23 - I2B_vvov[a, c, j, e] * T.ab[b, e, i, k]
        #     res_mm23 = res_mm23 - I2B_vvov[b, c, i, e] * T.ab[a, e, j, k]
        #     res_mm23 = res_mm23 + I2B_vvov[b, c, j, e] * T.ab[a, e, i, k]
        for m in range(noa):
            # -A(ij) h2b(mcjk) * t2a(abim)
            # res_mm23 = res_mm23 - I2B_ovoo[m, c, j, k] * T.aa[a, b, i, m]
            # res_mm23 = res_mm23 + I2B_ovoo[m, c, i, k] * T.aa[a, b, j, m]
            # -A(ab) h2a(amij) * t2b(bcmk)
            res_mm23 = res_mm23 - (I2A_vooo[a, m, i, j] - I2A_vooo[a, m, j, i]) * T.ab[b, c, m, k]
            res_mm23 = res_mm23 + (I2A_vooo[b, m, i, j] - I2A_vooo[b, m, j, i]) * T.ab[a, c, m, k]
        # for m in range(nob):
        #     # -A(ij)A(ab) h2b(amik) * t2b(bcjm)
        #     res_mm23 = res_mm23 - I2B_vooo[a, m, i, k] * T.ab[b, c, j, m]
        #     res_mm23 = res_mm23 + I2B_vooo[b, m, i, k] * T.ab[a, c, j, m]
        #     res_mm23 = res_mm23 + I2B_vooo[a, m, j, k] * T.ab[b, c, i, m]
        #     res_mm23 = res_mm23 - I2B_vooo[b, m, j, k] * T.ab[a, c, i, m]

        resid_aab[idet] = res_mm23

    return resid_aaa, resid_aab, I2A_vooo, I2A_vvov, I2B_vooo, I2B_ovoo, I2B_vvov, I2B_vvvo, I2C_vooo, I2C_vvov


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
    x3a, x3b, I2A_vooo, I2A_vvov, I2B_vooo, I2B_ovoo, I2B_vvov, I2B_vvvo, I2C_vooo, I2C_vvov = contract_vt3_exact(hbar, T)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction", end="")
    t1 = time.time()
    x3a_2, x3b_2, I2A_vooo_2, I2A_vvov_2, I2B_vooo_2, I2B_ovoo_2, I2B_vvov_2, I2B_vvvo_2, I2C_vooo_2, I2C_vvov_2 = contract_vt3_fly(hbar, H, T, T3_excitations, T3_amplitudes)
    print(" (Completed in ", time.time() - t1, "seconds)")


    print("")
    nua, noa = T.a.shape
    nub, nob = T.b.shape

    print("error in I2A_vooo = ", np.linalg.norm(I2A_vooo.flatten() - I2A_vooo_2.flatten()))
    print("error in I2A_vvov = ", np.linalg.norm(I2A_vvov.flatten() - I2A_vvov_2.flatten()))
    print("error in I2B_vvvo = ", np.linalg.norm(I2B_vvvo.flatten() - I2B_vvvo_2.flatten()))

    print("")
    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aaa"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aaa"][idet]]
        error = x3a_2[idet] - x3a[a, b, c, i, j, k]
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3A update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3A update FAILED!", "Cumulative Error = ", err_cum)

    flag = True
    err_cum = 0.0
    for idet in range(len(T3_amplitudes["aab"])):
        a, b, c, i, j, k = [x - 1 for x in T3_excitations["aab"][idet]]
        error = x3b_2[idet] - x3b[a, b, c, i, j, k]
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3B update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3B update FAILED!", "Cumulative Error = ", err_cum)
