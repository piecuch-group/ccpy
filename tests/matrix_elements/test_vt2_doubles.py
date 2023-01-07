"""This script demonstrates how to efficiently construct projections of the 
< dense | dense * sparse | 0 > type. In particular, it treats those cases
where there are lines extending to the left originating from H which are
antisymmetrized with lines extending to the left from T. Here, it is shown
that one can take advantage of vectorization over the indical dimensions
corresponding to lines originating from H."""
import numpy as np
import time

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation

def get_T2_list(T):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    T2_excitations = {"aa" : [], "ab" : [], "bb" : []}
    T2_amplitudes = {"aa" : [], "ab" : [], "bb" : []}

    for a in range(nua):
        for b in range(a + 1, nua):
            for i in range(noa):
                for j in range(i + 1, noa):
                    T2_excitations["aa"].append([a, b, i, j])
                    T2_amplitudes["aa"].append(T.aa[a, b, i, j])

    for a in range(nua):
        for b in range(nub):
            for i in range(noa):
                for j in range(nob):
                    T2_excitations["ab"].append([a, b, i, j])
                    T2_amplitudes["ab"].append(T.ab[a, b, i, j])

    for a in range(nub):
        for b in range(a + 1, nub):
            for i in range(nob):
                for j in range(i + 1, nob):
                    T2_excitations["bb"].append([a, b, i, j])
                    T2_amplitudes["bb"].append(T.bb[a, b, i, j])

    for key in T2_excitations.keys():
        T2_excitations[key] = np.asarray(T2_excitations[key])
        T2_amplitudes[key] = np.asarray(T2_amplitudes[key])

    return T2_excitations, T2_amplitudes

def contract_vt2(H, T):

    x2_bb = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    x2_aa = np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)
    x2_aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)
    x2_aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, T.aa, optimize=True)
    x2_aa += np.einsum("amie,bejm->abij", H.ab.voov, T.ab, optimize=True)
    x2_aa -= np.transpose(x2_aa, (1, 0, 2, 3))
    x2_aa -= np.transpose(x2_aa, (0, 1, 3, 2))

    x2_ab = np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True) 
    x2_ab += np.einsum("amie,ebmj->abij", H.aa.voov, T.ab, optimize=True)
    #x2_ab += np.einsum("amie,ebmj->abij", H.ab.voov, T.bb, optimize=True)

    return x2_aa, x2_ab, x2_bb

def contract_vt2_fly_v2(H, T2_excitations, T2_amplitudes):

    noa, nua = H.a.ov.shape
    nob, nub = H.b.ov.shape

    x2_aa = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2_ab = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2_bb = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    n2aa = len(T2_amplitudes["aa"])
    n2ab = len(T2_amplitudes["ab"])
    n2bb = len(T2_amplitudes["bb"])

    # Loop over aa determinants
    for idet in range(n2aa):

        t_amp = T2_amplitudes["aa"][idet]

        # x2a(abij) <- A(ij)A(ab)[ A(mj)A(eb) h_aa(amie) * t_aa(ebmj) ]
        e, b, m, j = T2_excitations["aa"][idet]
        x2_aa[:, b, :, j] += H.aa.voov[:, m, :, e] * t_amp # (1)
        x2_aa[:, b, :, m] -= H.aa.voov[:, j, :, e] * t_amp # (mj)
        x2_aa[:, e, :, j] -= H.aa.voov[:, m, :, b] * t_amp # (eb)
        x2_aa[:, e, :, m] += H.aa.voov[:, j, :, b] * t_amp # (eb)(mj)

        # x2a(abij) <- A(ij) h_aa(abef) * t_aa(efij) = 1/2 A(ij)A(ab)[ h_aa(abef) * t_aa(efij) ]
        e, f, i, j = T2_excitations["aa"][idet]
        x2_aa[:, :, i, j] += 0.5 * H.aa.vvvv[:, :, e, f] * t_amp # (1)

        # x2a(abij) <- A(ab) h_aa(mnij) * t_aa(abmn) = 1/2 A(ij)A(ab)[ h_aa(mnij) * t_aa(abmn) ]
        a, b, m, n = T2_excitations["aa"][idet]
        x2_aa[a, b, :, :] += 0.5 * H.aa.oooo[m, n, :, :] * t_amp # (1)

        # x2b(abij) <- A(im)A(ae) h_ab(mbej) * t_aa(aeim)
        a, e, i, m = T2_excitations["aa"][idet]
        x2_ab[a, :, i, :] += H.ab.ovvo[m, :, e, :] * t_amp # (1)
        x2_ab[a, :, m, :] -= H.ab.ovvo[i, :, e, :] * t_amp # (im)
        x2_ab[e, :, i, :] -= H.ab.ovvo[m, :, a, :] * t_amp # (ae)
        x2_ab[e, :, m, :] += H.ab.ovvo[i, :, a, :] * t_amp # (im)(ae)

    # Loop over ab determinants
    for idet in range(n2ab):

        t_amp = T2_amplitudes["ab"][idet]

        # x2a(abij) <- A(ij)A(ab)[ h_ab(amie) * t_ab(bejm) ]
        b, e, j, m = T2_excitations["ab"][idet]
        x2_aa[:, b, :, j] += H.ab.voov[:, m, :, e] * t_amp # (1)

        # x2b(abij) <- h_aa(amie) * t_ab(ebmj)
        e, b, m, j = T2_excitations["ab"][idet]
        x2_ab[:, b, :, j] += H.aa.voov[:, m, :, e] * t_amp # (1)

        # x2b(abij) <- 
    
    # Antisymmetrize; this takes care of setting diagonal elements to 0
    x2_aa -= np.transpose(x2_aa, (1, 0, 2, 3))
    x2_aa -= np.transpose(x2_aa, (0, 1, 3, 2))
    x2_bb -= np.transpose(x2_bb, (1, 0, 2, 3))
    x2_bb -= np.transpose(x2_bb, (0, 1, 3, 2))

    return x2_aa, x2_ab, x2_bb


def contract_vt2_fly(H, T2_excitations, T2_amplitudes):

    noa, nua = H.a.ov.shape
    nob, nub = H.b.ov.shape

    x2_aa = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2_ab = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2_bb = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    n2aa = len(T2_amplitudes["aa"])
    n2ab = len(T2_amplitudes["ab"])
    n2bb = len(T2_amplitudes["bb"])

    # Loop over aa determinants
    for idet in range(n2aa):

        t_amp = T2_amplitudes["aa"][idet]

        # x2a(abij) <- A(ij)A(ab)[ A(mj)A(eb) h_aa(amie) * t_aa(ebmj) ]
        e, b, m, j = T2_excitations["aa"][idet]

        x2_aa[:, b, :, j] += H.aa.voov[:, m, :, e] * t_amp # (1)
        x2_aa[b, :, :, j] = -1.0 * x2_aa[:, b, :, j] 
        x2_aa[:, b, j, :] = -1.0 * x2_aa[:, b, :, j]
        x2_aa[b, :, j, :] = 1.0 * x2_aa[:, b, :, j]

        x2_aa[:, b, :, m] -= H.aa.voov[:, j, :, e] * t_amp # (mj)
        x2_aa[b, :, :, m] = -1.0 * x2_aa[:, b, :, m]
        x2_aa[:, b, m, :] = -1.0 * x2_aa[:, b, :, m] 
        x2_aa[b, :, m, :] = 1.0 * x2_aa[:, b, :, m] 

        x2_aa[:, e, :, j] -= H.aa.voov[:, m, :, b] * t_amp # (eb)
        x2_aa[e, :, :, j] = -1.0 * x2_aa[:, e, :, j]
        x2_aa[:, e, j, :] = -1.0 * x2_aa[:, e, :, j]
        x2_aa[e, :, j, :] = 1.0 * x2_aa[:, e, :, j]

        x2_aa[:, e, :, m] += H.aa.voov[:, j, :, b] * t_amp # (eb)(mj)
        x2_aa[e, :, :, m] = -1.0 * x2_aa[:, e, :, m]
        x2_aa[:, e, m, :] = -1.0 * x2_aa[:, e, :, m]
        x2_aa[e, :, m, :] = 1.0 * x2_aa[:, e, :, m]

        # x2a(abij) <- A(ij) h_aa(abef) * t_aa(efij)
        e, f, i, j = T2_excitations["aa"][idet]

        x2_aa[:, :, i, j] += H.aa.vvvv[:, :, e, f] * t_amp # (1)
        x2_aa[:, :, j, i] = -1.0 * x2_aa[:, :, i, j]       # (ij)

        # x2a(abij) <- A(ab) h_aa(mnij) * t_aa(abmn)
        a, b, m, n = T2_excitations["aa"][idet]

        x2_aa[a, b, :, :] += H.aa.oooo[m, n, :, :] * t_amp # (1)
        x2_aa[b, a, :, :] = -1.0 * x2_aa[a, b, :, :]       # (ab)

        # x2b(abij) <- A(im)A(ae) h_ab(mbej) * t_aa(aeim)
        a, e, i, m = T2_excitations["aa"][idet]

        x2_ab[a, :, i, :] += H.ab.ovvo[m, :, e, :] * t_amp # (1)
        x2_ab[a, :, m, :] -= H.ab.ovvo[i, :, e, :] * t_amp # (im)
        x2_ab[e, :, i, :] -= H.ab.ovvo[m, :, a, :] * t_amp # (ae)
        x2_ab[e, :, m, :] += H.ab.ovvo[i, :, a, :] * t_amp # (im)(ae)

    # Loop over ab determinants
    for idet in range(n2ab):

        t_amp = T2_amplitudes["ab"][idet]

        # x2a(abij) <- A(ij)A(ab)[ h_ab(amie) * t_ab(bejm) ]
        b, e, j, m = T2_excitations["ab"][idet]

        x2_aa[:, b, :, j] += H.ab.voov[:, m, :, e] * t_amp # (1)
        x2_aa[b, :, :, j] = -1.0 * x2_aa[:, b, :, j] 
        x2_aa[:, b, j, :] = -1.0 * x2_aa[:, b, :, j] 
        x2_aa[b, :, j, :] = 1.0 * x2_aa[:, b, :, j] 

        # x2b(abij) <- h_aa(amie) * t_ab(ebmj)
        e, b, m, j = T2_excitations["ab"][idet]

        x2_ab[:, b, :, j] += H.aa.voov[:, m, :, e] * t_amp # (1)

        # x2b(abij) <- 

    # IMPORTANT: Vectorization across the "a" and "i" indices include a == b and i == j, resulting
    # in non-zero entries for these diagonal elements. We need to explicitly set them to 0 to get
    # the right answer.
    for a in range(nua):
        x2_aa[a, a, :, :] *= 0.0
    for i in range(noa):
        x2_aa[:, :, i, i] *= 0.0
    for a in range(nub):
        x2_bb[a, a, :, :] *= 0.0
    for i in range(nob):
        x2_bb[:, :, i, i] *= 0.0

    return x2_aa, x2_ab, x2_bb


if __name__ == "__main__":

    #ccpy_root = "/Users/karthik/Documents/Python/ccpy"
    ccpy_root = "/home2/gururang/ccpy"

    system, H = load_from_gamess(
            ccpy_root + "/examples/ext_corr/h2o-Re/h2o-Re.log",
            ccpy_root + "/examples/ext_corr/h2o-Re/onebody.inp",
            ccpy_root + "/examples/ext_corr/h2o-Re/twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    calculation = Calculation(calculation_type="ccsd")
    T, cc_energy, converged = cc_driver(calculation, system, H)

    T2_excitations, T2_amplitudes = get_T2_list(T)

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact V*T2 contraction")
    t1 = time.time()
    x2_aa_exact, x2_ab_exact, x2_bb_exact = contract_vt2(H, T)
    print("   Completed in ", time.time() - t1, "seconds")

    # Get the on-the-fly contraction result
    print("   On-the-fly V*T2 contraction")
    t1 = time.time()
    x2_aa, x2_ab, x2_bb = contract_vt2_fly_v2(H, T2_excitations, T2_amplitudes)
    print("   Completed in ", time.time() - t1, "seconds")

    print("")
    print("Error in x2a = ", np.linalg.norm(x2_aa.flatten() - x2_aa_exact.flatten()))
    print("Error in x2b = ", np.linalg.norm(x2_ab.flatten() - x2_ab_exact.flatten()))
    print("Error in x2c = ", np.linalg.norm(x2_bb.flatten() - x2_bb_exact.flatten()))

