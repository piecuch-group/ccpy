"""This script does a direct check of the update function of the CC(P) method against
the full CCSDT method."""
import numpy as np
import time
from itertools import permutations
from copy import deepcopy

from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver
from ccpy.models.calculation import Calculation
from ccpy.models.operators import ClusterOperator

from ccpy.cc.ccsdt import update
from ccpy.cc.ccsdt_p_quadratic import update as update_p

def get_T3_list(T):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    T3_excitations = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    T3_amplitudes = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    pspace = {"aaa" : np.full((nua, nua, nua, noa, noa, noa), fill_value=False, dtype=bool),
              "aab" : np.full((nua, nua, nub, noa, noa, nob), fill_value=False, dtype=bool),
              "abb" : np.full((nua, nub, nub, noa, nob, nob), fill_value=False, dtype=bool),
              "bbb" : np.full((nub, nub, nub, nob, nob, nob), fill_value=False, dtype=bool)}

    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(j + 1, noa):
                            T3_excitations["aaa"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["aaa"].append(T.aaa[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b, c)):
                                for perms_occ in permutations((i, j, k)):
                                    a1, b1, c1 = perms_unocc
                                    i1, j1, k1 = perms_occ
                                    pspace["aaa"][a1, b1, c1, i1, j1, k1] = True
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(nob):
                            T3_excitations["aab"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["aab"].append(T.aab[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b)):
                                for perms_occ in permutations((i, j)):
                                    a1, b1 = perms_unocc
                                    i1, j1 = perms_occ
                                    pspace["aab"][a1, b1, c, i1, j1, k] = True
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j + 1, nob):
                            T3_excitations["abb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["abb"].append(T.abb[a, b, c, i, j, k])
                            for perms_unocc in permutations((b, c)):
                                for perms_occ in permutations((j, k)):
                                    b1, c1 = perms_unocc
                                    j1, k1 = perms_occ
                                    pspace["abb"][a, b1, c1, i, j1, k1] = True
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        for k in range(j + 1, nob):
                            T3_excitations["bbb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            T3_amplitudes["bbb"].append(T.bbb[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b, c)):
                                for perms_occ in permutations((i, j, k)):
                                    a1, b1, c1 = perms_unocc
                                    i1, j1, k1 = perms_occ
                                    pspace["bbb"][a1, b1, c1, i1, j1, k1] = True

    for key in T3_excitations.keys():
        T3_excitations[key] = np.asarray(T3_excitations[key])
        T3_amplitudes[key] = np.asarray(T3_amplitudes[key])

    return T3_excitations, T3_amplitudes, pspace

def get_R3_list(R):

    nua, noa = R.a.shape
    nub, nob = R.b.shape

    R3_excitations = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    R3_amplitudes = {"aaa" : [], "aab" : [], "abb" : [], "bbb" : []}
    pspace = {"aaa" : np.full((nua, nua, nua, noa, noa, noa), fill_value=False, dtype=bool),
              "aab" : np.full((nua, nua, nub, noa, noa, nob), fill_value=False, dtype=bool),
              "abb" : np.full((nua, nub, nub, noa, nob, nob), fill_value=False, dtype=bool),
              "bbb" : np.full((nub, nub, nub, nob, nob, nob), fill_value=False, dtype=bool)}

    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(j + 1, noa):
                            R3_excitations["aaa"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            R3_amplitudes["aaa"].append(T.aaa[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b, c)):
                                for perms_occ in permutations((i, j, k)):
                                    a1, b1, c1 = perms_unocc
                                    i1, j1, k1 = perms_occ
                                    pspace["aaa"][a1, b1, c1, i1, j1, k1] = True
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(nob):
                            R3_excitations["aab"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            R3_amplitudes["aab"].append(T.aab[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b)):
                                for perms_occ in permutations((i, j)):
                                    a1, b1 = perms_unocc
                                    i1, j1 = perms_occ
                                    pspace["aab"][a1, b1, c, i1, j1, k] = True
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j + 1, nob):
                            R3_excitations["abb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            R3_amplitudes["abb"].append(T.abb[a, b, c, i, j, k])
                            for perms_unocc in permutations((b, c)):
                                for perms_occ in permutations((j, k)):
                                    b1, c1 = perms_unocc
                                    j1, k1 = perms_occ
                                    pspace["abb"][a, b1, c1, i, j1, k1] = True
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        for k in range(j + 1, nob):
                            R3_excitations["bbb"].append([a+1, b+1, c+1, i+1, j+1, k+1])
                            R3_amplitudes["bbb"].append(T.bbb[a, b, c, i, j, k])
                            for perms_unocc in permutations((a, b, c)):
                                for perms_occ in permutations((i, j, k)):
                                    a1, b1, c1 = perms_unocc
                                    i1, j1, k1 = perms_occ
                                    pspace["bbb"][a1, b1, c1, i1, j1, k1] = True

    for key in R3_excitations.keys():
        R3_excitations[key] = np.asarray(R3_excitations[key])
        R3_amplitudes[key] = np.asarray(R3_amplitudes[key])

    return R3_excitations, R3_amplitudes, pspace



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
        cart=True,
    )
    mf = scf.ROHF(mol).run()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(calculation_type="ccsdt")
    T, cc_energy, converged = cc_driver(calculation, system, H)

    calculation=Calculation(calculation_type="eomccsdt")

    T3_excitations, T3_amplitudes, pspace_t = get_T3_list(T)
    R3_excitations, R3_amplitudes, pspace_r = get_R3_list(R)
    excitation_count_t = [[T3_excitations["aaa"].shape[0],
                           T3_excitations["aab"].shape[0],
                           T3_excitations["abb"].shape[0],
                           T3_excitations["bbb"].shape[0]]]
    excitation_count_r = [[R3_excitations["aaa"].shape[0],
                           R3_excitations["aab"].shape[0],
                           R3_excitations["abb"].shape[0],
                           R3_excitations["bbb"].shape[0]]]

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T3 contraction", end="")
    T_ex = deepcopy(T)
    dT_ex = ClusterOperator(system, order=3)
    t1 = time.time()
    T_ex, dT_ex = update(T_ex, dT_ex, H, 0.0, False, system)
    print(" (Completed in ", time.time() - t1, "seconds)")

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T3 contraction (Fortran)", end="")
    dT_p = ClusterOperator(system, order=3, p_orders=[3], pspace_sizes=excitation_count)
    T_p = ClusterOperator(system, order=3, p_orders=[3], pspace_sizes=excitation_count)
    T_p.unflatten(
        np.hstack((T.a.flatten(), T.b.flatten(),
                   T.aa.flatten(), T.ab.flatten(), T.bb.flatten(),
                   T3_amplitudes["aaa"], T3_amplitudes["aab"], T3_amplitudes["abb"], T3_amplitudes["bbb"]))
    )
    t1 = time.time()
    T_p, dT_p = update_p(T_p, dT_p, H, 0.0, False, system, T3_excitations, pspace)
    print(" (Completed in ", time.time() - t1, "seconds)")

    print("")
    nua, noa = T.a.shape
    nub, nob = T.b.shape

    flag = True
    err_cum = 0.0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            error = T_p.a[a, i] - T_ex.a[a, i]
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
            error = T_p.b[a, i] - T_ex.b[a, i]
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
                    error = T_p.aa[a, b, i, j] - T_ex.aa[a, b, i, j]
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
                    error = T_p.ab[a, b, i, j] - T_ex.ab[a, b, i, j]
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
                    error = T_p.bb[a, b, i, j] - T_ex.bb[a, b, i, j]
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
        error = T_p.aaa[idet] - T_ex.aaa[a, b, c, i, j, k]
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
        error = T_p.aab[idet] - T_ex.aab[a, b, c, i, j, k]
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
        error = T_p.abb[idet] - T_ex.abb[a, b, c, i, j, k]
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
        error = T_p.bbb[idet] - T_ex.bbb[a, b, c, i, j, k]
        err_cum += abs(error)
        if abs(error) > 1.0e-010:
            flag = False
    if flag:
        print("T3D update passed!", "Cumulative Error = ", err_cum)
    else:
        print("T3D update FAILED!", "Cumulative Error = ", err_cum)
