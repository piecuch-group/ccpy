import numpy as np
from scipy.io import FortranFile
from numba import njit

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import eccc_driver
from ccpy.drivers.cc_energy import get_ci_energy, get_cc_energy

from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analysis

@njit
def check_error_unique_aaaa(t4a, t4a_exact):

    nua = t4a.shape[0]
    noa = t4a.shape[4]

    total_error = 0.0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for d in range(c + 1, nua):
                    for i in range(noa):
                        for j in range(i + 1, noa):
                            for k in range(j + 1, noa):
                                for l in range(k + 1, noa):
                                    total_error += t4a_exact[a, b, c, d, i, j, k, l] - t4a[a, b, c, d, i, j, k, l]

    print("Total error in aaaa = ", total_error)
    return 

@njit
def check_error_unique_aaab(t4b, t4b_exact):

    nua = t4b.shape[0]
    nub = t4b.shape[3]
    noa = t4b.shape[4]
    nob = t4b.shape[7]

    total_error = 0.0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for d in range(nub):
                    for i in range(noa):
                        for j in range(i + 1, noa):
                            for k in range(j + 1, noa):
                                for l in range(nob):
                                    total_error += t4b_exact[a, b, c, d, i, j, k, l] - t4b[a, b, c, d, i, j, k, l]

    print("Total error in aaab = ", total_error)
    return 

@njit
def check_error_unique_aabb(t4c, t4c_exact):

    nua = t4c.shape[0]
    nub = t4c.shape[3]
    noa = t4c.shape[4]
    nob = t4c.shape[7]

    total_error = 0.0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(nub):
                for d in range(c + 1, nub):
                    for i in range(noa):
                        for j in range(i + 1, noa):
                            for k in range(nob):
                                for l in range(k + 1, nob):
                                    total_error += t4c_exact[a, b, c, d, i, j, k, l] - t4c[a, b, c, d, i, j, k, l]

    print("Total error in aabb = ", total_error)
    return 

@njit
def check_error_unique_abbb(t4d, t4d_exact):

    nua = t4d.shape[0]
    nub = t4d.shape[3]
    noa = t4d.shape[4]
    nob = t4d.shape[7]

    total_error = 0.0
    for a in range(nua):
        for b in range(nub):
            for c in range(b + 1, nub):
                for d in range(c + 1, nub):
                    for i in range(noa):
                        for j in range(nob):
                            for k in range(j + 1, nob):
                                for l in range(k + 1, nob):
                                    total_error += t4d_exact[a, b, c, d, i, j, k, l] - t4d[a, b, c, d, i, j, k, l]

    print("Total error in abbb = ", total_error)
    return 

@njit
def check_error_unique_bbbb(t4e, t4e_exact):

    nub = t4e.shape[0]
    nob = t4e.shape[4]

    total_error = 0.0
    for a in range(nub):
        for b in range(a + 1, nub):
            for c in range(b + 1, nub):
                for d in range(c + 1, nub):
                    for i in range(nob):
                        for j in range(i + 1, nob):
                            for k in range(j + 1, nob):
                                for l in range(k + 1, nob):
                                    total_error += t4e_exact[a, b, c, d, i, j, k, l] - t4e[a, b, c, d, i, j, k, l]

    print("Total error in bbbb = ", total_error)
    return 




if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    C, C4_excits, C4_amps, excitation_count = parse_ci_wavefunction("ndet_50000/civecs.dat", system)

    print("Excitation Content")
    print("-------------------")
    print("Number of singles = ", excitation_count['a'] + excitation_count['b'])
    print("Number of doubles = ", excitation_count['aa'] + excitation_count['ab'] + excitation_count['bb'])
    print("Number of triples = ", excitation_count['aaa'] + excitation_count['aab'] + excitation_count['abb'] + excitation_count['bbb'])
    print("Number of quadruples = ", excitation_count['aaaa'] + excitation_count['aaab'] + excitation_count['aabb'] + excitation_count['abbb'] + excitation_count['bbbb'])
    print("")

    Ecorr_c = get_ci_energy(C, H)

    print("External correction energy = ", Ecorr_c)

    T_ext = cluster_analysis(C, C4_excits, C4_amps, system)

    Ecorr_t = get_cc_energy(T_ext, H)
    print("T vector energy = ", Ecorr_t)

    assert(abs(Ecorr_t - Ecorr_c) < 1.0e-07)


    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/T_vecs.moe", "r") as f:
        first_line_reals = f.read_reals()
        t1_to_t3 = f.read_reals(dtype=np.float64)

    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    n1a = noa * nua
    n1b = nob * nub
    n2a = noa**2 * nua**2
    n2b = noa*nua * nob*nub
    n2c = nob**2 * nub**2
    n3a = noa**3 * nua**3
    n3b = noa**2 * nua**2 * nob * nub
    n3c = noa * nua * nob**2 * nub**2
    n3d = nob**3 * nub**3

    t1a = t1_to_t3[:n1a].reshape((noa, nua)).transpose((1, 0))
    t1b = t1_to_t3[n1a:n1a+n1b].reshape((nob, nub)).transpose((1, 0))
    t2a = t1_to_t3[n1a+n1b:n1a+n1b+n2a].reshape((noa,noa,nua,nua)).transpose((2, 3, 0, 1))
    t2b = t1_to_t3[n1a+n1b+n2a:n1a+n1b+n2a+n2b].reshape((noa,nob,nua,nub)).transpose((2, 3, 0, 1))
    t2c = t1_to_t3[n1a+n1b+n2a+n2b:n1a+n1b+n2a+n2b+n2c].reshape((nob,nob,nub,nub)).transpose((2, 3, 0, 1))
    t3a = t1_to_t3[n1a+n1b+n2a+n2b+n2c:n1a+n1b+n2a+n2b+n2c+n3a].reshape((noa,noa,noa,nua,nua,nua)).transpose((3, 4, 5, 0, 1, 2))
    t3b = t1_to_t3[n1a+n1b+n2a+n2b+n2c+n3a:n1a+n1b+n2a+n2b+n2c+n3a+n3b].reshape((noa,noa,nob,nua,nua,nub)).transpose((3, 4, 5, 0, 1, 2))
    t3c = t1_to_t3[n1a+n1b+n2a+n2b+n2c+n3a+n3b:n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c].reshape((noa,nob,nob,nua,nub,nub)).transpose((3, 4, 5, 0, 1, 2))
    t3d = t1_to_t3[n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c:n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c+n3d].reshape((nob,nob,nob,nub,nub,nub)).transpose((3, 4, 5, 0, 1, 2))

    t_vec = {'a' : t1a, 'b' : t1b, 'aa' : t2a, 'ab' : t2b, 'bb' : t2c, 'aaa' : t3a, 'aab' : t3b, 'abb' : t3c, 'bbb' : t3d}
    for key, value in t_vec.items():
        print("Error in ", key, " = ", np.linalg.norm(value.flatten() - getattr(T_ext, key).flatten()))

    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/t4a", "r") as f:
        t4a = f.read_reals(dtype=np.float64).reshape((noa, noa, noa, noa, nua, nua, nua, nua)).transpose((4, 5, 6, 7, 0, 1, 2, 3))
    #print("Error in aaaa = ", np.linalg.norm(T_ext.aaaa.flatten() - t4a.flatten()))
    check_error_unique_aaaa(T_ext.aaaa, t4a)

    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/t4b", "r") as f:
        t4b = f.read_reals(dtype=np.float64).reshape((noa, noa, noa, nob, nua, nua, nua, nub)).transpose((4, 5, 6, 7, 0, 1, 2, 3))
    #print("Error in aaab = ", np.linalg.norm(T_ext.aaab.flatten() - t4b.flatten()))
    check_error_unique_aaab(T_ext.aaab, t4b)

    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/t4c", "r") as f:
        t4c = f.read_reals(dtype=np.float64).reshape((noa, noa, nob, nob, nua, nua, nub, nub)).transpose((4, 5, 6, 7, 0, 1, 2, 3))
    #print("Error in aabb = ", np.linalg.norm(T_ext.aabb.flatten() - t4c.flatten()))
    check_error_unique_aabb(T_ext.aabb, t4c)

    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/t4d", "r") as f:
        t4d = f.read_reals(dtype=np.float64).reshape((noa, nob, nob, nob, nua, nub, nub, nub)).transpose((4, 5, 6, 7, 0, 1, 2, 3))
    #print("Error in abbb = ", np.linalg.norm(T_ext.abbb.flatten() - t4d.flatten()))
    setattr(T_ext, "abbb", np.transpose(T_ext.aaab, (3, 2, 1, 0, 7, 6, 5, 4)))
    check_error_unique_abbb(T_ext.abbb, t4d)

    with FortranFile("/scratch/gururang/test_cluster_analysis/ndet_50000/t4e", "r") as f:
        t4e = f.read_reals(dtype=np.float64).reshape((nob, nob, nob, nob, nub, nub, nub, nub)).transpose((4, 5, 6, 7, 0, 1, 2, 3))
    #print("Error in bbbb = ", np.linalg.norm(T_ext.bbbb.flatten() - t4e.flatten()))
    check_error_unique_bbbb(T_ext.bbbb, t4e)

    calculation = Calculation(calculation_type="eccc2_slow")
    T, total_energy, converged = eccc_driver(calculation, system, H, T_ext)
