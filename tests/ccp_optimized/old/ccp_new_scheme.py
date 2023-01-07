import numpy as np

def main(civec_file):

    from ccpy.interfaces.gamess_tools import load_from_gamess
    from ccpy.utilities.pspace import get_pspace_from_cipsi, count_excitations_in_pspace

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.utilities.dumping import save_T_vector, load_T_vector

    run_ccp = False

    system, H0 = load_from_gamess(
            "f2-2re-for-karthik.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=2,
    )

    system.print_info()

    print("   Using P space file: ", civec_file)
    pspace, excitation_count = get_pspace_from_cipsi(civec_file, system, nexcit=3)
    print("   P space composition:")
    print("   ----------------------")
    count_excits = count_excitations_in_pspace(pspace, system)
    print("   Number of aaa = ", count_excits[0]["aaa"])
    print("   Number of aab = ", count_excits[0]["aab"])
    print("   Number of abb = ", count_excits[0]["abb"])
    print("   Number of bbb = ", count_excits[0]["bbb"])

    if run_ccp:
        calculation = Calculation(
            order=3,
            calculation_type="ccsdt_p",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            low_memory=False 
        )   

        T, total_energy, is_converged = cc_driver(calculation, system, H0, pspace=pspace)

        save_T_vector(T)
    else:
        T = load_T_vector(3, system)

    H = build_hbar_ccsd(T, H0)

    # traditional way of calculating
    triples_res = -1.0 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aaa, optimize=True)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)

    # new way of calculating
    n3a_p = count_excits[0]["aaa"]
    list_aaa = []
    t3a_p = np.zeros(n3a_p)
    ind3A = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):
                            if pspace[0]["aaa"][a, b, c, i, j, k] != 1:
                                continue
                            ind3A[a, b, c, i, j, k] = int(ct)
                            t3a_p[ct] = T.aaa[a, b, c, i, j, k]
                            list_aaa.append(np.array([a, b, c, i, j, k]))
                            ct += 1

    Hmat = np.zeros((n3a_p, n3a_p))
    new_triples_res = np.zeros_like(T.aaa)
    for idet in range(len(list_aaa)):

        a, b, c, i, j, k = list_aaa[idet]

        for m in range(system.noccupied_alpha):

            if pspace[0]["aaa"][a, b, c, m, j, k] != 0: 
                jdet = ind3A[a, b, c, m, j, k]
                Hmat[idet, jdet] += -1.0 * H.a.oo[i, m]

                jdet = ind3A[a, b, c, j, m, k]
                Hmat[idet, jdet] += 1.0 * H.a.oo[j, m]

                jdet = ind3A[a, b, c, k, j, m]
                Hmat[idet, jdet] += 1.0 * H.a.oo[k, m]

        new_triples_res[a, b, c, i, j, k] = np.dot(Hmat[idet, :], t3a_p)

    #for idx in range(len(list_aaa)):
    #    a, b, c, i, j, k = list_aaa[idx]
    #    t3val = T.aaa[a, b, c, i, j, k]
    #
    #    hmat = (
    #                    -1.0 * H.a.oo[i, :]
    #                    +1.0 * H.a.oo[j, :]
    #                    +1.0 * H.a.oo[k, :]
    #            )
    #
    #    new_triples_res[a, b, c, :, j, k]  += hmat * t3val

            #dgm1 = -(c1==c2)*(b1==b2)*(a1==a2)*(\
            #        (j1==j2)*(k1==k2)*H1A['oo'][i2,i1]\
            #        -(i1==j2)*(k1==k2)*H1A['oo'][i2,j1]\
            #        -(j1==j2)*(i1==k2)*H1A['oo'][i2,k1]\
            #        -(j1==i2)*(k1==k2)*H1A['oo'][j2,i1]\
            #        -(j1==j2)*(k1==i2)*H1A['oo'][k2,i1]\
            #        +(i1==i2)*(k1==k2)*H1A['oo'][j2,j1]\
            #        +(i1==j2)*(k1==i2)*H1A['oo'][k2,j1]\
            #        +(j1==i2)*(i1==k2)*H1A['oo'][j2,k1]\
            #        +(j1==j2)*(i1==i2)*H1A['oo'][k2,k1]\
            #        -(k1==j2)*(j1==k2)*H1A['oo'][i2,i1]\
            #        +(i1==j2)*(j1==k2)*H1A['oo'][i2,k1]\
            #        +(k1==j2)*(i1==k2)*H1A['oo'][i2,j1]\
            #        +(k1==i2)*(j1==k2)*H1A['oo'][j2,i1]\
            #        +(k1==j2)*(j1==i2)*H1A['oo'][k2,i1]\
            #        -(i1==i2)*(j1==k2)*H1A['oo'][j2,k1]\
            #        -(i1==j2)*(j1==i2)*H1A['oo'][k2,k1]\
            #        -(k1==i2)*(i1==k2)*H1A['oo'][j2,j1]\
            #        -(k1==j2)*(i1==i2)*H1A['oo'][k2,j1])

        #new_triples_res[a, b, c, i, :, k]  += 1.0 * H.a.oo[j, :] * t3val
        #new_triples_res[a, b, c, i, j, :]  += 1.0 * H.a.oo[k, :] * t3val

        #new_triples_res[a, b, c, j, :, k]  += 1.0 * H.a.oo[i, :] * t3val
        #new_triples_res[a, b, c, i, :, k]  -= 1.0 * H.a.oo[j, :] * t3val
        #new_triples_res[a, b, c, j, :, i]  += 1.0 * H.a.oo[k, :] * t3val

        #new_triples_res[a, b, c, j, k, :]  += 1.0 * H.a.oo[i, :] * t3val
        #new_triples_res[a, b, c, i, k, :]  -= 1.0 * H.a.oo[j, :] * t3val
        #new_triples_res[a, b, c, j, i, :]  += 1.0 * H.a.oo[k, :] * t3val

    #new_triples_res -= np.transpose(new_triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    #new_triples_res -= np.transpose(new_triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(new_triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    #new_triples_res -= np.transpose(new_triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    #new_triples_res -= np.transpose(new_triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(new_triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):
                            if pspace[0]["aaa"][a, b, c, i, j, k] != 1:
                                new_triples_res[a, b, c, i, j, k] = 0.0

    error = 0.0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):
                            error += triples_res[a, b, c, i, j, k] - new_triples_res[a, b, c, i, j, k]

    print("error = ", error)
    
if __name__ == "__main__":

    civec_file = "civecs-5000.dat"
    main(civec_file)



