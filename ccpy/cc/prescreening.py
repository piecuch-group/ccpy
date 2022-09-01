import numpy as np

from ccpy.utilities.pspace import count_excitations_in_pspace

def prescreening_t3a(pspace, system):

    excitation_count = count_excitations_in_pspace(pspace, system)

    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta
    noa = system.nunoccupied_alpha
    nob = system.nunoccupied_beta

    n3a_p = excitation_count['aaa']

    # allocate dictionaries for diagrams and their permutational variants

    # A(c/ab) h1a(ce) * t3a(abeijk)
    idx1_1 = np.zeros((n3a_p, nua + 1), dtype=np.int64)
    idx1_ac = np.zeros((n3a_p, nua + 1), dtype=np.int64)
    idx1_bc = np.zeros((n3a_p, nua + 1), dtype=np.int64)
    # A(c/ab) 1/2 h2a(abef) * t3a(efcijk)
    idx2_1 = np.zeros((n3a_p, nua*(nua - 1)/2 + 1), dtype=np.int64)
    idx2_ac = np.zeros((n3a_p, nua*(nua - 1)/2 + 1), dtype=np.int64)
    idx2_bc = np.zeros((n3a_p, nua*(nua - 1)/2 + 1), dtype=np.int64)

    # f * (no**3 * nu**3) * nu**2

    ct = 0
    for a in range(nua):
        for b in range(a + 1, nua):
            for c in range(b + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        for k in range(j + 1, noa):

                            if pspace['aaa'][a, b, c, i, j, k] != 1: continue

                            # Diagram 1
                            ct1 = 0
                            ct2 = 0
                            ct3 = 0
                            for e in range(nua):
                                if pspace['aaa'][a, b, e, i, j, k] == 1:
                                    idx1_1[ct, ct1] = e
                                    ct1 += 1
                                if pspace['aaa'][c, b, e, i, j, k] == 1:
                                    idx1_ac[ct, ct2] = e
                                    ct2 += 1
                                if pspace['aaa'][a, c, e, i, j, k] == 1:
                                    idx1_bc[ct, ct3] = e
                                    ct3 += 1
                            idx1_1[-1] = ct1
                            idx1_ac[-1] = ct2
                            idx1_bc[-1] = ct3

                            # Diagram 2
                            ct1 = 0
                            ct2 = 0
                            ct3 = 0
                            for e in range(nua):
                                for f in range(e + 1, nua):
                                    if pspace['aaa'][e, f, c, i, j, k] == 1:
                                        idx2_1[ct, ct1] = np.ravel_multi_index((e, f), (nua, nua))
                                        ct1 += 1
                                    if pspace['aaa'][e, f, a, i, j, k] == 1:
                                        idx2_ac[ct, ct2] = np.ravel_multi_index((e, f), (nua, nua))
                                        ct2 += 1
                                    if pspace['aaa'][e, f, b, i, j, k] == 1:
                                        idx2_bc[ct, ct3] = np.ravel_multi_index((e, f), (nua, nua))
                                        ct3 += 1
                            idx1_1[-1] = ct1
                            idx1_ac[-1] = ct2
                            idx1_bc[-1] = ct3


                            # Diagram 3

                            # Diagram 4

                            ct += 1

    return