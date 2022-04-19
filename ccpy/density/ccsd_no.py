import numpy as np
from scipy.linalg import eig

from ccpy.drivers.hf_energy import calc_g_matrix, calc_hf_energy
from ccpy.models.integrals import getHamiltonian
from ccpy.utilities.dumping import dumpIntegralstoPGFiles


def convert_to_ccsd_no(rdm1, H, system):

    slice_table = {
        "a": {
            "o": slice(0, system.noccupied_alpha),
            "v": slice(system.noccupied_alpha, system.norbitals),
        },
        "b": {
            "o": slice(0, system.noccupied_beta),
            "v": slice(system.noccupied_beta, system.norbitals),
        },
    }

    # Compute the HF-based G part of the Fock matrix
    G = calc_g_matrix(H, system)

    rdm1a_matrix = np.concatenate( (np.concatenate( (rdm1.a.oo, rdm1.a.ov), axis=1),
                                    np.concatenate( (rdm1.a.vo, rdm1.a.vv), axis=1)), axis=0)
    rdm1b_matrix = np.concatenate( (np.concatenate( (rdm1.b.oo, rdm1.b.ov), axis=1),
                                    np.concatenate( (rdm1.b.vo, rdm1.b.vv), axis=1)), axis=0)

    nocc_vals, V_left, V_right = eig(rdm1a_matrix + rdm1b_matrix, left=True, right=True)
    nocc_vals = np.real(nocc_vals)
    idx = np.flip(np.argsort(nocc_vals))
    nocc_vals = nocc_vals[idx]
    V_left = V_left[:, idx]
    V_right = V_right[:, idx]

    print("   Occupation numbers")
    print("   orbital       occupation #")
    print("   ----------------------------")
    for i in range(system.norbitals):
        print("     {}          {}".format(i + 1, nocc_vals[i]))

    for i in range(system.norbitals):
        V_left[:, i] /= np.dot(V_left[:, i].conj(), V_right[:, i])
    LR = V_left.conj().T @ V_right
    print("   Biorthogonality = ", np.linalg.norm(LR - np.eye(system.norbitals)))

    # Transform twobody integrals
    temp = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    temp[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.oooo
    temp[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.ooov
    temp[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.oovo
    temp[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.ovoo
    temp[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.vooo
    temp[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.oovv
    temp[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.vvoo
    temp[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.ovvo
    temp[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.voov
    temp[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.vovo
    temp[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.ovov
    temp[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.vvvo
    temp[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.vvov
    temp[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.vovv
    temp[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.ovvv
    temp[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.vvvv
    e2int_no = np.einsum("ip,jq,ijkl,kr,ls->pqrs", V_left.conj(), V_left.conj(), temp, V_right, V_right, optimize=True)

    # transform onebody integrals
    temp = np.zeros((system.norbitals, system.norbitals))
    temp[slice_table["a"]["o"], slice_table["a"]["o"]] = H.a.oo - G.a.oo
    temp[slice_table["a"]["o"], slice_table["a"]["v"]] = H.a.ov - G.a.ov
    temp[slice_table["a"]["v"], slice_table["a"]["o"]] = H.a.vo - G.a.vo
    temp[slice_table["a"]["v"], slice_table["a"]["v"]] = H.a.vv - G.a.vv
    e1int_no = np.einsum("ip,ij,jq->pq", V_left.conj(), temp, V_right)

    #dumpIntegralstoPGFiles(e1int_no, e2int_no, system)

    system.reference_energy = calc_hf_energy(e1int_no, e2int_no, system) + system.nuclear_repulsion
    H = getHamiltonian(e1int_no, e2int_no, system, normal_ordered=True)

    return H, system
