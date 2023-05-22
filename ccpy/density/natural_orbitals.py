import numpy as np
from scipy.linalg import eig

from ccpy.energy.hf_energy import calc_g_matrix, calc_hf_energy
from ccpy.models.integrals import getHamiltonian
from ccpy.utilities.dumping import dumpIntegralstoPGFiles

# The traditional CC/EOMCC methods are invariant with respect to independent
# occ-occ and virt-virt rotations. They are NOT invariant with respect to
# occ-virt rotations (this, by Thouless Theorem, generates new non-orthogonal references).
#
# Due to the presence of T1, there is a weaker dependence on occ-virt rotations
# of the MOs, but it's not strictly invariant. As a result, to get the same answers,
# you can only apply occ-occ and virt-virt rotations.
#
# You can play with this idea by, say, only applying a virt-virt rotations to clean
# up the HF-based virtual orbitals, which are of low quality anyway, and retain the
# original HF orbitals. This may give you better behaviors?

def transform_to_natorbs(rdm1, H, system, dump_integrals=False, print_diagnostics=False):

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

    rdm1a_matrix = np.concatenate( (np.concatenate( (rdm1.a.oo, rdm1.a.ov * 0.0), axis=1),
                                    np.concatenate( (rdm1.a.vo * 0.0, rdm1.a.vv), axis=1)), axis=0)
    rdm1b_matrix = np.concatenate( (np.concatenate( (rdm1.b.oo, rdm1.b.ov * 0.0), axis=1),
                                    np.concatenate( (rdm1.b.vo * 0.0, rdm1.b.vv), axis=1)), axis=0)
    rdm_matrix = rdm1a_matrix + rdm1b_matrix

    # symmetry block-diagonalize
    nocc_vals = np.zeros(system.norbitals)
    L = np.zeros((system.norbitals, system.norbitals))
    R = np.zeros((system.norbitals, system.norbitals))

    pg_order = len(system.point_group_irrep_to_number)
    idx = [[] for i in range(pg_order)]
    for p in range(system.norbitals):
        irrep_number = system.point_group_irrep_to_number[system.orbital_symmetries[p]]
        idx[irrep_number].append(p)

    for sym in range(pg_order):
        n = len(idx[sym])

        rdm_sym_block = np.zeros((n, n))
        for p in range(n):
            for q in range(n):
                rdm_sym_block[p, q] = rdm_matrix[idx[sym][p], idx[sym][q]]
        nval, left, right = eig(rdm_sym_block, left=True, right=True)

        for p in range(n):
            nocc_vals[idx[sym][p]] = np.real(nval[p])
            for q in range(n):
                L[idx[sym][q], idx[sym][p]] = left[q, p]
                R[idx[sym][q], idx[sym][p]] = right[q, p]
    idx = np.flip(np.argsort(nocc_vals))
    nocc_vals = nocc_vals[idx]
    L = L[:, idx]
    R = R[:, idx]

    print("   CCSD Natural Orbitals:")
    print("   Orbital        Occupation       Importance Score")
    print("   ------------------------------------------------")
    for i in range(system.norbitals):
        score = 100.0 * ( 2.0 * nocc_vals[i] - nocc_vals[i]**2 )
        print("     {:>2}          {:>10f}          {:>10f}".format(i + 1, nocc_vals[i], score))

    # Biorthogonalize the left and right NO vectors
    for i in range(system.norbitals):
        L[:, i] /= abs(np.dot(L[:, i].conj(), R[:, i]))
    LR = L.conj().T @ R

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
    e2int_no = np.einsum("ip,jq,ijkl,kr,ls->pqrs", L.conj(), L.conj(), temp, R, R, optimize=True)
    e2int_no = np.pad(e2int_no, ((system.nfrozen, 0), (system.nfrozen, 0), (system.nfrozen, 0), (system.nfrozen, 0)))

    # transform onebody integrals
    temp = np.zeros((system.norbitals, system.norbitals))
    temp[slice_table["a"]["o"], slice_table["a"]["o"]] = H.a.oo - G.a.oo
    temp[slice_table["a"]["o"], slice_table["a"]["v"]] = H.a.ov - G.a.ov
    temp[slice_table["a"]["v"], slice_table["a"]["o"]] = H.a.vo - G.a.vo
    temp[slice_table["a"]["v"], slice_table["a"]["v"]] = H.a.vv - G.a.vv
    e1int_no = np.einsum("ip,ij,jq->pq", L.conj(), temp, R)
    e1int_no = np.pad(e1int_no, ((system.nfrozen, 0), (system.nfrozen, 0)))

    if dump_integrals:
        dumpIntegralstoPGFiles(e1int_no, e2int_no, system)

    system.reference_energy = (
                                calc_hf_energy(e1int_no, e2int_no, system)
                                + system.nuclear_repulsion
                                + system.frozen_energy
    )
    H = getHamiltonian(e1int_no, e2int_no, system, normal_ordered=True)

    if print_diagnostics:
        print("   Diagnostics:")
        print("   -------------")
        print("   Biorthogonality = ", np.linalg.norm(LR - np.eye(system.norbitals)))
        print("   |imag(R)| = ", np.linalg.norm(np.imag(R)))
        print("   |imag(L)| = ", np.linalg.norm(np.imag(L)))
        print("   |imag(e1int)| = ", np.linalg.norm(np.imag(e1int_no)))
        print("   |imag(e2int)| = ", np.linalg.norm(np.imag(e2int_no.flatten())))
        print("   -------------\n")

    return H, system
