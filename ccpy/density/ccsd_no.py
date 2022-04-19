import numpy as np
from scipy.linalg import eig

from ccpy.drivers.hf_energy import calc_g_matrix
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

    nocc_vals_a, Wa, Va = eig(rdm1a_matrix, left=True, right=True)
    nocc_vals_a = np.real(nocc_vals_a)
    idx = np.flip(np.argsort(nocc_vals_a))
    nocc_vals_a = nocc_vals_a[idx]
    Wa = Wa[:, idx]
    Va = Va[:, idx]

    rdm1b_matrix = np.concatenate( (np.concatenate( (rdm1.b.oo, rdm1.b.ov), axis=1),
                                    np.concatenate( (rdm1.b.vo, rdm1.b.vv), axis=1)), axis=0)

    nocc_vals_b, Wb, Vb = eig(rdm1b_matrix, left=True, right=True)
    nocc_vals_b = np.real(nocc_vals_b)
    idx = np.flip(np.argsort(nocc_vals_b))
    nocc_vals_b = nocc_vals_b[idx]
    Wb = Wb[:, idx]
    Vb = Vb[:, idx]

    print("   Occupation numbers")
    print("   orbital       occupation #")
    print("   ----------------------------")
    for i in range(system.norbitals):
        print("     {}          {}".format(i+1, nocc_vals_a[i] + nocc_vals_b[i]))


    # Transform twobody integrals
    temp = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    temp[slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["o"]] = H.aa.oooo
    temp[slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["v"]] = H.aa.ooov
    temp[slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["o"]] = H.aa.oovo
    temp[slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["o"]] = H.aa.ovoo
    temp[slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["o"]] = H.aa.vooo
    temp[slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["v"]] = H.aa.oovv
    temp[slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["o"]] = H.aa.vvoo
    temp[slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["o"]] = H.aa.ovvo
    temp[slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["o"], slice_table["a"]["v"]] = H.aa.voov
    temp[slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["o"]] = H.aa.vovo
    temp[slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["v"]] = H.aa.ovov
    temp[slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["o"]] = H.aa.vvvo
    temp[slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["v"]] = H.aa.vvov
    temp[slice_table["a"]["v"], slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["v"]] = H.aa.vovv
    temp[slice_table["a"]["o"], slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["v"]] = H.aa.ovvv
    temp[slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["v"], slice_table["a"]["v"]] = H.aa.vvvv
    for s in H.aa.slices:
        x1 = Wa[:, slice_table['a'][s[0]]]
        x2 = Wa[:, slice_table['a'][s[1]]]
        x3 = Va[:, slice_table['a'][s[2]]]
        x4 = Va[:, slice_table['a'][s[3]]]
        setattr(H.aa,
                s,
                np.real(np.einsum("pi,qj,ijkl,kr,ls->pqrs", x1.conj().T, x2.conj().T, temp, x3, x4, optimize=True)))

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
    for s in H.ab.slices:
        x1 = Wa[:, slice_table['a'][s[0]]]
        x2 = Wb[:, slice_table['b'][s[1]]]
        x3 = Va[:, slice_table['a'][s[2]]]
        x4 = Vb[:, slice_table['b'][s[3]]]
        setattr(H.ab,
                s,
                np.real(np.einsum("pi,qj,ijkl,kr,ls->pqrs", x1.conj().T, x2.conj().T, temp, x3, x4, optimize=True)))

    temp = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    temp[slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["o"]] = H.bb.oooo
    temp[slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["v"]] = H.bb.ooov
    temp[slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["o"]] = H.bb.oovo
    temp[slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["o"]] = H.bb.ovoo
    temp[slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["o"]] = H.bb.vooo
    temp[slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["v"]] = H.bb.oovv
    temp[slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["o"]] = H.bb.vvoo
    temp[slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["o"]] = H.bb.ovvo
    temp[slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["o"], slice_table["b"]["v"]] = H.bb.voov
    temp[slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["o"]] = H.bb.vovo
    temp[slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["v"]] = H.bb.ovov
    temp[slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["o"]] = H.bb.vvvo
    temp[slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["v"]] = H.bb.vvov
    temp[slice_table["b"]["v"], slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["v"]] = H.bb.vovv
    temp[slice_table["b"]["o"], slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["v"]] = H.bb.ovvv
    temp[slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["v"], slice_table["b"]["v"]] = H.bb.vvvv
    for s in H.bb.slices:
        x1 = Wb[:, slice_table['b'][s[0]]]
        x2 = Wb[:, slice_table['b'][s[1]]]
        x3 = Vb[:, slice_table['b'][s[2]]]
        x4 = Vb[:, slice_table['b'][s[3]]]
        setattr(H.bb,
                s,
                np.real(np.einsum("pi,qj,ijkl,kr,ls->pqrs", x1.conj().T, x2.conj().T, temp, x3, x4, optimize=True)))

    # transform onebody integrals
    temp = np.zeros((system.norbitals, system.norbitals))
    temp[slice_table["a"]["o"], slice_table["a"]["o"]] = H.a.oo - G.a.oo
    temp[slice_table["a"]["o"], slice_table["a"]["v"]] = H.a.ov - G.a.ov
    temp[slice_table["a"]["v"], slice_table["a"]["o"]] = H.a.vo - G.a.vo
    temp[slice_table["a"]["v"], slice_table["a"]["v"]] = H.a.vv - G.a.vv
    #temp -= G_a # subtract off G to get Z
    for s in H.a.slices:
        x1 = Wa[:, slice_table['a'][s[0]]]
        x2 = Va[:, slice_table['a'][s[1]]]

        setattr(H.a,
                s,
                np.real(
                    np.einsum("pi,ij,jq->pq", x1.conj().T, temp, x2)
                    + np.einsum("piqi->pq", getattr(H.aa, s[0] + 'o' + s[1] + 'o'))
                    + np.einsum("piqi->pq", getattr(H.ab, s[0] + 'o' + s[1] + 'o'))
                )
                )

    temp = np.zeros((system.norbitals, system.norbitals))
    temp[slice_table["b"]["o"], slice_table["b"]["o"]] = H.b.oo - G.b.oo
    temp[slice_table["b"]["o"], slice_table["b"]["v"]] = H.b.ov - G.b.ov
    temp[slice_table["b"]["v"], slice_table["b"]["o"]] = H.b.vo - G.b.vo
    temp[slice_table["b"]["v"], slice_table["b"]["v"]] = H.b.vv - G.b.vv
    #temp -= G_b # subtract off G to get Z
    for s in H.b.slices:
        x1 = Wb[:, slice_table['b'][s[0]]]
        x2 = Vb[:, slice_table['b'][s[1]]]

        setattr(H.b,
                s,
                np.real(
                    np.einsum("pi,ij,jq->pq", x1.conj().T, temp, x2)
                    + np.einsum("piqi->pq", getattr(H.bb, s[0] + 'o' + s[1] + 'o'))
                    + np.einsum("ipiq->pq", getattr(H.ab, 'o' + s[0] + 'o' + s[1]))
                )
                )


    # # save Fa integrals as e1int
    # e1int_no = np.zeros((system.norbitals, system.norbitals))
    # e1int_no[slice_table["a"]["o"], slice_table["a"]["o"]] = H.a.oo
    # e1int_no[slice_table["a"]["o"], slice_table["a"]["v"]] = H.a.ov
    # e1int_no[slice_table["a"]["v"], slice_table["a"]["o"]] = H.a.vo
    # e1int_no[slice_table["a"]["v"], slice_table["a"]["v"]] = H.a.vv

    # # save Vab integrals as e2int
    # e2int_no = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.oooo
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.ooov
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.oovo
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.ovoo
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.vooo
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.oovv
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["o"]] = H.ab.vvoo
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.ovvo
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.voov
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.vovo
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.ovov
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["o"]] = H.ab.vvvo
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["o"], slice_table["b"]["v"]] = H.ab.vvov
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["o"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.vovv
    # e2int_no[slice_table["a"]["o"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.ovvv
    # e2int_no[slice_table["a"]["v"], slice_table["b"]["v"], slice_table["a"]["v"], slice_table["b"]["v"]] = H.ab.vvvv

    # dumpIntegralstoPGFiles(e1int_no, e2int_no, system)

    # Escf = system.nuclear_repulsion
    # Escf += np.einsum('ii,i->', H.a.oo, nocc_vals_a[:system.noccupied_alpha])
    # Escf += np.einsum('aa,a->', H.a.vv, nocc_vals_a[system.noccupied_alpha:])
    # Escf += np.einsum('ii,i->', H.b.oo, nocc_vals_b[:system.noccupied_beta])
    # Escf += np.einsum('aa,a->', H.b.vv, nocc_vals_b[system.noccupied_beta:])
    #
    # Escf -= 0.5 * np.einsum('ijij,i,j->', H.aa.oooo, nocc_vals_a[:system.noccupied_alpha], nocc_vals_a[:system.noccupied_alpha])
    # Escf -= np.einsum('iaia,i,a->', H.aa.ovov, nocc_vals_a[:system.noccupied_alpha], nocc_vals_a[system.noccupied_alpha:])
    # Escf -= 0.5 * np.einsum('abab,a,b->', H.aa.vvvv, nocc_vals_a[system.noccupied_alpha:], nocc_vals_a[system.noccupied_alpha:])
    #
    # Escf -= 0.5 * np.einsum('ijij,i,j->', H.bb.oooo, nocc_vals_b[:system.noccupied_beta], nocc_vals_b[:system.noccupied_beta])
    # Escf -= np.einsum('iaia,i,a->', H.bb.ovov, nocc_vals_b[:system.noccupied_beta], nocc_vals_b[system.noccupied_beta:])
    # Escf -= 0.5 * np.einsum('abab,a,b->', H.bb.vvvv, nocc_vals_b[system.noccupied_beta:], nocc_vals_b[system.noccupied_beta:])
    #
    # Escf -= np.einsum('ijij,i,j->', H.ab.oooo, nocc_vals_a[:system.noccupied_alpha], nocc_vals_b[:system.noccupied_beta])
    # Escf -= np.einsum('iaia,i,a->', H.ab.ovov, nocc_vals_a[:system.noccupied_alpha], nocc_vals_b[system.noccupied_beta:])
    # Escf -= np.einsum('aiai,a,i->', H.ab.vovo, nocc_vals_a[system.noccupied_alpha:], nocc_vals_b[:system.noccupied_beta])
    # Escf -= np.einsum('abab,a,b->', H.ab.vvvv, nocc_vals_a[system.noccupied_alpha:], nocc_vals_b[system.noccupied_beta:])


    Escf = system.nuclear_repulsion
    Escf += np.einsum('ii->', H.a.oo, optimize=True)
    Escf += np.einsum('ii->', H.b.oo, optimize=True)
    Escf -= 0.5 * np.einsum('ijij->', H.aa.oooo, optimize=True)
    Escf -= 0.5 * np.einsum('ijij->', H.bb.oooo, optimize=True)
    Escf -= np.einsum('ijij->', H.ab.oooo, optimize=True)

    system.reference_energy = Escf

    return H, system


