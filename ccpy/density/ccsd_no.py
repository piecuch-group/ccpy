import numpy as np
from scipy.linalg import eig

# Things to try:
# Maybe the transformation shuld be carried out at the level of Z and V
# We cannot simply transform F and get a new F matrix. To see this, F
# is still diagonal after the transformation when it should not be.

def convert_to_ccsd_no(rdm1, H, system):

    oa = slice(0, system.noccupied_alpha)
    va = slice(system.noccupied_alpha, system.norbitals)
    ob = slice(0, system.noccupied_beta)
    vb = slice(system.noccupied_beta, system.norbitals)

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

    # Remove the G part of the Fock matrix
    G_a = np.zeros((system.norbitals, system.norbitals))
    G_b = np.zeros((system.norbitals, system.norbitals))
    # <p|g|q> = <pi|v|qi> + <pi~|v|qi~>
    G_a[oa, oa] = (
        + np.einsum("piqi->pq", H.aa.oooo)
        + np.einsum("piqi->pq", H.ab.oooo)
    )
    G_a[oa, va] = (
        + np.einsum("piqi->pq", H.aa.oovo)
        + np.einsum("piqi->pq", H.ab.oovo)
    )
    G_a[va, oa] = (
        + np.einsum("piqi->pq", H.aa.vooo)
        + np.einsum("piqi->pq", H.ab.vooo)
    )
    G_a[va, va] = (
        + np.einsum("piqi->pq", H.aa.vovo)
        + np.einsum("piqi->pq", H.ab.vovo)
    )
    # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    G_b[ob, ob] = (
        + np.einsum("piqi->pq", H.bb.oooo)
        + np.einsum("ipiq->pq", H.ab.oooo)
    )
    G_b[ob, vb] = (
        + np.einsum("piqi->pq", H.bb.oovo)
        + np.einsum("ipiq->pq", H.ab.ooov)
    )
    G_b[vb, ob] = (
        + np.einsum("piqi->pq", H.bb.vooo)
        + np.einsum("ipiq->pq", H.ab.ovoo)
    )
    G_b[vb, vb] = (
        + np.einsum("piqi->pq", H.bb.vovo)
        + np.einsum("ipiq->pq", H.ab.ovov)
    )

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
    temp[oa, oa, oa, oa] = H.aa.oooo
    temp[oa, oa, oa, va] = H.aa.ooov
    temp[oa, oa, va, oa] = H.aa.oovo
    temp[oa, va, oa, oa] = H.aa.ovoo
    temp[va, oa, oa, oa] = H.aa.vooo
    temp[oa, oa, va, va] = H.aa.oovv
    temp[va, va, oa, oa] = H.aa.vvoo
    temp[oa, va, va, oa] = H.aa.ovvo
    temp[va, oa, oa, va] = H.aa.voov
    temp[va, oa, va, oa] = H.aa.vovo
    temp[oa, va, oa, va] = H.aa.ovov
    temp[va, va, va, oa] = H.aa.vvvo
    temp[va, va, oa, va] = H.aa.vvov
    temp[va, oa, va, va] = H.aa.vovv
    temp[oa, va, va, va] = H.aa.ovvv
    temp[va, va, va, va] = H.aa.vvvv
    for s in H.aa.slices:
        x1 = Wa[:, slice_table['a'][s[0]]]
        x2 = Wa[:, slice_table['a'][s[1]]]
        x3 = Va[:, slice_table['a'][s[2]]]
        x4 = Va[:, slice_table['a'][s[3]]]
        setattr(H.aa,
                s,
                np.real(np.einsum("pi,qj,ijkl,kr,ls->pqrs", x1.conj().T, x2.conj().T, temp, x3, x4, optimize=True)))

    temp = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    temp[oa, ob, oa, ob] = H.ab.oooo
    temp[oa, ob, oa, vb] = H.ab.ooov
    temp[oa, ob, va, ob] = H.ab.oovo
    temp[oa, vb, oa, ob] = H.ab.ovoo
    temp[va, ob, oa, ob] = H.ab.vooo
    temp[oa, ob, va, vb] = H.ab.oovv
    temp[va, vb, oa, ob] = H.ab.vvoo
    temp[oa, vb, va, ob] = H.ab.ovvo
    temp[va, ob, oa, vb] = H.ab.voov
    temp[va, ob, va, ob] = H.ab.vovo
    temp[oa, vb, oa, vb] = H.ab.ovov
    temp[va, vb, va, ob] = H.ab.vvvo
    temp[va, vb, oa, vb] = H.ab.vvov
    temp[va, ob, va, vb] = H.ab.vovv
    temp[oa, vb, va, vb] = H.ab.ovvv
    temp[va, vb, va, vb] = H.ab.vvvv
    for s in H.ab.slices:
        x1 = Wa[:, slice_table['a'][s[0]]]
        x2 = Wb[:, slice_table['b'][s[1]]]
        x3 = Va[:, slice_table['a'][s[2]]]
        x4 = Vb[:, slice_table['b'][s[3]]]
        setattr(H.ab,
                s,
                np.real(np.einsum("pi,qj,ijkl,kr,ls->pqrs", x1.conj().T, x2.conj().T, temp, x3, x4, optimize=True)))

    temp = np.zeros((system.norbitals, system.norbitals, system.norbitals, system.norbitals))
    temp[ob, ob, ob, ob] = H.bb.oooo
    temp[ob, ob, ob, vb] = H.bb.ooov
    temp[ob, ob, vb, ob] = H.bb.oovo
    temp[ob, vb, ob, ob] = H.bb.ovoo
    temp[vb, ob, ob, ob] = H.bb.vooo
    temp[ob, ob, vb, vb] = H.bb.oovv
    temp[vb, vb, ob, ob] = H.bb.vvoo
    temp[ob, vb, vb, ob] = H.bb.ovvo
    temp[vb, ob, ob, vb] = H.bb.voov
    temp[vb, ob, vb, ob] = H.bb.vovo
    temp[ob, vb, ob, vb] = H.bb.ovov
    temp[vb, vb, vb, ob] = H.bb.vvvo
    temp[vb, vb, ob, vb] = H.bb.vvov
    temp[vb, ob, vb, vb] = H.bb.vovv
    temp[ob, vb, vb, vb] = H.bb.ovvv
    temp[vb, vb, vb, vb] = H.bb.vvvv
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
    temp[oa, oa] = H.a.oo
    temp[oa, va] = H.a.ov
    temp[va, oa] = H.a.vo
    temp[va, va] = H.a.vv
    temp -= G_a
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
    temp[ob, ob] = H.b.oo
    temp[ob, vb] = H.b.ov
    temp[vb, ob] = H.b.vo
    temp[vb, vb] = H.b.vv
    temp -= G_b
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



    Escf = system.nuclear_repulsion
    Escf += np.einsum('ii->', H.a.oo, optimize=True)
    Escf += np.einsum('ii->', H.b.oo, optimize=True)
    Escf -= 0.5 * np.einsum('ijij->', H.aa.oooo, optimize=True)
    Escf -= 0.5 * np.einsum('ijij->', H.bb.oooo, optimize=True)
    Escf -= np.einsum('ijij->', H.ab.oooo, optimize=True)

    system.reference_energy = Escf

    return H, system


