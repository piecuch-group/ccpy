import numpy as np
from pyscf import ao2mo

def get_integrals_from_pyscf(meanfield, nfrozen):

    molecule = meanfield.mol
    mo_coeff = meanfield.mo_coeff
    norbitals = mo_coeff.shape[1]

    kinetic_aoints = molecule.intor_symmetric("int1e_kin")
    nuclear_aoints = molecule.intor_symmetric("int1e_nuc")
    e1int = np.einsum("pi,pq,qj->ij", mo_coeff, kinetic_aoints + nuclear_aoints, mo_coeff)
    e2int = np.transpose(
        np.reshape(ao2mo.kernel(molecule, mo_coeff, compact=False), 4 * (norbitals,)),
        (0, 2, 1, 3),
    )

    z, g = spatial_to_spinorb(e1int, e2int)
    g -= np.transpose(g, (0, 1, 3, 2))

    return z, g, molecule.energy_nuc(), meanfield.mo_energy,

def spatial_to_spinorb(e1int, e2int):

    n = e1int.shape[0]
    z = np.zeros((2*n, 2*n))
    g = np.zeros((2*n, 2*n, 2*n, 2*n))

    for i in range(2*n):
        for j in range(2*n):
            if i % 2 == j % 2:
                i0 = int(np.floor(i/2))
                j0 = int(np.floor(j/2))
                z[i, j] = e1int[i0, j0]
    for i in range(2*n):
        for j in range(2*n):
            for k in range(2*n):
                for l in range(2*n):
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        i0 = int(np.floor(i/2))
                        j0 = int(np.floor(j/2))
                        k0 = int(np.floor(k/2))
                        l0 = int(np.floor(l/2))
                        g[i, j, k, l] = e2int[i0, j0, k0, l0]
    return z, g


def get_fock(z, g, o, v):

    f = z + np.einsum("piqi->pq", g[:, o, :, o])

    return f

