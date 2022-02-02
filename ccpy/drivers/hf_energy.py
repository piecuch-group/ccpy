import numpy as np


def calc_hf_energy(e1int, e2int, system):

    occ_a = slice(0, system.noccupied_alpha + system.nfrozen)
    occ_b = slice(0, system.noccupied_beta + system.nfrozen)

    e1a = np.einsum("ii->", e1int[occ_a, occ_a])
    e1b = np.einsum("ii->", e1int[occ_b, occ_b])
    e2a = 0.5 * (
        np.einsum("ijij->", e2int[occ_a, occ_a, occ_a, occ_a])
        - np.einsum("ijji->", e2int[occ_a, occ_a, occ_a, occ_a])
    )
    e2b = np.einsum("ijij->", e2int[occ_a, occ_b, occ_a, occ_b])
    e2c = 0.5 * (
        np.einsum("ijij->", e2int[occ_b, occ_b, occ_b, occ_b])
        - np.einsum("ijji->", e2int[occ_b, occ_b, occ_b, occ_b])
    )

    hf_energy = e1a + e1b + e2a + e2b + e2c

    return hf_energy


def calc_khf_energy(e1int, e2int, system):
    # Note that any V must have a factor of 1/Nkpts!
    e1a = 0.0
    e1b = 0.0
    e2a = 0.0
    e2b = 0.0
    e2c = 0.0

    # slices
    occ_a = slice(0, system.noccupied_alpha + system.nfrozen)
    occ_b = slice(0, system.noccupied_beta + system.nfrozen)

    e1a = np.einsum("uuii->", e1int[:, :, occ_a, occ_a])
    e1b = np.einsum("uuii->", e1int[:, :, occ_b, occ_b])
    e2a = 0.5 * (
        np.einsum("uvuvijij->", e2int[:, :, :, :, occ_a, occ_a, occ_a, occ_a])
        - np.einsum("uvvuijji->", e2int[:, :, :, :, occ_a, occ_a, occ_a, occ_a])
    )
    e2b = 1.0 * (np.einsum("uvuvijij->", e2int[:, :, :, :, occ_a, occ_b, occ_a, occ_b]))
    e2c = 0.5 * (
        np.einsum("uvuvijij->", e2int[:, :, :, :, occ_b, occ_b, occ_b, occ_b])
        - np.einsum("uvvuijji->", e2int[:, :, :, :, occ_b, occ_b, occ_b, occ_b])
    )

    e1a + e1b + e2a + e2b + e2c

    return np.real(Escf) / system.nkpts
