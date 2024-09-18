import numpy as np

def calc_g_matrix(H, system):

    from ccpy.models.integrals import Integral

    G = Integral.from_empty(system, 1, data_type=H.a.oo.dtype, use_none=True)

    # <p|g|q> = <pi|v|qi> + <pi~|v|qi~>
    G.a.oo = (
        + np.einsum("piqi->pq", H.aa.oooo)
        + np.einsum("piqi->pq", H.ab.oooo)
    )
    G.a.ov = (
        + np.einsum("piqi->pq", H.aa.oovo)
        + np.einsum("piqi->pq", H.ab.oovo)
    )
    G.a.vo = (
        + np.einsum("piqi->pq", H.aa.vooo)
        + np.einsum("piqi->pq", H.ab.vooo)
    )
    G.a.vv = (
        + np.einsum("piqi->pq", H.aa.vovo)
        + np.einsum("piqi->pq", H.ab.vovo)
    )
    # <p~|g|q~> = <p~i~|v|q~i~> + <ip~|v|iq~>
    G.b.oo = (
        + np.einsum("piqi->pq", H.bb.oooo)
        + np.einsum("ipiq->pq", H.ab.oooo)
    )
    G.b.ov = (
        + np.einsum("piqi->pq", H.bb.oovo)
        + np.einsum("ipiq->pq", H.ab.ooov)
    )
    G.b.vo = (
        + np.einsum("piqi->pq", H.bb.vooo)
        + np.einsum("ipiq->pq", H.ab.ovoo)
    )
    G.b.vv = (
        + np.einsum("piqi->pq", H.bb.vovo)
        + np.einsum("ipiq->pq", H.ab.ovov)
    )

    return G

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

def calc_hf_energy_unsorted(H, occ_a, occ_b):

    e1a = np.einsum("ii->", H.a[np.ix_(occ_a, occ_a)])
    e1b = np.einsum("ii->", H.b[np.ix_(occ_b, occ_b)])
    e2a = 0.5 * np.einsum("ijij->", H.aa[np.ix_(occ_a, occ_a, occ_a, occ_a)])
    e2b = np.einsum("ijij->", H.ab[np.ix_(occ_a, occ_b, occ_a, occ_b)])
    e2c = 0.5 * np.einsum("ijij->", H.bb[np.ix_(occ_b, occ_b, occ_b, occ_b)])

    hf_energy = e1a + e1b + e2a + e2b + e2c

    return hf_energy

def calc_hf_frozen_core_energy(e1int, e2int, system):

    if system.nfrozen == 0:
        return 0.0

    occ_a = slice(0, system.nfrozen)
    occ_b = slice(0, system.nfrozen)

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

    Escf = e1a + e1b + e2a + e2b + e2c

    return np.real(Escf) / system.nkpts

def calc_hf_energy_chol(e1int, R_chol, system):
    oa = slice(0, system.nfrozen + system.noccupied_alpha)
    ob = slice(0, system.nfrozen + system.noccupied_beta)

    g_aa_oooo = (
                    np.einsum("xmi,xnj->mnij", R_chol[:, oa, oa], R_chol[:, oa, oa], optimize=True)
                    - np.einsum("xmj,xni->mnij", R_chol[:, oa, oa], R_chol[:, oa, oa], optimize=True)
    )
    g_ab_oooo = (
                    np.einsum("xmi,xnj->mnij", R_chol[:, oa, oa], R_chol[:, ob, ob], optimize=True)
    )
    g_bb_oooo = (
                    np.einsum("xmi,xnj->mnij", R_chol[:, ob, ob], R_chol[:, ob, ob], optimize=True)
                    - np.einsum("xmj,xni->mnij", R_chol[:, ob, ob], R_chol[:, ob, ob], optimize=True)
    )
    e1a = np.einsum("ii->", e1int[oa, oa])
    e1b = np.einsum("ii->", e1int[ob, ob])
    e2a = 0.5 * np.einsum("ijij->", g_aa_oooo)
    e2b = 1.0 * np.einsum("ijij->", g_ab_oooo)
    e2c = 0.5 * np.einsum("ijij->", g_bb_oooo)
    E_scf = e1a + e1b + e2a + e2b + e2c
    return E_scf