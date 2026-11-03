"""Occupied-pair DF-MP2 energies and unrelaxed virtual occupation densities.

Only three-index factors and one virtual-pair matrix are needed. No four-index
Hamiltonian blocks or full doubles-amplitude tensor are constructed. Occupied
and virtual Fock blocks are semicanonicalized internally; returned densities are
in the input basis. For nonstationary spin determinants (e.g. ROHF), this defines
the spin-Fock MP partition, including its first-order singles contribution.
"""
import numpy as np


def _diagonalize_blocks(fock, labels):
    """Hermitian diagonalization without mixing irreps."""
    rotation = np.zeros_like(fock)
    energies = np.empty(len(fock))
    labels = np.asarray(labels)
    for label in dict.fromkeys(labels.tolist()):
        idx = np.flatnonzero(labels == label)
        energies[idx], rotation[np.ix_(idx, idx)] = np.linalg.eigh(fock[np.ix_(idx, idx)])
    return energies, rotation


def dfmp2(reference, need_density=True, denominator_tolerance=1.0e-12):
    """Return ``(correlation_energy, {spin: Dvv})`` using pairwise DF contractions.

    Spin-resolved occupations have the 0--1 convention. The RHF spatial density
    is the sum of the two spin densities. Peak pair workspace is O(v**2), in
    addition to O(naux*o*v) transformed factors and O(v**2) densities.
    """
    H, system = reference.H, reference.system
    eps_o, eps_v, vectors, rotations, singles = {}, {}, {}, {}, {}
    density = {} if need_density else None
    energy = 0.0
    for spin, no in (("a", system.noccupied_alpha), ("b", system.noccupied_beta)):
        labels = system.orbital_symmetries_all
        if reference.is_uhf:
            labels = labels[spin]
        labels = labels[reference.nfrozen:reference.nfrozen + system.norbitals]
        fock = getattr(H, spin)
        eo, Uo = _diagonalize_blocks(fock.oo, labels[:no])
        ev, Uv = _diagonalize_blocks(fock.vv, labels[no:])
        eps_o[spin], eps_v[spin], rotations[spin] = eo, ev, Uv
        vectors[spin] = np.einsum("Pia,ij,ab->Pjb", getattr(H.chol, spin).ov,
                                  Uo, Uv, optimize=True)
        fov = Uo.T @ fock.ov @ Uv
        denom = eo[:, None] - ev[None, :]
        if np.any(np.abs(denom) < denominator_tolerance):
            raise ValueError("DF-MP2: singular occupied-virtual denominator.")
        singles[spin] = fov / denom
        energy += np.sum(fov * singles[spin])
        if need_density:
            density[spin] = singles[spin].T @ singles[spin]

    for s, t, factor in (("a", "a", 0.25), ("a", "b", 1.0), ("b", "b", 0.25)):
        Bs, Bt = vectors[s], vectors[t]
        for i, ei in enumerate(eps_o[s]):
            for j, ej in enumerate(eps_o[t]):
                g = Bs[:, i, :].T @ Bt[:, j, :]
                if s == t:
                    g = g - g.T
                denom = ei + ej - eps_v[s][:, None] - eps_v[t][None, :]
                if np.any(np.abs(denom) < denominator_tolerance):
                    raise ValueError("DF-MP2: singular doubles denominator.")
                amplitude = g / denom
                energy += factor * np.sum(g * amplitude)
                if need_density:
                    if s == t:
                        density[s] += 0.5 * (amplitude @ amplitude.T)
                    else:
                        density[s] += amplitude @ amplitude.T
                        density[t] += amplitude.T @ amplitude
    if need_density:
        density = {s: rotations[s] @ density[s] @ rotations[s].T for s in ("a", "b")}
    return float(energy), density
