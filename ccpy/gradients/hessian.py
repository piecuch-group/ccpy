"""Numerical Cartesian Hessians and harmonic vibrational frequencies for any method.

The method is given the same way as for ccpy.gradients.fd_gradient: a callable that runs
the whole calculation for a ccpy.integrals.Molecule (built with symmetry=False, since every
displaced geometry is a new Molecule from Molecule.with_geometry):

    energy(mol) -> float                                      (always enough)
    gradient(mol) -> ndarray (natoms, 3), or (E, gradient)    (optional; much cheaper)

With a gradient, the Hessian is the central difference of gradients (6 natoms gradient
evaluations); with only an energy it is the second difference of energies (1 + 18 natoms^2
evaluations, using the symmetry H_ij = H_ji).

With dipole=True the gradient callable also returns the dipole moment at each displaced
geometry, e.g. (method.at_geometry(mol).gradient_and_dipole()), and the same central
differences give the atomic polar tensor P_{a,i} = d mu_a / d x_i. harmonic_analysis then
adds double-harmonic IR intensities, A_k = N_A / (12 eps_0 c^2) |d mu / d Q_k|^2.

    from ccpy.gradients.hessian import vibrational_analysis
    result = vibrational_analysis(mol, energy=ccsd_energy)
    result["frequencies"]        # cm^-1, imaginary frequencies as negative numbers

Frequencies come from the mass-weighted Hessian with the translations and rotations
projected out (5 for linear molecules), using the mass of the most abundant isotope of
each element by default.
"""

import time

import numpy as np

from ccpy.integrals.atom_data import ATOM_DATA, Z_TO_ATOM_SYMBOL

# mass (amu) of the most abundant isotope, Z = 1..36 (same values as PySCF's table)
ISOTOPE_MASSES = [
    1.007825, 4.002603, 7.016004, 9.012182, 11.009305, 12.0, 14.003074, 15.994915, 18.998403,
    19.99244, 22.98977, 23.985042, 26.981538, 27.976927, 30.973762, 31.972071, 34.968853,
    39.962383, 38.963707, 39.962591, 44.95591, 47.947947, 50.943964, 51.940512, 54.93805,
    55.934942, 58.9332, 57.935348, 62.929601, 63.929147, 68.925581, 73.921178, 74.921596,
    79.916522, 78.918338, 83.911507,
]

# CODATA 2018
HARTREE_J = 4.3597447222071e-18
BOHR_M = 5.29177210903e-11
AMU_KG = 1.66053906660e-27
C_CM_S = 2.99792458e10
E_C = 1.602176634e-19
EPS0 = 8.8541878128e-12
N_A = 6.02214076e23
# |d mu / dQ|^2 in (e / amu^1/2)^2 -> IR intensity (km/mol)
AU_TO_KM_MOL = N_A * E_C**2 / (12.0 * EPS0 * (C_CM_S / 100.0) ** 2 * AMU_KG) / 1000.0
# sqrt(Eh / (bohr^2 amu)) in rad/s -> wavenumber (cm^-1)
AU_TO_WAVENUMBER = np.sqrt(HARTREE_J / (BOHR_M**2 * AMU_KG)) / (2.0 * np.pi * C_CM_S)
WAVENUMBER_TO_HARTREE = 4.556335252912e-06


def isotope_masses(molecule):
    masses = []
    for Z in molecule.atomic_charges:
        Z = int(Z)
        masses.append(ISOTOPE_MASSES[Z - 1] if Z <= len(ISOTOPE_MASSES) else ATOM_DATA[Z]["mass"])
    return np.array(masses)


def _coordinates(molecule):
    return np.array([xyz for _, xyz in molecule.atoms], dtype=float)


def fd_hessian(molecule, energy=None, gradient=None, step=2.5e-3, dipole=False, verbose=True):
    """Cartesian Hessian, shape (3 natoms, 3 natoms), in Eh/bohr^2, atom-major (x, y, z).

    gradient : callable, optional
        Molecule -> gradient (natoms, 3) or (energy, gradient). Then
        H[:, i] = (g(x + h e_i) - g(x - h e_i)) / 2h, symmetrized.
    dipole : bool
        The gradient callable returns (gradient, dipole) or (energy, gradient, dipole), the
        dipole moment (3,) in atomic units; (H, P) is returned with the atomic polar tensor
        P[:, i] = (mu(x + h e_i) - mu(x - h e_i)) / 2h, shape (3, 3 natoms), in e.
    energy : callable
        Molecule -> energy, used when no gradient is given:
        H_ii = (E(+h) - 2 E0 + E(-h)) / h^2,
        H_ij = (E(+h,+h) - E(+h,-h) - E(-h,+h) + E(-h,-h)) / 4h^2.
    step : float
        Cartesian displacement h in bohr.
    """
    if energy is None and gradient is None:
        raise ValueError("Give an energy or a gradient callable.")
    if dipole and gradient is None:
        raise ValueError("Dipole derivatives need a gradient callable that returns the dipole.")
    t0 = time.perf_counter()
    X0 = _coordinates(molecule)
    n = X0.size
    count = [0]

    def at(displacement):
        count[0] += 1
        return molecule.with_geometry(X0 + displacement.reshape(X0.shape), unit="bohr")

    def unit(i, h):
        d = np.zeros(n)
        d[i] = h
        return d

    H = np.zeros((n, n))
    P = np.zeros((3, n))
    if gradient is not None:
        def grad(displacement):
            out = gradient(at(displacement))
            if dipole:
                return np.asarray(out[-2], dtype=float).ravel(), np.asarray(out[-1], dtype=float)
            g = out[1] if isinstance(out, tuple) else out
            return np.asarray(g, dtype=float).ravel(), None

        for i in range(n):
            (gp, mup), (gm, mum) = grad(unit(i, step)), grad(unit(i, -step))
            H[:, i] = (gp - gm) / (2.0 * step)
            if dipole:
                P[:, i] = (mup - mum) / (2.0 * step)
        H = 0.5 * (H + H.T)
        kind = f"central differences of {count[0]} gradients"
    else:
        E0 = energy(at(np.zeros(n)))
        Ep = [energy(at(unit(i, step))) for i in range(n)]
        Em = [energy(at(unit(i, -step))) for i in range(n)]
        for i in range(n):
            H[i, i] = (Ep[i] - 2.0 * E0 + Em[i]) / step**2
            for j in range(i):
                Epp = energy(at(unit(i, step) + unit(j, step)))
                Epm = energy(at(unit(i, step) + unit(j, -step)))
                Emp = energy(at(unit(i, -step) + unit(j, step)))
                Emm = energy(at(unit(i, -step) + unit(j, -step)))
                H[i, j] = H[j, i] = (Epp - Epm - Emp + Emm) / (4.0 * step**2)
        kind = f"second differences of {count[0]} energies"
    if verbose:
        print(f"\n   Finite-difference Hessian: {kind}, step {step} bohr "
              f"({time.perf_counter() - t0:.1f} s)")
    return (H, P) if dipole else H


def _rigid_body_basis(coordinates, masses):
    """Orthonormal basis (columns) of the mass-weighted translations and rotations about
    the center of mass; 5 vectors for a linear molecule, 6 otherwise."""
    natoms = len(masses)
    sqrt_m = np.sqrt(masses)
    r = coordinates - masses @ coordinates / masses.sum()
    vectors = []
    for x in range(3):
        t = np.zeros((natoms, 3))
        t[:, x] = sqrt_m
        vectors.append(t.ravel())
    for x in range(3):
        axis = np.zeros(3)
        axis[x] = 1.0
        vectors.append((np.cross(axis, r) * sqrt_m[:, None]).ravel())
    u, s, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    return u[:, s > 1.0e-6 * s.max()]


def harmonic_analysis(hessian, molecule, masses=None, dipole_derivatives=None, verbose=True):
    """Harmonic frequencies and normal modes from a Cartesian Hessian (Eh/bohr^2).

    The mass-weighted Hessian is diagonalized in the space orthogonal to the translations
    and rotations. Returns a dict with
        frequencies     (nvib,) cm^-1, imaginary ones as negative numbers, ascending
        normal_modes    (nvib, natoms, 3) Cartesian displacements, normalized
        reduced_masses  (nvib,) amu
        force_constants (nvib,) mdyn/angstrom
        zpe             zero-point energy (Eh) of the real modes
        eigenvalues     (nvib,) of the projected mass-weighted Hessian, Eh/(bohr^2 amu)
    and, with the atomic polar tensor dipole_derivatives (3, 3 natoms) of fd_hessian(dipole=True),
        dipole_derivatives (nvib, 3) d mu / dQ_k, e / amu^1/2
        ir_intensities     (nvib,) double-harmonic IR intensities, km/mol
    """
    masses = isotope_masses(molecule) if masses is None else np.asarray(masses, dtype=float)
    X = _coordinates(molecule)
    m3 = np.repeat(masses, 3)
    Hm = hessian / np.sqrt(np.outer(m3, m3))

    # basis of the vibrational space: orthogonal complement of the rigid-body motions
    rb = _rigid_body_basis(X, masses)
    u, s, _ = np.linalg.svd(np.eye(len(m3)) - rb @ rb.T)
    vib = u[:, : len(m3) - rb.shape[1]]
    eigenvalues, c = np.linalg.eigh(vib.T @ Hm @ vib)
    modes_mw = vib @ c

    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * AU_TO_WAVENUMBER
    cart = modes_mw / np.sqrt(m3)[:, None]
    norms = np.linalg.norm(cart, axis=0)
    reduced_masses = 1.0 / norms**2
    # k = mu omega^2 in mdyn/angstrom: 1 Eh/bohr^2 = 15.569141 mdyn/angstrom
    force_constants = reduced_masses * eigenvalues * 15.569141
    zpe = 0.5 * np.sum(frequencies[frequencies > 0]) * WAVENUMBER_TO_HARTREE
    result = {
        "frequencies": frequencies,
        "normal_modes": (cart / norms).T.reshape(-1, len(masses), 3),
        "reduced_masses": reduced_masses,
        "force_constants": force_constants,
        "zpe": zpe,
        "eigenvalues": eigenvalues,
    }
    if dipole_derivatives is not None:
        # Q_k in bohr amu^1/2, x = sum_k cart[:, k] Q_k
        dmu_dQ = (np.asarray(dipole_derivatives) @ cart).T
        result["dipole_derivatives"] = dmu_dQ
        result["ir_intensities"] = AU_TO_KM_MOL * np.sum(dmu_dQ**2, axis=1)
    if verbose:
        print("\n   Harmonic Vibrational Analysis")
        print("   -----------------------------")
        print("   Masses (amu): " + ", ".join(
            f"{Z_TO_ATOM_SYMBOL[int(Z)]} {m:.6f}" for Z, m in zip(molecule.atomic_charges, masses)))
        print(f"   Rigid-body modes projected out: {rb.shape[1]}")
        ir = result.get("ir_intensities")
        print("\n   Mode    Frequency (cm-1)    Red. mass (amu)    Force const. (mdyn/A)"
              + ("    IR int. (km/mol)" if ir is not None else ""))
        for k, (w, mu, f) in enumerate(zip(frequencies, reduced_masses, force_constants), 1):
            label = f"{abs(w):12.2f}{'i' if w < 0 else ' '}"
            print(f"   {k:4d}    {label}        {mu:12.4f}       {f:14.4f}"
                  + (f"    {ir[k - 1]:16.4f}" if ir is not None else ""))
        print(f"\n   Zero-point energy: {zpe:.8f} Eh ({zpe / WAVENUMBER_TO_HARTREE:.2f} cm-1)\n")
    return result


def vibrational_analysis(molecule, energy=None, gradient=None, step=2.5e-3, masses=None, dipole=False,
                         verbose=True):
    """fd_hessian followed by harmonic_analysis; the result also holds the Hessian (and, with
    dipole=True, the atomic polar tensor and the IR intensities)."""
    H = fd_hessian(molecule, energy=energy, gradient=gradient, step=step, dipole=dipole, verbose=verbose)
    H, P = H if dipole else (H, None)
    result = harmonic_analysis(H, molecule, masses=masses, dipole_derivatives=P, verbose=verbose)
    result["hessian"] = H
    if dipole:
        result["atomic_polar_tensor"] = P
    return result
