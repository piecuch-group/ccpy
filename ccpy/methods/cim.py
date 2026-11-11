"""Cluster-in-molecule (CIM) local correlation for ground-state methods.

The correlation energy is a sum of contributions of the occupied localized orbitals (LMOs),

    Ec = sum_i' Ec_i',    Ec_i' = (1 / M_i') sum_{P containing i' as central} Ec_i'(P),

each obtained from an ordinary calculation in the small orbital space of a CIM subsystem P.
This module implements

* the single-environment (SECIM) subsystems of W. Li and P. Piecuch, J. Phys. Chem. A 114,
  8644 (2010): the central LMOs of a subsystem are the LMOs with a Mulliken charge above
  ``central_charge`` on one non-hydrogen atom or the hydrogen atoms bonded to it, the
  environment are the LMOs j with |<i|f|j>| > ``zeta`` for a central i, and subsystems
  embedded in larger ones are merged into them. Anchoring the subsystems on atoms keeps them
  from rearranging unnecessarily along potential energy surfaces;
* the quasi-canonical (QCMO) subsystem orbitals and the mixed-index energy contributions of
  W. Li, P. Piecuch, J. R. Gour, and S. Li, J. Chem. Phys. 131, 114109 (2009), Eqs. (14)-(22):
  each subsystem is solved in the QCMOs that diagonalize the occupied-occupied and
  unoccupied-unoccupied Fock blocks, so the noniterative triples corrections keep their
  canonical form, and dE_i'(P) is the energy with its first occupied index rotated back to
  the LMO i';
* the fixed-domain option of the same paper: subsystems found at one geometry are reused at
  others (matching the LMOs by their atomic populations), giving smooth potential energy
  surfaces; finite-difference gradients always use it;
* the multilevel CIM-CC/MP2 scheme of W. Li and P. Piecuch, J. Phys. Chem. A 114, 6721 (2010):
  subsystems whose central LMOs belong entirely to the atoms of a higher level are treated by
  that level, and are not merged into lower-level subsystems that contain them;
* adaptive CC(P;Q) (not in the papers): every subsystem runs the adaptive loop in its QCMO
  basis, and the CC(P;3) correction is partitioned like CR-CC(2,3).

Each subsystem is a Reference containing only its own orbitals, which are linear
combinations of the AOs of its extended domain (its atoms plus the buffer atoms within
``buffer_radius``, steps 7-12 of the 2009 paper): the occupied LMOs (core included) whose
AO domains lie in the extended domain are projected onto its AOs and orthonormalized, the
subsystem's own projected LMOs are its occupied orbitals, and its unoccupied orbitals are
the part of the orthogonal complement of all of them (the span of the PAOs of the extended
domain) that lies on its AO-domain atoms (the eta selection). The Fock matrix is that of
the full system and the two-electron integrals are density fitted with the auxiliary
functions of the extended domain, so no full-system MO integrals are formed and every
method runs unchanged:

    e = CRCC23(mf, nfrozen=1, cim=CIMOptions(zeta=0.003)).energy()
    e = MP2(mf, nfrozen=1, cim=CIMOptions(levels={CRCC23: [0, 1]})).energy()

Differences from the papers: the subsystem orbitals stay in the extended domain and are
strictly orthonormal (an option in the papers), and the eta selection picks the subspace
of the unoccupied space with weight above 1 - eta^2 on the AO-domain atoms instead of
testing its Boys-localized orbitals one by one; when the extended domain is the whole
molecule this recovers the canonical calculation for a subsystem containing all orbitals. X-H bonds use covalent radii
(+ ``bond_tolerance``). Closed-shell RHF references with density fitting only.
"""
import contextlib
import io
import warnings
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from ccpy.methods.base import METHOD_OPTIONS, SOLVER_OPTIONS, MethodBase
from ccpy.methods.hamiltonian import DFHamiltonian, build_fock
from ccpy.methods.reference import Reference
from ccpy.models.system import System

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class CIMOptions:
    """CIM settings.

    zeta : environment threshold on |<i|f|j>| between LMOs (SECIM(zeta)); smaller is more accurate
    central_charge : Mulliken charge (of 2 electrons) on an atom group that makes an LMO central
    domain_charge : the AO domain of an LMO holds its largest-charge atoms adding up to this
    bond_tolerance : X-H bond if the distance is below the sum of covalent radii plus this (angstrom)
    eta : an unoccupied orbital of the extended domain belongs to the subsystem if it differs
          from its projection onto the AOs of the AO-domain atoms by less than eta (in norm)
    buffer_radius : the extended domain of a subsystem adds the atoms (with their bonded
                    hydrogens) within this distance of its AO-domain atoms (angstrom)
    levels : {method class: atom indices}, higher levels than the method CIM is called on, the
             highest first; hydrogen atoms follow the atom they are bonded to
    domains : CIMDomains of another geometry to reuse (fixed-domain CIM)
    verbose : print the output of the subsystem calculations
    """

    zeta: float = 0.003
    central_charge: float = 0.3
    domain_charge: float = 1.98
    bond_tolerance: float = 0.168
    eta: float = 0.5  # the 2009 paper's choice when the unoccupied orbitals are strictly orthogonalized
    buffer_radius: float = 4.0
    levels: dict = field(default_factory=dict)
    domains: "CIMDomains" = None
    localization_tol: float = 1.0e-12
    localization_maxiter: int = 1000
    verbose: bool = False


@dataclass(frozen=True)
class CIMSubsystem:
    """One subsystem: LMO indices (central ones first in ``occupied``), AO-domain atoms, method,
    and the atoms of its extended domain (set from ``buffer_radius`` when None)."""

    central: tuple
    occupied: tuple
    atoms: tuple
    method: type
    extended: tuple = None


@dataclass(frozen=True)
class CIMDomains:
    """Subsystems of one geometry, with the atomic Mulliken populations (nlmo, natoms) of its
    LMOs used to identify the LMOs at other geometries."""

    subsystems: tuple
    populations: np.ndarray


# ---------------------------------------------------------------------- orbitals
def boys_localize(C, mol, tol=1.0e-12, maxiter=1000):
    """Unitary U (n, n) such that C @ U are the Boys orbitals of the space of C (Jacobi sweeps
    maximizing sum_i |<i|r|i>|^2)."""
    from ccpy.gradients.derivative_integrals import multipole_integrals

    _, D = multipole_integrals(mol, 1)
    R = np.einsum("pi,xpq,qj->xij", C, D, C, optimize=True)
    n = C.shape[1]
    U = np.eye(n)
    for _ in range(maxiter):
        gain = 0.0
        for i in range(n):
            for j in range(i):
                x = R[:, i, i] - R[:, j, j]
                y = R[:, i, j]
                A = y @ y - 0.25 * (x @ x)
                B = x @ y
                if A * A + B * B < 1.0e-30:
                    continue
                gain += A + np.sqrt(A * A + B * B)
                theta = 0.25 * np.arctan2(B, -A)
                c, s = np.cos(theta), np.sin(theta)
                Ui, Uj = U[:, i].copy(), U[:, j].copy()
                U[:, i], U[:, j] = c * Ui + s * Uj, c * Uj - s * Ui
                Ri, Rj = R[:, :, i].copy(), R[:, :, j].copy()
                R[:, :, i], R[:, :, j] = c * Ri + s * Rj, c * Rj - s * Ri
                Ri, Rj = R[:, i, :].copy(), R[:, j, :].copy()
                R[:, i, :], R[:, j, :] = c * Ri + s * Rj, c * Rj - s * Ri
        if gain < tol:
            return U
    raise RuntimeError("Boys localization did not converge.")


def lmo_populations(C, mol):
    """Mulliken charges (of 2 electrons) of the orbitals C on the atoms, (norb, natoms)."""
    SC = mol.ints_overlap() @ C
    return np.array([2.0 * np.einsum("mi,mi->i", C[first:last], SC[first:last])
                     for first, last in mol.basis.center_first_and_last]).T


def hydrogen_anchors(mol, tolerance):
    """Atom each atom is grouped with: non-hydrogen atoms with themselves, hydrogen atoms with
    the closest non-hydrogen atom they are bonded to (or themselves)."""
    from ccpy.integrals.atom_data import ATOM_DATA

    Z = mol.atomic_charges.astype(int)
    xyz = mol.atomic_positions * BOHR_TO_ANGSTROM
    anchors = np.arange(mol.natoms)
    for h in np.flatnonzero(Z == 1):
        heavy = np.flatnonzero(Z > 1)
        if heavy.size == 0:
            continue
        r = np.linalg.norm(xyz[heavy] - xyz[h], axis=1)
        bonded = r < ATOM_DATA[1]["radius_covalent"] + np.array([ATOM_DATA[z]["radius_covalent"] for z in Z[heavy]]) \
            + tolerance
        if bonded.any():
            anchors[h] = heavy[bonded][np.argmin(r[bonded])]
    return anchors


def domain_atoms(populations, charge):
    """AO-domain atoms of each LMO: the largest-charge atoms whose charges exceed `charge`."""
    domains = []
    for q in populations:
        order = np.argsort(-q, kind="stable")
        count = min(len(q), int(np.searchsorted(np.cumsum(q[order]), charge, side="right")) + 1)
        domains.append(set(order[:count].tolist()))
    return domains


def buffer_atoms(mol, atoms, anchors, radius):
    """Extended domain: `atoms` plus every atom whose anchor (itself, or the non-hydrogen atom
    a hydrogen is bonded to) lies within `radius` angstrom of one of them."""
    xyz = mol.atomic_positions * BOHR_TO_ANGSTROM
    r = np.linalg.norm(xyz[anchors][:, None, :] - xyz[list(atoms)][None, :, :], axis=2).min(axis=1)
    return tuple(sorted(set(atoms) | set(np.flatnonzero(r < radius).tolist())))


# ---------------------------------------------------------------------- subsystems
def secim_subsystems(fock, populations, anchors, options, method):
    """SECIM subsystems from the LMO Fock matrix (nlmo, nlmo) and populations (nlmo, natoms)."""
    nlmo = fock.shape[0]
    levels = list(options.levels.items())
    # rank 0 is `method`; levels[0] (the highest) has rank len(levels)
    level_atoms = [set(atoms) | {a for a in range(len(anchors)) if anchors[a] in set(atoms)}
                   for _, atoms in levels]
    assigned = [set(np.flatnonzero(q > options.central_charge).tolist()) or {int(np.argmax(q))}
                for q in populations]
    lmo_rank = [max([len(levels) - r for r, atoms in enumerate(level_atoms) if a <= atoms], default=0)
                for a in assigned]

    groups = []
    for anchor in dict.fromkeys(anchors.tolist()):
        members = {a for a in range(len(anchors)) if anchors[a] == anchor}
        central = {i for i in range(nlmo) if assigned[i] & members}
        if central:
            environment = {j for j in range(nlmo) if np.max(np.abs(fock[sorted(central), j])) > options.zeta}
            groups.append([central, central | environment, max(lmo_rank[i] for i in central)])
    # merge each group embedded in others of no lower rank into them (identical ones into the first)
    merged = True
    while merged:
        merged = False
        for g in sorted(range(len(groups)), key=lambda g: len(groups[g][1])):
            hosts = [h for h in range(len(groups)) if h != g and groups[g][1] <= groups[h][1]
                     and groups[g][2] <= groups[h][2] and (groups[g][1] != groups[h][1] or h < g)]
            if hosts:
                for h in hosts:
                    groups[h][0] |= groups[g][0]
                del groups[g]
                merged = True
                break

    atoms = domain_atoms(populations, options.domain_charge)
    subsystems = []
    for central, occupied, rank in groups:
        cls = method if rank == 0 else levels[len(levels) - rank][0]
        order = sorted(central) + sorted(occupied - central)
        subsystems.append(CIMSubsystem(tuple(order[:len(central)]), tuple(order),
                                       tuple(sorted(set().union(*(atoms[i] for i in occupied)))), cls))
    return tuple(subsystems)


def match_lmos(populations_ref, populations):
    """Index of the LMO matching each reference LMO (assignment on atomic populations)."""
    from scipy.optimize import linear_sum_assignment

    if populations_ref.shape != populations.shape:
        raise ValueError("Fixed CIM domains need the same molecule and frozen core at every geometry.")
    cost = np.abs(populations_ref[:, None, :] - populations[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]


def sub_basis(basis, data, atoms):
    """Basis of the shells of `atoms` (sorted) from `basis` and its serialization `data`,
    with the number of shells of each atom."""
    from ccpy.integrals.basis import build_basis_from_dict

    ranges = [basis.center_first_and_last_shell[a] for a in atoms]
    shells = [data["shells"][k] for first, last in ranges for k in range(first, last)]
    return build_basis_from_dict({"schema_version": 1, "nshells": len(shells), "shells": shells}), \
        [last - first for first, last in ranges]


def local_df_vectors(mol, atoms, C, basis_data, aux_data):
    """B^P_pq of the orbitals C (AOs of `atoms` x orbitals), density fitted with the auxiliary
    functions of `atoms` only; the three-center integrals are formed one auxiliary atom at a
    time and transformed at once."""
    from ccpy.integrals import integrals
    from ccpy.integrals.linalg import cholesky_metric_inverse_sqrt, invsqrt_matrix

    orb, _ = sub_basis(mol.basis, basis_data, atoms)
    aux, counts = sub_basis(mol.auxiliary_basis, aux_data, atoms)
    M = integrals.coulomb_2c(mol, aux, aux)
    X = cholesky_metric_inverse_sqrt(M) if mol.df_metric_tol is None else invsqrt_matrix(M, rtol=mol.df_metric_tol)[0]
    raw = np.empty((aux.size, C.shape[1], C.shape[1]))
    shell, function = 0, 0
    for count in counts:
        block = integrals.coulomb_3c_by_shell(mol, [(shell, shell + count), (0, orb.nshells), (0, orb.nshells)],
                                              aux, orb)
        raw[function:function + block.shape[0]] = np.einsum("Pmn,mp,nq->Ppq", block, C, C, optimize=True)
        shell, function = shell + count, function + block.shape[0]
    return np.asfortranarray(np.tensordot(X, raw, axes=([1], [0])))


def subsystem_reference(parent, C, aos, atoms, F_d, nocc, basis_data, aux_data):
    """Reference of a subsystem in its orthonormal orbitals C (AOs `aos` of the extended
    domain `atoms` x orbitals, the nocc occupied ones first) with the full-system Fock matrix
    (F_d, its block on `aos`) and local DF integrals. The one-electron integrals are the Fock
    matrix minus the subsystem's own mean field, so the Fock matrix rebuilt from them is the
    full-system one."""
    mol = parent.molecule
    B = local_df_vectors(mol, atoms, C, basis_data, aux_data)
    F = C.T @ F_d @ C
    zero = np.zeros_like(F)
    h = F - build_fock(zero, zero, B, B, nocc, nocc)[0]
    norb = C.shape[1]
    C_full = np.zeros((mol.nbf, norb))
    C_full[aos] = C
    eps = np.diag(F).copy()
    occupation = np.zeros(norb)
    occupation[:nocc] = 2.0
    child = object.__new__(Reference)
    child.scf, child.nfrozen, child.ndelete = parent.scf, 0, 0
    child.is_uhf, child.rhf_symmetry = False, True
    child.mo_coeff, child.mo_energy = {"a": C_full, "b": C_full}, {"a": eps, "b": eps.copy()}
    child.h_mo, child.B_mo = {"a": h, "b": h}, {"a": B, "b": B}
    child.system = System(2 * nocc, norb, 1, 0, reference_energy=parent.energy, cholesky=True,
                          mo_energies=eps, mo_occupation=occupation)
    child.H = DFHamiltonian(child.system, F, F, B, B)
    child._legacy_H = None
    child.fno_result = child.fno_parent = child.orbital_transform = None
    return child


def light_reference(scf, nfrozen, ndelete):
    """Reference of an RHF calculation without its MO integrals (CIM never needs them)."""
    if scf.e_tot is None:
        scf.run()
    mol = scf.mol
    ref = object.__new__(Reference)
    ref.scf, ref.nfrozen, ref.ndelete, ref.is_uhf = scf, nfrozen, ndelete, False
    ref.system = System(mol.nelectrons, scf.mo_coeff.shape[1], mol.multiplicity, nfrozen, ndelete=ndelete,
                        point_group=mol.point_group, orbital_symmetries=scf.orbital_symmetries(),
                        charge=mol.charge, nuclear_repulsion=mol.nuclear_repulsion, reference_energy=scf.e_tot,
                        mo_energies=scf.mo_energy, mo_occupation=scf.mo_occ, cholesky=True)
    ref.mo_coeff = {"a": scf.mo_coeff, "b": scf.mo_coeff}
    ref.mo_energy = {"a": scf.mo_energy, "b": scf.mo_energy}
    ref.rhf_symmetry = ref.system.noccupied_alpha == ref.system.noccupied_beta
    ref._legacy_H = None
    ref.fno_result = ref.fno_parent = ref.orbital_transform = None
    return ref


# ---------------------------------------------------------------------- energy contributions
def cc_central_energy(T, H, P):
    """sum_i'k' P_i'k' dE_k'i' of a CC energy from T1 and T2, Eqs. (15) and (17)-(20):
    e_ki = sum_a f_ka t_ai + 1/4 sum_jab <kj||ab> tau_abij per spin (opposite-spin pairs
    counted from both of their spins), P the (weighted) central projector in the QCMO basis."""
    tau_aa = T.aa + np.einsum("ai,bj->abij", T.a, T.a) - np.einsum("aj,bi->abij", T.a, T.a)
    tau_ab = T.ab + np.einsum("ai,bj->abij", T.a, T.b)
    tau_bb = T.bb + np.einsum("ai,bj->abij", T.b, T.b) - np.einsum("aj,bi->abij", T.b, T.b)
    e_a = (H.a.ov @ T.a
           + 0.25 * np.einsum("kjab,abij->ki", H.aa.oovv, tau_aa, optimize=True)
           + 0.5 * np.einsum("kjab,abij->ki", H.ab.oovv, tau_ab, optimize=True))
    e_b = (H.b.ov @ T.b
           + 0.25 * np.einsum("kjab,abij->ki", H.bb.oovv, tau_bb, optimize=True)
           + 0.5 * np.einsum("jkab,abji->ki", H.ab.oovv, tau_ab, optimize=True))
    return np.sum(P * e_a) + np.sum(P * e_b)


def mp2_central_energy(H, P):
    """Central MP2 energy in the QCMO basis; the small f_ia of the projected subsystem
    orbitals (zero for HF orbitals) is neglected, as in CIM-MP2."""
    eo, ev = np.diag(H.a.oo), np.diag(H.a.vv)
    D = eo[None, None, :, None] + eo[None, None, None, :] - ev[:, None, None, None] - ev[None, :, None, None]
    no, nv = len(eo), len(ev)
    T = SimpleNamespace(a=np.zeros((nv, no)), b=np.zeros((nv, no)),
                        aa=H.aa.oovv.transpose(2, 3, 0, 1) / D,
                        ab=H.ab.oovv.transpose(2, 3, 0, 1) / D,
                        bb=H.bb.oovv.transpose(2, 3, 0, 1) / D)
    return cc_central_energy(T, H, P)


def occupied_derivative(X, P, lower=(), upper=()):
    """d/dl of X under the occupied similarity transformation exp(l P): +P on the lower
    (annihilated, ket) occupied slots and -P on the upper (bra) ones. Contracted occupied
    indices cancel, so in a moment M_ijk the result is (P_1 + P_2 + P_3) M."""
    Y = np.zeros_like(X)
    for slots, sign in ((lower, 1.0), (upper, -1.0)):
        for s in slots:
            Y += sign * np.moveaxis(np.tensordot(P, X, axes=([1], [s])), 0, s)
    return np.asfortranarray(Y)


def crcc23_central_correction(T, L, H, H0, system, P, t3_excitations=None):
    """Central part sum_i' P_i'i' dE_i'(2,3), Eqs. (16), (21), and (22), of all CR-CC(2,3)
    variants, closed-shell RHF; with a P space of triples ``t3_excitations``, of the CC(P;3)
    corrections (two-body approximation), whose triples sum skips the P space.

    The kernels of ccpy.moments.crcc23_chol (ccp3_chol) sum l_ijk^abc M^ijk_abc / 36 with M
    linear in its inputs (H-bar blocks times T2) and l built from separate inputs. Replacing
    the moment inputs by their occupied derivatives, one factor at a time, gives
    sum l (P_1 + P_2 + P_3) M / 36 = 3 sum_i'k' P_i'k' dE_k'(2,3). The P space is defined in
    the QCMO basis, so with it this is the central part by definition (exact when the
    subsystem projectors add up to one), not a rotated canonical quantity."""
    from ccpy.lib.core import ccp3_opt_loops, crcc_loops
    from ccpy.moments.crcc23_chol import (aaa_H3_aaa_diagonal, aab_H3_aab_diagonal, abb_H3_abb_diagonal,
                                          get_vvvv_diagonal)

    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    h_aa_vvvv, h_ab_vvvv, _ = get_vvvv_diagonal(H, T)
    noa, nua = system.noccupied_alpha, system.nunoccupied_alpha
    nob, nub = system.noccupied_beta, system.nunoccupied_beta
    if t3_excitations is None:
        loop_a = lambda *args: crcc_loops.crcc23a_opt(*args, noa, nua)  # noqa: E731
        loop_b = lambda *args: crcc_loops.crcc23b_opt(*args, noa, nua, nob, nub)  # noqa: E731
    else:
        loop_a = lambda *args: ccp3_opt_loops.ccp3a_2ba(t3_excitations["aaa"].T, *args)  # noqa: E731
        loop_b = lambda *args: ccp3_opt_loops.ccp3b_2ba(t3_excitations["aab"].T, *args)  # noqa: E731

    #### aaa correction ####
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    # moment inputs: H(vooo), I(vvov), T2
    H2A_vooo_P = occupied_derivative(H.aa.vooo, P, lower=(2, 3), upper=(1,))
    I2A_vvov_P = occupied_derivative(I2A_vvov, P, lower=(2,))
    t2a_P = occupied_derivative(T.aa, P, lower=(2, 3))
    d_aaa = np.zeros(4)
    for vooo, vvov, t2a in ((H2A_vooo_P, I2A_vvov_P, T.aa), (H.aa.vooo, I2A_vvov, t2a_P)):
        d_aaa += loop_a(
            t2a, L.a, L.aa,
            vooo, vvov, H.aa.oovv, H.a.ov,
            H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
            H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
            h_aa_vvvv,
            d3aaa_o, d3aaa_v,
        )

    #### aab correction ####
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    # moment inputs: I(ovoo), I(vooo) (both spins), H(vvvo), H(vvov) (both spins), T2
    I2B_ovoo_P = occupied_derivative(I2B_ovoo, P, lower=(2, 3), upper=(0,))
    I2B_vooo_P = occupied_derivative(I2B_vooo, P, lower=(2, 3), upper=(1,))
    I2A_vooo_P = occupied_derivative(I2A_vooo, P, lower=(2, 3), upper=(1,))
    H2B_vvvo_P = occupied_derivative(H.ab.vvvo, P, lower=(3,))
    H2B_vvov_P = occupied_derivative(H.ab.vvov, P, lower=(2,))
    H2A_vvov_P = occupied_derivative(H.aa.vvov, P, lower=(2,))
    t2a_P = occupied_derivative(T.aa, P, lower=(2, 3))
    t2b_P = occupied_derivative(T.ab, P, lower=(2, 3))
    d_aab = np.zeros(4)
    for t2a, t2b, ovoo, vooo_b, vooo_a, vvvo, vvov_b, vvov_a in (
            (T.aa, T.ab, I2B_ovoo_P, I2B_vooo_P, I2A_vooo_P, H2B_vvvo_P, H2B_vvov_P, H2A_vvov_P),
            (t2a_P, t2b_P, I2B_ovoo, I2B_vooo, I2A_vooo, H.ab.vvvo, H.ab.vvov, H.aa.vvov)):
        d_aab += loop_b(
            t2a, t2b, L.a, L.b, L.aa, L.ab,
            ovoo, vooo_b, vooo_a,
            vvvo, vvov_b, vvov_a,
            H.ab.vovv, H.ab.ovvv, H.aa.vovv,
            H.ab.ooov, H.ab.oovo, H.aa.ooov,
            H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.aa.oooo, h_aa_vvvv, H.ab.ovov,
            H.ab.vovo, H.ab.oooo, h_ab_vvvv, H.bb.voov,
            d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        )

    delta = 2.0 * (d_aaa + d_aab) / 3.0
    return dict(zip("ABCD", delta))


# ---------------------------------------------------------------------- driver
class CIMDevelopmentWarning(UserWarning):
    """Issued by every CIM calculation: CIM is under development, and its limited tests do not
    establish general local-correlation accuracy or performance."""


class CIM(MethodBase):
    """CIM energy of a ground-state method; created by passing ``cim=CIMOptions(...)`` to a
    method, e.g. ``CCSD(mf, nfrozen=1, cim=CIMOptions())``.

    Supported methods: ground-state CC (CCSD, CCSDT, CC3, ...), MP2, CR-CC(2,3) (all
    variants), and adaptive CC(P;Q) (each subsystem selects its own P space). Gradients are finite differences with the subsystems fixed at the
    reference geometry.

    Attributes
    ----------
    domains : CIMDomains, the subsystems used (as defined at their geometry)
    subsystems : list of dicts with the orbital counts, method, and energy of each subsystem
    """

    name = "CIM"

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, method, upstream, nfrozen=None, ndelete=None, *, cim, fno=None, **kwargs):
        warnings.warn("CIM is under development: its subsystems, energies and gradients are not yet "
                      "validated beyond the included tests.", CIMDevelopmentWarning, stacklevel=3)
        if not isinstance(cim, CIMOptions):
            raise TypeError("cim must be CIMOptions.")
        if fno is not None and fno is not False:
            raise NotImplementedError("CIM cannot be combined with FNOs yet.")
        if isinstance(upstream, MethodBase):
            raise ValueError("CIM builds its subsystem calculations itself; start it from an SCF or Reference.")
        options = {k: v for k, v in kwargs.items() if k in SOLVER_OPTIONS or k in METHOD_OPTIONS}
        super().__init__(upstream, nfrozen, ndelete, **options)
        self._ctor_args = ((method, upstream, nfrozen, ndelete), dict(cim=cim, **kwargs))
        self.method = method
        self.method_options = {k: v for k, v in kwargs.items() if k not in options}
        self.cim = cim
        self.correlation_energy = 0.0
        self.domains = None
        self.subsystems = []

    @staticmethod
    def resolve_reference(upstream, nfrozen=None, ndelete=None):
        from ccpy.scf.rhf import RHF

        if isinstance(upstream, RHF):
            return light_reference(upstream, nfrozen or 0, ndelete or 0)
        return MethodBase.resolve_reference(upstream, nfrozen, ndelete)

    def at_geometry(self, molecule):
        """The same CIM calculation at another geometry, with the subsystems of this one."""
        (method, upstream, nfrozen, ndelete), kwargs = self._ctor_args
        self.ensure_solved()
        kwargs = dict(kwargs, cim=replace(self.cim, domains=self.domains))
        return CIM(method, upstream.with_molecule(molecule), nfrozen, ndelete, **kwargs)

    def has_analytic_gradient(self, **kwargs):
        return False

    def energy(self):
        self.ensure_solved()
        return self.reference_energy + self.correlation_energy

    def raw_energy(self):
        return self.energy()

    def solve(self):
        from ccpy.methods.cc import CCBase
        from ccpy.methods.ccpq import AdaptiveCCPQ
        from ccpy.methods.corrections import CRCC23
        from ccpy.methods.hbar import HBarCCSD
        from ccpy.methods.mbpt import MP2
        from ccpy.scf.rhf import RHF

        ref, mol, options = self.reference, self.molecule, self.cim
        if not isinstance(ref.scf, RHF) or not ref.rhf_symmetry or mol.point_group != "C1" or ref.ndelete:
            raise NotImplementedError("CIM needs a closed-shell RHF reference without symmetry or deleted orbitals.")
        if mol.auxiliary_basis is None:
            raise NotImplementedError("CIM needs density fitting (an auxiliary basis), not cholesky_tei.")
        for cls in (self.method, *options.levels):
            if not (issubclass(cls, (CCBase, MP2)) or cls in (CRCC23, AdaptiveCCPQ)):
                raise NotImplementedError(f"CIM supports ground-state CC methods, MP2, CR-CC(2,3), and adaptive "
                                          f"CC(P;Q), not {cls.__name__}.")

        # LMOs of the correlated occupied space and their Fock matrix
        nfrozen, nocc = ref.nfrozen, ref.nfrozen + self.system.noccupied_alpha
        C = ref.mo_coeff["a"]
        U = boys_localize(C[:, nfrozen:nocc], mol, options.localization_tol, options.localization_maxiter)
        fock = U.T @ np.diag(ref.mo_energy["a"][nfrozen:nocc]) @ U
        # every occupied LMO, localized core first (core LMOs only enter the PAO projectors)
        L = np.hstack([C[:, :nfrozen] @ boys_localize(C[:, :nfrozen], mol, options.localization_tol,
                                                      options.localization_maxiter) if nfrozen else C[:, :0],
                       C[:, nfrozen:nocc] @ U])
        all_populations = lmo_populations(L, mol)
        populations = all_populations[nfrozen:]
        anchors = hydrogen_anchors(mol, options.bond_tolerance)
        if options.domains is None:
            self.domains = CIMDomains(secim_subsystems(fock, populations, anchors, options, self.method), populations)
            subsystems = self.domains.subsystems
        else:
            self.domains = options.domains
            lmo = match_lmos(options.domains.populations, populations)
            subsystems = [replace(s, central=tuple(lmo[list(s.central)]), occupied=tuple(lmo[list(s.occupied)]))
                          for s in options.domains.subsystems]
        if any(s.extended is None for s in self.domains.subsystems):
            # extended domains belong to the domains, so fixed-domain copies reuse them
            extended = [s.extended or buffer_atoms(mol, s.atoms, anchors, options.buffer_radius)
                        for s in self.domains.subsystems]
            self.domains = replace(self.domains, subsystems=tuple(
                replace(s, extended=e) for s, e in zip(self.domains.subsystems, extended)))
            subsystems = [replace(s, extended=e) for s, e in zip(subsystems, extended)]
        multiplicity = np.zeros(len(fock))
        for s in subsystems:
            multiplicity[list(s.central)] += 1

        S = mol.ints_overlap()
        F_ao = S @ (C * ref.mo_energy["a"]) @ C.T @ S
        occupied_domains = domain_atoms(all_populations, options.domain_charge)
        atom_aos = [np.arange(*mol.basis.center_first_and_last[a]) for a in range(mol.natoms)]
        basis_data, aux_data = mol.basis.serialize(), mol.auxiliary_basis.serialize()
        nvirt = C.shape[1] - nocc
        self.correlation_energy, self.subsystems = 0.0, []
        print(f"\n   CIM: {len(subsystems)} subsystems, zeta = {options.zeta}")
        for n, s in enumerate(subsystems):
            occupied = list(s.occupied)
            # occupied LMOs of the extended domain, the subsystem's first, projected onto its AOs
            # (least squares) and orthonormalized
            aos = np.concatenate([atom_aos[a] for a in s.extended])
            inside = [k for k in range(L.shape[1]) if occupied_domains[k] <= set(s.extended)]
            order = [nfrozen + i for i in occupied] + [k for k in inside if k - nfrozen not in set(occupied)]
            S_d = S[np.ix_(aos, aos)]
            A = np.linalg.solve(S_d, S[aos] @ L[:, order])
            lam, V = np.linalg.eigh(A.T @ S_d @ A)
            A = A @ (V / np.sqrt(lam)) @ V.T
            # unoccupied space of the extended domain: the exact orthogonal complement of its
            # projected LMOs (the span of its PAOs), in canonically orthonormalized AOs
            s_ao, U_ao = np.linalg.eigh(S_d)
            keep = s_ao > mol.overlap_tol * s_ao.max()
            X = U_ao[:, keep] / np.sqrt(s_ao[keep])
            a = X.T @ S_d @ A
            lam, Y = np.linalg.eigh(np.eye(a.shape[0]) - a @ a.T)
            V = X @ Y[:, lam > 0.5]
            # the part of it on the AO-domain atoms: orbitals whose projection onto their AOs
            # has the weight w with ||phi - projection|| = sqrt(1 - w) < eta (step 12, rotation
            # invariant)
            p = np.flatnonzero(np.isin(aos, np.concatenate([atom_aos[a] for a in s.atoms])))
            SP = S_d[:, p]
            lam, Y = np.linalg.eigh(V.T @ SP @ np.linalg.solve(S_d[np.ix_(p, p)], SP.T) @ V)
            V = V @ Y[:, lam > 1.0 - options.eta**2]
            # QCMOs of the subsystem occupied orbitals and of the PAOs
            F_d = F_ao[np.ix_(aos, aos)]
            O = A[:, :len(occupied)]
            eps_o, W = np.linalg.eigh(O.T @ F_d @ O)
            eps_v, Z = np.linalg.eigh(V.T @ F_d @ V)
            if eps_v[0] < eps_o[-1]:
                raise ValueError(f"CIM subsystem {n + 1}: an unoccupied orbital lies below the occupied ones "
                                 f"({eps_v[0]:.3f} < {eps_o[-1]:.3f} hartree), i.e. occupied orbitals of atoms "
                                 "outside its extended domain leak into it; increase buffer_radius.")
            child = subsystem_reference(ref, np.hstack([O @ W, V @ Z]), aos, s.extended, F_d, len(occupied),
                                        basis_data, aux_data)
            # central projector sum_i' R_ii' R_ki' / M_i' in the QCMO basis, R = W^T
            c = np.arange(len(s.central))
            P = W[c].T @ (W[c] / multiplicity[list(s.central)][:, None])

            stdout = contextlib.nullcontext() if options.verbose else contextlib.redirect_stdout(io.StringIO())
            with stdout:
                if issubclass(s.method, MP2):
                    energy = mp2_central_energy(child.H, P)
                else:
                    kwargs = dict(self.user_options, **(self.method_options if s.method is self.method else {}))
                    calc = s.method(child, **kwargs)
                    if s.method is CRCC23:
                        # CCSD, HBar, and left-CCSD only: the full correction of the
                        # subsystem is never needed, just its central part
                        cc, left = calc.cc.ensure_solved(), calc.left.ensure_solved()
                        hbar = calc.hbar
                        energy = cc_central_energy(cc.T, child.H, P) + crcc23_central_correction(
                            hbar.T, left.L, hbar.H, child.H, child.system, P)[calc.variant]
                    elif s.method is AdaptiveCCPQ:
                        # each subsystem selects its own P space (in its QCMO basis), ranking the
                        # triples (and quadruples) by their contributions to its central energy; its
                        # CC(P;Q) energy is CC(P) plus CC(P;3)_D (and the CC(P;4)_D quadruples part
                        # for target="ccsdtq") on the P space of the last macroiteration
                        if calc.stochastic is not None:
                            raise NotImplementedError("CIM with semi-stochastic adaptive CC(P;Q) is not supported.")
                        calc.central_projector = P
                        calc.ensure_solved()
                        hbar = calc.cc.get_hbar(HBarCCSD)
                        energy = cc_central_energy(calc.cc.T, child.H, P) + crcc23_central_correction(
                            hbar.T, calc.left.L, hbar.H, child.H, child.system, P,
                            t3_excitations=calc.cc.t3_excitations)["D"]
                        if calc.target == "ccsdtq":
                            from ccpy.methods.hbar import HBarCCSDLegacy
                            from ccpy.moments.ccp34 import calc_ccp4_with_selection

                            hbar4 = calc.cc.get_hbar(HBarCCSDLegacy)
                            energy += calc_ccp4_with_selection(hbar4.T, calc.left.L, calc.cc.t4_excitations, child.H,
                                                               hbar4.H, child.system, use_RHF=True, central=P)[1]
                    else:
                        calc.ensure_solved()
                        energy = cc_central_energy(calc.T, child.H, P)
            self.correlation_energy += energy
            self.subsystems.append(dict(method=s.method.__name__, ncentral=len(s.central), noccupied=len(occupied),
                                        nunoccupied=V.shape[1], natoms=len(s.atoms), energy=energy))
            print(f"   {n + 1:4d}  {s.method.__name__:>8s}  central {len(s.central):3d}  occupied {len(occupied):4d}"
                  f"  virtual {V.shape[1]:5d} / {nvirt}  atoms {len(s.atoms):3d}  dE = {energy:.10f}")
        print(f"   CIM correlation energy = {self.correlation_energy:.12f}")
        print(f"   CIM total energy = {self.reference_energy + self.correlation_energy:.12f}\n")
        self._finish_solve(True)
        return self
