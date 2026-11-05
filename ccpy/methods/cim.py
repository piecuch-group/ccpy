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
  that level, and are not merged into lower-level subsystems that contain them.

Each subsystem is a Reference in which the occupied LMOs outside the subsystem are frozen
and the unoccupied orbitals outside it are dropped, so every method runs unchanged:

    e = CRCC23(mf, nfrozen=1, cim=CIMOptions(zeta=0.003)).energy()
    e = MP2(mf, nfrozen=1, cim=CIMOptions(levels={CRCC23: [0, 1]})).energy()

Differences from the papers: the occupied LMOs of a subsystem are the exact LMOs and its
unoccupied orbitals are the projected atomic orbitals (PAOs) of the atoms of its AO domain,
taken from the full virtual space, so every subsystem space is exactly orthonormal and the
canonical calculation is recovered when a subsystem contains all orbitals; X-H bonds use
covalent radii (+ ``bond_tolerance``). Closed-shell RHF references only.
"""
import contextlib
import io
import warnings
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from ccpy.methods.base import METHOD_OPTIONS, SOLVER_OPTIONS, MethodBase
from ccpy.methods.hamiltonian import DFHamiltonian, build_fock, determinant_energy
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
    pao_threshold : smallest kept overlap eigenvalue of the normalized PAOs of a subsystem
    levels : {method class: atom indices}, higher levels than the method CIM is called on, the
             highest first; hydrogen atoms follow the atom they are bonded to
    domains : CIMDomains of another geometry to reuse (fixed-domain CIM)
    verbose : print the output of the subsystem calculations
    """

    zeta: float = 0.003
    central_charge: float = 0.3
    domain_charge: float = 1.98
    bond_tolerance: float = 0.168
    pao_threshold: float = 1.0e-6
    levels: dict = field(default_factory=dict)
    domains: "CIMDomains" = None
    localization_tol: float = 1.0e-12
    localization_maxiter: int = 1000
    verbose: bool = False


@dataclass(frozen=True)
class CIMSubsystem:
    """One subsystem: LMO indices (central ones first in ``occupied``), AO-domain atoms, method."""

    central: tuple
    occupied: tuple
    atoms: tuple
    method: type


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


def subsystem_reference(parent, Q, nfrozen):
    """Reference in the orbitals Q (parent MOs x child MOs) of a subsystem: the first nfrozen
    columns are frozen, the remaining occupied ones (Q spans the occupied space in its first
    columns) are active, and the rest are the active virtuals."""
    mol = parent.molecule
    child = object.__new__(Reference)
    child.scf, child.nfrozen, child.ndelete = parent.scf, nfrozen, 0
    child.is_uhf, child.rhf_symmetry = False, parent.rhf_symmetry
    C = parent.mo_coeff["a"] @ Q
    h = Q.T @ parent.h_mo["a"] @ Q
    B = np.asfortranarray(np.einsum("Ppq,pi,qj->Pij", parent.B_mo["a"], Q, Q, optimize=True))
    child.mo_coeff, child.h_mo, child.B_mo = {"a": C, "b": C}, {"a": h, "b": h}, {"a": B, "b": B}
    nocc = parent.nfrozen + parent.system.noccupied_alpha
    F, _ = build_fock(h, h, B, B, nocc, nocc)
    eps = np.diag(F).copy()
    child.mo_energy = {"a": eps, "b": eps.copy()}
    norb = Q.shape[1]
    occupation = np.zeros(norb)
    occupation[:nocc] = 2.0
    child.system = System(mol.nelectrons, norb, mol.multiplicity, nfrozen, charge=mol.charge,
                          nuclear_repulsion=mol.nuclear_repulsion, reference_energy=parent.energy,
                          frozen_energy=determinant_energy(h, h, B, B, nfrozen, nfrozen), cholesky=True,
                          mo_energies=eps, mo_occupation=occupation)
    energy = mol.nuclear_repulsion + determinant_energy(h, h, B, B, nocc, nocc)
    if not np.isclose(energy, parent.energy, atol=1.0e-8, rtol=0):
        raise RuntimeError("CIM subsystem orbitals changed the reference determinant energy.")
    corr = slice(nfrozen, norb)
    child.H = DFHamiltonian(child.system, F[corr, corr], F[corr, corr], B[:, corr, corr], B[:, corr, corr])
    child._legacy_H = None
    child.fno_result = child.fno_parent = None
    child.orbital_transform = {"a": Q, "b": Q}
    return child


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
    """Central MP2 energy in the QCMO basis (f_ia = 0 for HF orbitals)."""
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


def crcc23_central_correction(T, L, H, H0, system, P):
    """Central part sum_i' P_i'i' dE_i'(2,3), Eqs. (16), (21), and (22), of all CR-CC(2,3)
    variants, closed-shell RHF.

    The kernels of ccpy.moments.crcc23_chol sum l_ijk^abc M^ijk_abc / 36 with M linear in
    its inputs (H-bar blocks times T2) and l built from separate inputs. Replacing the moment
    inputs by their occupied derivatives, one factor at a time, gives
    sum l (P_1 + P_2 + P_3) M / 36 = 3 sum_i'k' P_i'k' dE_k'(2,3)."""
    from ccpy.lib.core import crcc_loops
    from ccpy.moments.crcc23_chol import (aaa_H3_aaa_diagonal, aab_H3_aab_diagonal, abb_H3_abb_diagonal,
                                          get_vvvv_diagonal)

    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    h_aa_vvvv, h_ab_vvvv, _ = get_vvvv_diagonal(H, T)
    noa, nua = system.noccupied_alpha, system.nunoccupied_alpha
    nob, nub = system.noccupied_beta, system.nunoccupied_beta

    #### aaa correction ####
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    # moment inputs: H(vooo), I(vvov), T2
    H2A_vooo_P = occupied_derivative(H.aa.vooo, P, lower=(2, 3), upper=(1,))
    I2A_vvov_P = occupied_derivative(I2A_vvov, P, lower=(2,))
    t2a_P = occupied_derivative(T.aa, P, lower=(2, 3))
    d_aaa = np.zeros(4)
    for vooo, vvov, t2a in ((H2A_vooo_P, I2A_vvov_P, T.aa), (H.aa.vooo, I2A_vvov, t2a_P)):
        d_aaa += crcc_loops.crcc23a_opt(
            t2a, L.a, L.aa,
            vooo, vvov, H.aa.oovv, H.a.ov,
            H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
            H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
            h_aa_vvvv,
            d3aaa_o, d3aaa_v,
            noa, nua,
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
        d_aab += crcc_loops.crcc23b_opt(
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
            noa, nua, nob, nub,
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

    Supported methods: ground-state CC (CCSD, CCSDT, CC(P), CC3, ...), MP2, and CR-CC(2,3)
    (all variants). Gradients are finite differences with the subsystems fixed at the
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
        from ccpy.methods.corrections import CRCC23
        from ccpy.methods.mbpt import MP2
        from ccpy.scf.rhf import RHF

        ref, mol, options = self.reference, self.molecule, self.cim
        if not isinstance(ref.scf, RHF) or not ref.rhf_symmetry or mol.point_group != "C1" or ref.ndelete:
            raise NotImplementedError("CIM needs a closed-shell RHF reference without symmetry or deleted orbitals.")
        for cls in (self.method, *options.levels):
            if not (issubclass(cls, (CCBase, MP2)) or cls is CRCC23):
                raise NotImplementedError(f"CIM supports ground-state CC methods, MP2, and CR-CC(2,3), not {cls.__name__}.")

        # LMOs of the correlated occupied space and their Fock matrix
        nfrozen, nocc = ref.nfrozen, ref.nfrozen + self.system.noccupied_alpha
        C = ref.mo_coeff["a"]
        U = boys_localize(C[:, nfrozen:nocc], mol, options.localization_tol, options.localization_maxiter)
        fock = U.T @ np.diag(ref.mo_energy["a"][nfrozen:nocc]) @ U
        populations = lmo_populations(C[:, nfrozen:nocc] @ U, mol)
        if options.domains is None:
            self.domains = CIMDomains(secim_subsystems(fock, populations, hydrogen_anchors(mol, options.bond_tolerance),
                                                       options, self.method), populations)
            subsystems = self.domains.subsystems
        else:
            self.domains = options.domains
            lmo = match_lmos(options.domains.populations, populations)
            subsystems = [replace(s, central=tuple(lmo[list(s.central)]), occupied=tuple(lmo[list(s.occupied)]))
                          for s in options.domains.subsystems]
        multiplicity = np.zeros(len(fock))
        for s in subsystems:
            multiplicity[list(s.central)] += 1

        # PAOs of all AOs in the virtual MO basis
        S = mol.ints_overlap()
        paos = C[:, nocc:].T @ S
        nvirt = C.shape[1] - nocc
        self.correlation_energy, self.subsystems = 0.0, []
        print(f"\n   CIM: {len(subsystems)} subsystems, zeta = {options.zeta}")
        for n, s in enumerate(subsystems):
            occupied = list(s.occupied)
            # QCMOs of the occupied LMOs and of the PAOs of the AO domain
            eps_o, W = np.linalg.eigh(fock[np.ix_(occupied, occupied)])
            X = paos[:, np.concatenate([np.arange(*mol.basis.center_first_and_last[a]) for a in s.atoms])]
            X = X / np.linalg.norm(X, axis=0)
            lam, V = np.linalg.eigh(X.T @ X)
            V = X @ V[:, lam > options.pao_threshold] / np.sqrt(lam[lam > options.pao_threshold])
            eps_v, Z = np.linalg.eigh(V.T @ (ref.mo_energy["a"][nocc:, None] * V))
            V = V @ Z
            # parent MOs -> [frozen core, other LMOs | subsystem occupied QCMOs | subsystem virtual QCMOs]
            others = [i for i in range(len(fock)) if i not in set(occupied)]
            Q = np.zeros((C.shape[1], nocc + V.shape[1]))
            Q[:nfrozen, :nfrozen] = np.eye(nfrozen)
            Q[nfrozen:nocc, nfrozen:nfrozen + len(others)] = U[:, others]
            Q[nfrozen:nocc, nocc - len(occupied):nocc] = U[:, occupied] @ W
            Q[nocc:, nocc:] = V
            child = subsystem_reference(ref, Q, nocc - len(occupied))
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
