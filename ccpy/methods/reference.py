"""Reference determinant of a correlated calculation: the converged SCF, the orbital
partitioning (System), and the bare DF Hamiltonian in the correlated MO basis.

Downstream of the SCF, methods only see the bare DF vectors (``reference.H.chol``), the
T1-transformed DF vectors and the HBar of ccpy.methods.hbar.
"""
import copy
import warnings

import numpy as np

from ccpy.methods.hamiltonian import DFHamiltonian, build_fock, determinant_energy
from ccpy.models.system import System, UHFSystem


class DFPortWarning(UserWarning):
    """Issued by methods whose equations still need four-index integrals (including vvvv),
    which are then materialized from the DF vectors. These methods must be ported to DF."""


class Reference:
    """Converged SCF reference with its DF Hamiltonian.

    Parameters
    ----------
    scf : ccpy.scf.RHF | ccpy.scf.ROHF | ccpy.scf.UHF
        Run here if it has not been run yet.
    nfrozen : int
        Number of frozen core spatial orbitals.
    ndelete : int
        Number of deleted (highest) virtual orbitals.
    """

    def __init__(self, scf, nfrozen=0, ndelete=0):
        from ccpy.scf.scf_base import SCFBase
        from ccpy.scf.uhf import UHF

        if not isinstance(scf, SCFBase):
            raise TypeError("Expected a ccpy.scf RHF, ROHF, or UHF object.")
        if scf.e_tot is None:
            scf.run()
        if not scf.converged:
            print("   WARNING: building the reference from an unconverged SCF solution.")
        self.scf = scf
        self.nfrozen = nfrozen
        self.ndelete = ndelete
        self.is_uhf = isinstance(scf, UHF)
        mol = scf.mol

        if self.is_uhf:
            C_a, C_b = scf.mo_coeff
            norbitals = C_a.shape[1]
            self.system = UHFSystem(
                mol.nelectrons, norbitals, mol.multiplicity, nfrozen, ndelete=ndelete,
                point_group=mol.point_group,
                orbital_symmetries={"a": scf.orbital_symmetries(0), "b": scf.orbital_symmetries(1)},
                charge=mol.charge, nuclear_repulsion=mol.nuclear_repulsion,
                mo_energies={"a": scf.mo_energy[0], "b": scf.mo_energy[1]},
                mo_occupation={"a": scf.mo_occ[0], "b": scf.mo_occ[1]},
            )
        else:
            C_a = C_b = scf.mo_coeff
            norbitals = C_a.shape[1]
            self.system = System(
                mol.nelectrons, norbitals, mol.multiplicity, nfrozen, ndelete=ndelete,
                point_group=mol.point_group, orbital_symmetries=scf.orbital_symmetries(),
                charge=mol.charge, nuclear_repulsion=mol.nuclear_repulsion,
                mo_energies=scf.mo_energy, mo_occupation=scf.mo_occ,
            )
        system = self.system
        system.cholesky = True

        # full MO basis (frozen core and deleted virtuals included), per spin
        hcore = scf.get_hcore()
        eps_a, eps_b = scf.mo_energy if self.is_uhf else (scf.mo_energy, scf.mo_energy)
        self.mo_coeff = {"a": C_a, "b": C_b}
        self.mo_energy = {"a": eps_a, "b": eps_b}
        self.h_mo = {"a": C_a.T @ hcore @ C_a, "b": C_b.T @ hcore @ C_b}
        B_a = mol.df.mo_cholesky_vectors(C_a)
        B_b = B_a if C_b is C_a else mol.df.mo_cholesky_vectors(C_b)
        self.B_mo = {"a": B_a, "b": B_b}
        h_a, h_b = self.h_mo["a"], self.h_mo["b"]
        nocc_a = nfrozen + system.noccupied_alpha
        nocc_b = nfrozen + system.noccupied_beta
        f_a, f_b = build_fock(h_a, h_b, B_a, B_b, nocc_a, nocc_b)

        e_ref = mol.nuclear_repulsion + determinant_energy(h_a, h_b, B_a, B_b, nocc_a, nocc_b)
        if not np.isclose(e_ref, scf.e_tot, atol=1.0e-06, rtol=0.0):
            raise RuntimeError(f"DF integrals don't match the SCF energy ({e_ref:.10f} vs {scf.e_tot:.10f})")
        system.reference_energy = e_ref
        system.frozen_energy = determinant_energy(h_a, h_b, B_a, B_b, nfrozen, nfrozen)

        corr = slice(nfrozen, nfrozen + system.norbitals)
        self.H = DFHamiltonian(system, f_a[corr, corr], f_b[corr, corr],
                               B_a[:, corr, corr], B_b[:, corr, corr])
        # closed-shell RHF: alpha and beta amplitudes are identical
        self.rhf_symmetry = (not self.is_uhf) and system.noccupied_alpha == system.noccupied_beta
        self._legacy_H = None
        self.fno_result = None
        self.fno_parent = None
        self.orbital_transform = None

    def from_orbital_transform(self, transforms, orbital_symmetries):
        """Fresh reference under real rectangular, occupied-preserving MO transforms.

        Matrices include frozen orbitals and act on all stored MO data. This
        factory never reads the SCF coefficients to reconstruct the child basis.
        Both spins must have the same total dimension, as required by System.
        """
        if self.ndelete:
            raise ValueError("Orbital transformation requires a reference without deleted virtuals.")
        sizes = []
        for s, no in (("a", self.system.noccupied_alpha), ("b", self.system.noccupied_beta)):
            Q = np.asarray(transforms[s])
            occupied = no + self.nfrozen
            if (Q.ndim != 2 or Q.shape[0] != self.mo_coeff[s].shape[1] or
                    Q.shape[1] < occupied or np.iscomplexobj(Q) or not np.all(np.isfinite(Q))):
                raise ValueError("Invalid real orbital transform.")
            expected = np.eye(Q.shape[0])[:, :occupied]
            if not np.array_equal(Q[:, :occupied], expected):
                raise ValueError("Orbital transforms must leave every occupied orbital unchanged.")
            if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1.0e-10, rtol=0):
                raise ValueError("Orbital transforms must have orthonormal columns.")
            if len(orbital_symmetries[s]) != Q.shape[1]:
                raise ValueError("Orbital symmetry labels do not match the transformed basis.")
            sizes.append(Q.shape[1])
        if sizes[0] != sizes[1]:
            raise ValueError("CCpy requires equal total alpha/beta orbital dimensions.")
        if not self.is_uhf and not np.array_equal(transforms["a"], transforms["b"]):
            raise ValueError("Restricted references require the same spatial orbital transform.")
        child = object.__new__(type(self))
        child.scf, child.nfrozen, child.ndelete = self.scf, self.nfrozen, 0
        child.is_uhf, child.rhf_symmetry = self.is_uhf, self.rhf_symmetry
        child.mo_coeff, child.h_mo, child.B_mo, child.mo_energy = {}, {}, {}, {}
        for s in ("a", "b"):
            Q = transforms[s]
            child.mo_coeff[s] = self.mo_coeff[s] @ Q
            child.h_mo[s] = Q.T @ self.h_mo[s] @ Q
            child.B_mo[s] = np.asfortranarray(np.einsum("Ppq,pi,qj->Pij", self.B_mo[s], Q, Q, optimize=True))
        na = self.nfrozen + self.system.noccupied_alpha
        nb = self.nfrozen + self.system.noccupied_beta
        Fa, Fb = build_fock(child.h_mo["a"], child.h_mo["b"], child.B_mo["a"], child.B_mo["b"], na, nb)
        if self.is_uhf:
            child.mo_energy = {"a": np.diag(Fa).copy(), "b": np.diag(Fb).copy()}
        else:
            from ccpy.scf.rohf import ROHF

            ca, cb = (ROHF.CANONICALIZATIONS[self.scf.canonicalization][2]
                      if isinstance(self.scf, ROHF) else (0.5, 0.5))
            eps = np.diag(ca * Fa + cb * Fb).copy()
            child.mo_energy = {"a": eps, "b": eps.copy()}
        for s, no in (("a", na), ("b", nb)):
            child.mo_energy[s][:no] = self.mo_energy[s][:no]
        if not self.is_uhf:
            child.mo_energy["b"] = child.mo_energy["a"].copy()
        mol = self.molecule
        occupations = {s: np.zeros(sizes[0]) for s in ("a", "b")}
        occupations["a"][:na] = 1
        occupations["b"][:nb] = 1
        kwargs = dict(point_group=self.system.point_group, charge=mol.charge,
                      nuclear_repulsion=mol.nuclear_repulsion, reference_energy=self.energy,
                      frozen_energy=self.system.frozen_energy, cholesky=True,
                      nact_occupied=self.system.num_act_occupied_beta,
                      nact_unoccupied=self.system.num_act_unoccupied_alpha)
        if self.is_uhf:
            child.system = UHFSystem(mol.nelectrons, sizes[0], mol.multiplicity, self.nfrozen,
                                     orbital_symmetries=orbital_symmetries,
                                     mo_energies=child.mo_energy, mo_occupation=occupations, **kwargs)
        else:
            child.system = System(mol.nelectrons, sizes[0], mol.multiplicity, self.nfrozen,
                                  orbital_symmetries=orbital_symmetries["a"],
                                  mo_energies=child.mo_energy["a"],
                                  mo_occupation=occupations["a"] + occupations["b"], **kwargs)
        energy = mol.nuclear_repulsion + determinant_energy(child.h_mo["a"], child.h_mo["b"],
                                                            child.B_mo["a"], child.B_mo["b"], na, nb)
        if not np.isclose(energy, self.energy, atol=1.0e-9, rtol=0):
            raise RuntimeError("Orbital transform changed the reference determinant energy.")
        corr = slice(self.nfrozen, sizes[0])
        child.H = DFHamiltonian(child.system, Fa[corr, corr], Fb[corr, corr],
                                 child.B_mo["a"][:, corr, corr], child.B_mo["b"][:, corr, corr])
        child._legacy_H = None
        child.fno_result = child.fno_parent = None
        child.orbital_transform = {s: np.array(transforms[s], copy=True) for s in ("a", "b")}
        return child

    def one_body_from_ao(self, operator):
        """Transform an AO operator (optionally with leading component axes).

        Returns per-spin matrices in the correlated working basis, including
        the FNO transform when present. AO dimensions come from coefficients.
        """
        return {s: np.einsum("pi,...pq,qj->...ij", C[:, self.nfrozen:self.nfrozen + self.system.norbitals],
                             operator, C[:, self.nfrozen:self.nfrozen + self.system.norbitals], optimize=True)
                for s, C in self.mo_coeff.items()}

    def one_body_operator_from_ao(self, operator):
        """One AO matrix as an Integral bound to this working reference for LR-CC."""
        from ccpy.models.integrals import Integral

        operator = np.asarray(operator)
        nao = self.mo_coeff["a"].shape[0]
        if operator.shape != (nao, nao) or np.iscomplexobj(operator) or not np.all(np.isfinite(operator)):
            raise ValueError("A one-body operator must be a finite real AO matrix.")
        result = Integral(self.system, 1, self.one_body_from_ao(operator))
        result.reference = self
        return result

    @property
    def molecule(self):
        return self.scf.mol

    @property
    def energy(self):
        """Reference (SCF) energy."""
        return self.system.reference_energy

    def with_molecule(self, molecule):
        """Reference of the same SCF type and settings for another Molecule."""
        if self.fno_result is not None:
            from ccpy.methods.fno import prepare_fno

            return prepare_fno(self.fno_parent.with_molecule(molecule), self.fno_result.options)
        return Reference(self.scf.with_molecule(molecule), self.nfrozen, self.ndelete)

    def legacy_hamiltonian(self, copy_=True):
        """Four-index Hamiltonian (ccpy.models.integrals.Integral, vvvv included) formed from
        the DF vectors, for methods that have not been ported to DF yet. Cached; a deep copy
        is returned unless copy_=False, since several legacy routines overwrite it."""
        if self._legacy_H is None:
            from ccpy.interfaces.scf_tools import getUHFHamiltonianDF
            from ccpy.models.integrals import getHamiltonian

            system = self.system
            if self.is_uhf:
                C_a, C_b = self.mo_coeff["a"], self.mo_coeff["b"]
                self._legacy_H = getUHFHamiltonianDF(C_a, C_b, self.scf.get_hcore(), self.molecule.df,
                                                     system, normal_ordered=True)
            else:
                B = self.B_mo["a"]
                e2int = np.asfortranarray(np.einsum("xpr,xqs->pqrs", B, B, optimize=True))
                self._legacy_H = getHamiltonian(self.h_mo["a"], e2int, system, normal_ordered=True)
        return copy.deepcopy(self._legacy_H) if copy_ else self._legacy_H


def warn_df_port(name):
    warnings.warn(f"{name} still uses four-index integrals (vvvv included) built from the DF "
                  "vectors; its equations need to be ported to DF.", DFPortWarning, stacklevel=3)
