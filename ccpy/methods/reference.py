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

        # full MO basis (frozen core and deleted virtuals included)
        hcore = scf.get_hcore()
        self.mo_coeff = (C_a, C_b)
        self.h_mo = (C_a.T @ hcore @ C_a, C_b.T @ hcore @ C_b)
        B_a = mol.df.mo_cholesky_vectors(C_a)
        B_b = B_a if C_b is C_a else mol.df.mo_cholesky_vectors(C_b)
        self.B_mo = (B_a, B_b)
        nocc_a = nfrozen + system.noccupied_alpha
        nocc_b = nfrozen + system.noccupied_beta
        f_a, f_b = build_fock(*self.h_mo, B_a, B_b, nocc_a, nocc_b)

        e_ref = mol.nuclear_repulsion + determinant_energy(*self.h_mo, B_a, B_b, nocc_a, nocc_b)
        if not np.isclose(e_ref, scf.e_tot, atol=1.0e-06, rtol=0.0):
            raise RuntimeError(f"DF integrals don't match the SCF energy ({e_ref:.10f} vs {scf.e_tot:.10f})")
        system.reference_energy = e_ref
        system.frozen_energy = determinant_energy(*self.h_mo, B_a, B_b, nfrozen, nfrozen)

        corr = slice(nfrozen, nfrozen + system.norbitals)
        self.H = DFHamiltonian(system, f_a[corr, corr], f_b[corr, corr],
                               B_a[:, corr, corr], B_b[:, corr, corr])
        # closed-shell RHF: alpha and beta amplitudes are identical
        self.rhf_symmetry = (not self.is_uhf) and system.noccupied_alpha == system.noccupied_beta
        self._legacy_H = None

    @property
    def molecule(self):
        return self.scf.mol

    @property
    def energy(self):
        """Reference (SCF) energy."""
        return self.system.reference_energy

    def with_molecule(self, molecule):
        """Reference of the same SCF type and settings for another Molecule."""
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
                C_a, C_b = self.mo_coeff
                self._legacy_H = getUHFHamiltonianDF(C_a, C_b, self.scf.get_hcore(), self.molecule.df,
                                                     system, normal_ordered=True)
            else:
                B = self.B_mo[0]
                e2int = np.asfortranarray(np.einsum("xpr,xqs->pqrs", B, B, optimize=True))
                self._legacy_H = getHamiltonian(self.h_mo[0], e2int, system, normal_ordered=True)
        return copy.deepcopy(self._legacy_H) if copy_ else self._legacy_H


def warn_df_port(name):
    warnings.warn(f"{name} still uses four-index integrals (vvvv included) built from the DF "
                  "vectors; its equations need to be ported to DF.", DFPortWarning, stacklevel=3)
