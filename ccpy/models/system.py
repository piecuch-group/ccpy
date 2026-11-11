from ccpy.utilities.printing import SystemPrinter, UHFSystemPrinter
from ccpy.utilities.symmetry import get_pg_irreps

# [TODO]: For single-reference methods, this should really be called "Reference"
class System:
    """Class that holds information about the molecular or periodic system."""

    def __init__(
        self,
        nelectrons,
        norbitals,
        multiplicity,
        nfrozen,
        point_group="C1",
        orbital_symmetries=None,
        ndelete=0,
        charge=0,
        nkpts=0,
        reference_energy=0.0,
        frozen_energy=0.0,
        nuclear_repulsion=0.0,
        mo_energies=None,
        mo_occupation=None,
        nact_occupied=0,
        nact_unoccupied=0,
        cholesky=False,
    ):
        # basic information
        self.nelectrons = nelectrons - 2 * nfrozen
        self.norbitals = norbitals - nfrozen - ndelete
        self.nfrozen = nfrozen
        self.ndelete = ndelete
        self.multiplicity = multiplicity
        self.charge = charge
        # orbital partitioning
        self.noccupied_alpha = int((self.nelectrons + self.multiplicity - 1) / 2)
        self.noccupied_beta = int((self.nelectrons - self.multiplicity + 1) / 2)
        self.nunoccupied_alpha = self.norbitals - self.noccupied_alpha
        self.nunoccupied_beta = self.norbitals - self.noccupied_beta
        # active space orbital partitioning
        self.set_active_space(nact_occupied, nact_unoccupied)
        # symmetry information
        self.point_group = point_group
        self.point_group_irrep_to_number = get_pg_irreps(self.point_group)
        self.point_group_number_to_irrep = {v: k for k, v in self.point_group_irrep_to_number.items()}
        if orbital_symmetries is None:
            self.orbital_symmetries_all = ["A"] * norbitals
        else:
            self.orbital_symmetries_all = orbital_symmetries
        # MO energies and occupation
        if mo_occupation is None:
            mo_occupation = [2.0] * nfrozen +\
                            [2.0] * self.noccupied_beta +\
                            [1.0] * (self.noccupied_alpha - self.noccupied_beta) +\
                            [0.0] * (self.norbitals - self.noccupied_alpha)

            assert len(mo_occupation) == self.norbitals + nfrozen, "Occupation vector has wrong size"
            self.mo_occupation = mo_occupation
        else:
            self.mo_occupation = mo_occupation
        self.mo_energies = mo_energies
        # reference energies
        self.reference_energy = reference_energy
        self.frozen_energy = frozen_energy
        self.nuclear_repulsion = nuclear_repulsion
        # periodic system information
        self.nkpts = nkpts
        # Cholesky integrals
        self.cholesky = cholesky

        # Get the point group symmetry of the reference state by exploiting
        # homomorphism between Abelian groups and binary vector spaces
        # sym( irrep1, irrep2 ) = xor( irrep1, irrep2 ), where irreps are
        # numbered in the convention (for D2H):
        # Ag = 0, B1g = 1, B2g = 2, B3g = 3, Au = 4, B1u = 5, B2u = 6, B3u = 7
        sym = 0
        for i in range(self.nfrozen + self.noccupied_alpha):
            for j in range(int(self.mo_occupation[i])): # does this extend to fractional occupation numbers (e.g., NOs)?
               sym = sym ^ self.point_group_irrep_to_number[self.orbital_symmetries_all[i]]
        self.reference_symmetry = self.point_group_number_to_irrep[sym]

        # once we've found the reference irrep, we don't need the frozen orbital irreps anymore.
        self.orbital_symmetries = self.orbital_symmetries_all[self.nfrozen:]

    def set_active_space(self, nact_occupied=0, nact_unoccupied=0, active_occupied=None, active_unoccupied=None):
        set_active_space(self, nact_occupied, nact_unoccupied, active_occupied, active_unoccupied)

    def set_full_active_space(self):
        self.set_active_space(nact_occupied=self.noccupied_alpha, nact_unoccupied=self.nunoccupied_beta)

    def print_info(self):
        SystemPrinter(self).header()

class UHFSystem:
    """Class that holds information about the molecular or periodic system using UHF reference."""

    def __init__(
        self,
        nelectrons,
        norbitals,
        multiplicity,
        nfrozen,
        point_group="C1",
        orbital_symmetries=None,
        ndelete=0,
        charge=0,
        nkpts=0,
        reference_energy=0.0,
        frozen_energy=0.0,
        nuclear_repulsion=0.0,
        mo_energies=None,
        mo_occupation=None,
        nact_occupied=0,
        nact_unoccupied=0,
        cholesky=False,
    ):
        # basic information
        self.nelectrons = nelectrons - 2 * nfrozen
        self.norbitals = norbitals - nfrozen - ndelete
        self.nfrozen = nfrozen
        self.ndelete = ndelete
        self.multiplicity = multiplicity
        self.charge = charge
        # orbital partitioning
        self.noccupied_alpha = int((self.nelectrons + self.multiplicity - 1) / 2)
        self.noccupied_beta = int((self.nelectrons - self.multiplicity + 1) / 2)
        self.nunoccupied_alpha = self.norbitals - self.noccupied_alpha
        self.nunoccupied_beta = self.norbitals - self.noccupied_beta
        # active space orbital partitioning
        self.set_active_space(nact_occupied, nact_unoccupied)
        # symmetry information
        self.point_group = point_group
        self.point_group_irrep_to_number = get_pg_irreps(self.point_group)
        self.point_group_number_to_irrep = {v: k for k, v in self.point_group_irrep_to_number.items()}
        if orbital_symmetries is None:
            self.orbital_symmetries_all = {"a" : ["A"] * norbitals, "b": ["A"] * norbitals}
        else:
            self.orbital_symmetries_all = orbital_symmetries
        # MO energies and occupation
        self.mo_occupation = mo_occupation
        self.mo_energies = mo_energies
        # reference energies
        self.reference_energy = reference_energy
        self.frozen_energy = frozen_energy
        self.nuclear_repulsion = nuclear_repulsion
        # periodic system information
        self.nkpts = nkpts
        # Cholesky integrals
        self.cholesky = cholesky

        # Get the point group symmetry of the reference state by exploiting
        # homomorphism between Abelian groups and binary vector spaces
        # sym( irrep1, irrep2 ) = xor( irrep1, irrep2 ), where irreps are
        # numbered in the convention (for D2H):
        # Ag = 0, B1g = 1, B2g = 2, B3g = 3, Au = 4, B1u = 5, B2u = 6, B3u = 7
        sym = 0
        for i in range(self.nfrozen + self.noccupied_alpha):
            for j in range(int(self.mo_occupation["a"][i])): # does this extend to fractional occupation numbers (e.g., NOs)?
               sym = sym ^ self.point_group_irrep_to_number[self.orbital_symmetries_all["a"][i]]
            for j in range(int(self.mo_occupation["b"][i])): # does this extend to fractional occupation numbers (e.g., NOs)?
               sym = sym ^ self.point_group_irrep_to_number[self.orbital_symmetries_all["b"][i]]
        self.reference_symmetry = self.point_group_number_to_irrep[sym]

        # once we've found the reference irrep, we don't need the frozen orbital irreps anymore.
        #self.orbital_symmetries = {"a": self.orbital_symmetries_all["a"][self.nfrozen:],
        #                           "b": self.orbital_symmetries_all["b"][self.nfrozen:]}
        self.orbital_symmetries = self.orbital_symmetries_all["a"][self.nfrozen:]

    def set_active_space(self, nact_occupied=0, nact_unoccupied=0, active_occupied=None, active_unoccupied=None):
        set_active_space(self, nact_occupied, nact_unoccupied, active_occupied, active_unoccupied)

    def print_info(self):
        UHFSystemPrinter(self).header()


def set_active_space(system, nact_occupied, nact_unoccupied, active_occupied=None, active_unoccupied=None):
    """Set the active orbitals used by CCSDt-type P spaces (get_active_*_pspace) and active-space methods.

    By default the active orbitals are the nact_occupied highest occupied and nact_unoccupied lowest unoccupied
    orbitals (plus the singly occupied ones for open shells). active_occupied / active_unoccupied instead give
    the active orbitals as 1-based MO numbers, counted from the first MO including frozen ones (as printed
    by the SCF), e.g. active_occupied=[3, 5], active_unoccupied=[6, 7]; the counts are then taken from them.

    The results are system.active_occupied_alpha/beta and system.active_unoccupied_alpha/beta: 0-based
    indices into the correlated occupied and unoccupied orbitals of each spin. system.active_space_contiguous
    is False when they are not the default block; slice-based active-space code refuses such a space.
    """
    noa, nob, nua, nub = system.noccupied_alpha, system.noccupied_beta, system.nunoccupied_alpha, system.nunoccupied_beta
    nf = system.nfrozen
    somo = list(range(nob, noa))  # singly occupied orbitals, 0-based in the correlated space; always active
    if active_occupied is None and active_unoccupied is None:
        nacto_a = min(noa, (system.multiplicity - 1) + nact_occupied)
        nacto_b = min(nob, nact_occupied)
        nactu_a = min(nua, nact_unoccupied)
        nactu_b = min(nub, (system.multiplicity - 1) + nact_unoccupied)
        system.active_occupied_alpha = list(range(noa - nacto_a, noa))
        system.active_occupied_beta = list(range(nob - nacto_b, nob))
        system.active_unoccupied_alpha = list(range(nactu_a))
        system.active_unoccupied_beta = list(range(nactu_b))
    else:
        occ = sorted({p - 1 - nf for p in (active_occupied or [])})
        unocc = sorted({p - 1 - nf for p in (active_unoccupied or [])})
        if any(p < 0 or p >= nob for p in occ):
            raise ValueError(f"active_occupied must be doubly occupied, correlated MOs {nf + 1}..{nf + nob}; got {active_occupied}")
        if any(p < noa or p >= noa + nua for p in unocc):
            raise ValueError(f"active_unoccupied must be unoccupied MOs {nf + noa + 1}..{nf + noa + nua}; got {active_unoccupied}")
        system.active_occupied_alpha = sorted(occ + somo)
        system.active_occupied_beta = occ
        system.active_unoccupied_alpha = [p - noa for p in unocc]
        system.active_unoccupied_beta = sorted(p - nob for p in somo + unocc)
    system.num_act_occupied_alpha = len(system.active_occupied_alpha)
    system.num_act_occupied_beta = len(system.active_occupied_beta)
    system.num_act_unoccupied_alpha = len(system.active_unoccupied_alpha)
    system.num_act_unoccupied_beta = len(system.active_unoccupied_beta)
    system.num_core_alpha = noa - system.num_act_occupied_alpha
    system.num_core_beta = nob - system.num_act_occupied_beta
    system.num_virt_alpha = nua - system.num_act_unoccupied_alpha
    system.num_virt_beta = nub - system.num_act_unoccupied_beta
    system.active_space_contiguous = (
        system.active_occupied_alpha == list(range(system.num_core_alpha, noa))
        and system.active_occupied_beta == list(range(system.num_core_beta, nob))
        and system.active_unoccupied_alpha == list(range(system.num_act_unoccupied_alpha))
        and system.active_unoccupied_beta == list(range(system.num_act_unoccupied_beta)))
