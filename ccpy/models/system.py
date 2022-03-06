from ccpy.utilities.printing import SystemPrinter
from ccpy.utilities.symmetry_count import get_pg_irreps


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
        charge=0,
        nkpts=0,
        reference_energy=0,
        nuclear_repulsion=0,
        mo_energies=None,
        mo_occupation=None,
        num_act_holes_alpha = 0,
        num_act_particles_alpha = 0,
        num_act_holes_beta = 0,
        num_act_particles_beta = 0,
    ):
        # basic information
        self.nelectrons = nelectrons - 2 * nfrozen
        self.norbitals = norbitals - nfrozen
        self.nfrozen = nfrozen
        self.multiplicity = multiplicity
        self.charge = charge
        # orbital partitioning
        self.noccupied_alpha = int((self.nelectrons + self.multiplicity - 1) / 2)
        self.noccupied_beta = int((self.nelectrons - self.multiplicity + 1) / 2)
        self.nunoccupied_alpha = self.norbitals - self.noccupied_alpha
        self.nunoccupied_beta = self.norbitals - self.noccupied_beta
        # active space orbital partitioning
        self.num_act_occupied_alpha = min(self.noccupied_alpha, num_act_holes_alpha)
        self.num_core_alpha = max(0, self.noccupied_alpha - self.num_act_occupied_alpha)
        self.num_act_occupied_beta = min(self.noccupied_beta, num_act_holes_beta)
        self.num_core_beta = max(0, self.noccupied_beta - self.num_act_occupied_beta)
        self.num_act_unoccupied_alpha = min(self.nunoccupied_alpha, num_act_particles_alpha)
        self.num_virt_alpha = max(0, self.nunoccupied_alpha - self.num_act_unoccupied_alpha)
        self.num_act_unoccupied_beta = min(self.nunoccupied_beta, num_act_particles_beta)
        self.num_virt_beta = max(0, self.nunoccupied_alpha - self.num_act_unoccupied_beta)
        # symmetry information
        self.point_group = point_group
        self.point_group_irrep_to_number = get_pg_irreps(self.point_group)
        self.point_group_number_to_irrep = {v: k for k, v in self.point_group_irrep_to_number.items()}
        if orbital_symmetries is None:
            self.orbital_symmetries_all = ["A1"] * norbitals
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
        self.nuclear_repulsion = nuclear_repulsion
        # periodic system information
        self.nkpts = nkpts

        # Get the point group symmetry of the reference state by exploiting
        # homomorphism between Abelian groups and binary vector spaces
        # sym(irrep1, irrep2) = xor( irrep1, irrep2 ), where irrep's are numbered
        # in the convention (for D2H):
        # Ag = 0, B1g = 1, B2g = 2, B3g = 3, Au = 4, B1u = 5, B2u = 6, B3u = 7
        sym = 0
        for i in range(self.nfrozen + self.noccupied_alpha):
            for j in range(int(self.mo_occupation[i])):
               sym = sym ^ self.point_group_irrep_to_number[self.orbital_symmetries_all[i]]
        self.reference_symmetry = self.point_group_number_to_irrep[sym]

        # once we've found the reference irrep, we don't need the frozen orbital irreps anymore.
        self.orbital_symmetries = self.orbital_symmetries_all[self.nfrozen:]

    def print_info(self):
        SystemPrinter(self).header()
