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
    ):
        """Default constructor for the System object.

        Arguments:
        ----------
        nelectrons : int -> total number of electrons
        norbitals : int -> total number of spatial orbitals
        multiplicity : int -> spin multiplicity of reference state (2S+1)
        nfrozen : int -> number of frozen spatial orbitals
        point_group : str -> spatial point group, default = 'C1'
        orbital_symmetries : list -> point group irreps of each molecular orbital, default = [None]*norbitals
        charge : int -> total charge on the molecule, default = 0
        nkpts : int -> number of k-points used in reciprocal space sampling (periodic calculations only)
        e_nuclear : float -> nuclear repulsion energy at the molecular geometry (in hartree)
        Returns:
        ----------
        sys : Object -> System object"""
        self.nelectrons = nelectrons - 2 * nfrozen
        self.norbitals = norbitals - nfrozen
        self.nfrozen = nfrozen
        self.multiplicity = multiplicity
        self.noccupied_alpha = int((self.nelectrons + self.multiplicity - 1) / 2)
        self.noccupied_beta = int((self.nelectrons - self.multiplicity + 1) / 2)
        self.nunoccupied_alpha = self.norbitals - self.noccupied_alpha
        self.nunoccupied_beta = self.norbitals - self.noccupied_beta
        self.charge = charge
        self.nkpts = nkpts
        self.point_group = point_group
        self.point_group_irrep_to_number = get_pg_irreps(self.point_group)
        self.point_group_number_to_irrep = {v: k for k, v in self.point_group_irrep_to_number.items()}
        if orbital_symmetries is None:
            self.orbital_symmetries_all = ["A1"] * norbitals
        else:
            self.orbital_symmetries_all = orbital_symmetries
        self.reference_energy = reference_energy
        self.nuclear_repulsion = nuclear_repulsion
        self.mo_energies = mo_energies
        self.mo_occupation = mo_occupation

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
