from ccpy.utilities.printing import SystemPrinter


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
        if orbital_symmetries is None:
            self.orbital_symmetries = ["A1"] * norbitals
        else:
            self.orbital_symmetries = orbital_symmetries
        self.reference_energy = reference_energy
        self.nuclear_repulsion = nuclear_repulsion
        self.mo_energies = mo_energies
        self.mo_occupation = mo_occupation

    def print_info(self):
        SystemPrinter(self).header()
