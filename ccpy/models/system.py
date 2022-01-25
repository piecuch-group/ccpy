class System:
    """Class that holds information about the molecular or periodic system."""
    def __init__(self, nelectrons, norbitals, multiplicity, nfrozen, point_group='C1',orbital_symmetries=None,charge=0):
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
        Returns:
        ----------
        sys : Object -> System object"""
        self.nelectrons = nelectrons - 2*nfrozen
        self.norbitals = norbitals - nfrozen
        self.multiplicity = multiplicity
        self.noccupied_alpha = (self.nelectrons + (self.multiplicity -1)) // 2
        self.noccupied_beta = (self.nelectrons + (self.multiplicity - 1)) // 2
        self.nunoccupied_alpha = self.norbitals - self.noccupied_alpha
        self.nunoccupied_beta = self.norbitals - self.noccupied_beta
        self.charge = charge
        self.point_group = point_group
        if orbital_symmetries is None:
            self.orbital_symmetries = ['A1'] * norbitals
        else:
            self.orbital_symmetries = orbital_symmetries

    def __repr__(self):
        for key,value in vars(self).items():
            print('     ',key,'->',value)
        return ''

    @classmethod
    def fromGamessFile(cls, gamessFile, nfrozen):
        """Builds the System object using the SCF information contained within a
        GAMESS log file.

        Arguments:
        ----------
        gamessFile : str -> Path to GAMESS log file
        nfrozen : int -> number of frozen electrons
        Returns:
        ----------
        sys : Object -> System object"""
        import cclib
        data = cclib.io.ccread(gamessFile)
        return cls(data.nelectrons,
                   data.nmo,
                   data.mult,
                   nfrozen,
                   cls.getGamessPointGroup(gamessFile),
                   data.mosyms[0],
                   data.charge)

    @classmethod
    def fromPyscfMolecular(cls, meanFieldObj, nfrozen):
        """Builds the System object using the information contained within a PySCF
        mean-field object for a molecular system.

        Arguments:
        ----------
        meanFieldObj : Object -> PySCF SCF/mean-field object
        nfrozen : int -> number of frozen electrons
        Returns:
        ----------
        sys : Object -> System object"""
        return cls(meanFieldObj.mol.nelectron,
                meanFieldObj.mo_coeff.shape[1],
                2*meanFieldObj.mol.spin + 1,
                nfrozen,
                meanFieldObj.mol.symmetry,
                [meanFieldObj.mol.irrep_name[x] for x in meanFieldObj.orbsym],
                meanFieldObj.mol.charge)

    #[TODO] Interface to periodic system calculations in PySCF (supercell framework only)
    # @classmethod
    # def fromPyscfPeriodicSuperCell(cls, periodicMeanFieldObj, nfrozen):
    #     """Builds the System object using the information contained within a PySCF
    #     mean-field object for a periodic system run using a supercell."""

    @staticmethod
    def getGamessPointGroup(gamessFile):
        """Dumb way of getting the point group from GAMESS log files.

        Arguments:
        ----------
        gamessFile : str -> Path to GAMESS log file
        Returns:
        ----------
        point_group : str -> Molecular point group"""
        point_group = 'C1'
        flag_found = False
        with open(gamessFile, 'r') as f:
            for line in f.readlines():
                if flag_found:
                    order = line.split()[-1]
                    if len(point_group) == 3:
                        point_group = point_group[0] + order + point_group[2]
                    if len(point_group) == 2:
                        point_group = point_group[0] + order
                    if len(point_group) == 1:
                        point_group = point_group[0] + order
                    break
                if 'THE POINT GROUP OF THE MOLECULE IS' in line:
                    point_group = line.split()[-1]
                    flag_found = True
        return point_group

if __name__ == "__main__":

    test_case = 'gamess'

    if test_case == 'native':
        # H2O / cc-pVDZ
        nfrozen = 1
        norbitals = 24
        nelectrons = 10
        multiplicity = 1
        sys = System(nelectrons, norbitals, multiplicity, nfrozen)

    if test_case == 'gamess':
        # F2+ / 6-31G
        nfrozen = 2
        gamessFile = "/Users/harellab/Documents/ccpy/tests/F2+-1.0-631g/F2+-1.0-631g.log"
        sys = System.fromGamessFile(gamessFile, nfrozen)

    if test_case == 'pyscf':
        from pyscf import gto, scf
        mol = gto.Mole()
        mol.build(
            atom = '''F 0.0 0.0 -2.66816
                      F 0.0 0.0  2.66816''',
            basis = 'ccpvdz',
            charge = 0,
            spin = 0,
            symmetry = 'D2H',
            cart = True,
            unit = 'Bohr',
        )
        rhf = scf.HF(mol)
        rhf.kernel()

        nfrozen = 2
        sys = System.fromPyscfMolecular(rhf, nfrozen)

    print(sys)