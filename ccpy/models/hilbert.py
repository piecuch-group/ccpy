import numpy as np

from ccpy.models.operators import get_operator_name

class Determinant:

    def __init__(self, spatial_excitation, spincase, system, bit_length=32):

        self.excitation = spatial_excitation # [vir, occ], vir indices range from 0 - system.nunoccupied
        self.spincase = spincase
        self.excitation_degree = int(len(spatial_excitation) / 2)

        self.symmetry = 'C1'
        self.get_symmetry(system)

        self.spinorbital_occupation = [i for i in range(1, system.nelectrons + 1)]
        self.get_spinorbital_occupation(system)

        self.N_int = int(np.floor((system.norbitals - 1) / bit_length) + 1)
        self.bitstring = tuple(([0 for i in range(self.N_int)],  # hole alpha
                                [0 for i in range(self.N_int)],  # hole beta
                                [0 for i in range(self.N_int)],  # particle alpha
                                [0 for i in range(self.N_int)]))  # particle beta
        self.get_bitstring(bit_length)

    def get_symmetry(self, system):
        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')

        sym_val = system.point_group_irrep_to_number[system.reference_symmetry]
        for n in range(num_alpha):
            a = self.excitation[n] - 1
            i = self.excitation[n + self.excitation_degree] - 1
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a]]
            sym_val = sym_val ^ orb_sym_val

        for n in range(num_beta):
            a = self.excitation[self.excitation_degree - n - 1] - 1
            i = self.excitation[self.excitation_degree - n + self.excitation_degree - 1] - 1
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a]]
            sym_val = sym_val ^ orb_sym_val

        self.symmetry = system.point_group_number_to_irrep[sym_val]

    def get_bitstring(self, bit_length):

        def _get_int_index(x):
            return int(np.floor(x / bit_length))

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')
        for n in range(num_alpha):
            a = self.excitation[n] - 1
            i = self.excitation[n + self.excitation_degree] - 1
            self.bitstring[0][_get_int_index(i)] += 2 ** (i % bit_length)  # hole alpha
            self.bitstring[2][_get_int_index(a)] += 2 ** (a % bit_length)  # particle alpha
        for n in range(num_beta):
            a = self.excitation[self.excitation_degree - n - 1] - 1
            i = self.excitation[self.excitation_degree - n + self.excitation_degree - 1] - 1
            self.bitstring[1][_get_int_index(i)] += 2 ** (i % bit_length)  # hole beta
            self.bitstring[3][_get_int_index(a)] += 2 ** (a % bit_length)  # particle beta

    def get_spinorbital_occupation(self, system):

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')

        spinorb_occ = [i for i in range(1, system.nelectrons + 1)]

        for n in range(num_alpha):
            a = self.excitation[n] + system.noccupied_alpha
            i = self.excitation[n + self.excitation_degree]
            idx = spinorb_occ.index(2 * i - 1)
            spinorb_occ[idx] = 2 * a - 1

        for n in range(num_beta):
            a = self.excitation[self.excitation_degree - n - 1] + system.noccupied_beta
            i = self.excitation[self.excitation_degree - n + self.excitation_degree - 1]
            idx = spinorb_occ.index(2 * i)
            spinorb_occ[idx] = 2 * a

        self.spinorbital_occupation = sorted(spinorb_occ)


    def get_orbital_partitioning(self, system):
        # Numpy arrays can be used to slice!
        occ_a = [i for i in range(system.noccupied_alpha)]
        unocc_a = [i for i in range(system.noccupied_alpha, system.norbitals)]
        occ_b = [i for i in range(system.noccupied_beta)]
        unocc_b = [i for i in range(system.noccupied_beta, system.norbitals)]

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')

        for n in range(num_alpha):
            a = self.excitation[n] - 1
            i = self.excitation[n + self.excitation_degree] - 1
            idx = occ_a.index(i)
            occ_a[idx] = a + system.noccupied_alpha
            idx = unocc_a.index(a + system.noccupied_alpha)
            unocc_a[idx] = i

        for n in range(num_beta):
            a = self.excitation[self.excitation_degree - n - 1] - 1
            i = self.excitation[self.excitation_degree - n + self.excitation_degree - 1] - 1
            idx = occ_b.index(i)
            occ_b[idx] = a + system.noccupied_beta
            idx = unocc_b.index(a + system.noccupied_beta)
            unocc_b[idx] = i

        return np.array(sorted(occ_a)), np.array(sorted(unocc_a)), np.array(sorted(occ_b)), np.array(sorted(unocc_b))

class DeterminantalSubspace:

    def __init__(self, system, order, fill_level=2):
        self.order = order
        self.spin_cases = []
        self.dimensions = []

        ndim = 0
        for i in range(1, order + 1):
            for j in range(i + 1):
                name = get_operator_name(i, j)
                setattr(self, name, [])
                self.spin_cases.append(name)

        if fill_level >= 1:
            self.fill_singles(system)
        if fill_level >= 2:
            self.fill_doubles(system)
        if fill_level >= 3:
            self.fill_triples(system)

    def add_determinant(self, determinant):
        setattr(self, determinant.spincase, self.__getattribute__(determinant.spincase) + [determinant])

    # [TODO] add determinants of particular symmetry by filtering according to symmetry attribute
    def fill_singles(self, system):
        setattr(self, 'a', self.__getattribute__('a') +
                [Determinant([a, i], 'a', system) for a in range(system.nunoccupied_alpha) for i in
                 range(system.noccupied_alpha)])
        setattr(self, 'b', self.__getattribute__('b') +
                [Determinant([a, i], 'b', system) for a in range(system.nunoccupied_beta) for i in
                 range(system.noccupied_beta)])

    def fill_doubles(self, system):
        setattr(self, 'aa', self.__getattribute__('aa') +
                [Determinant([a, b, i, j], 'aa', system) for a in range(system.nunoccupied_alpha) for b in
                 range(a + 1, system.nunoccupied_alpha)
                 for i in range(system.noccupied_alpha) for j in range(i + 1, system.noccupied_alpha)])
        setattr(self, 'ab', self.__getattribute__('ab') +
                [Determinant([a, b, i, j], 'ab', system) for a in range(system.nunoccupied_alpha) for b in
                 range(system.nunoccupied_beta)
                 for i in range(system.noccupied_alpha) for j in range(system.noccupied_beta)])
        setattr(self, 'bb', self.__getattribute__('bb') +
                [Determinant([a, b, i, j], 'bb', system) for a in range(system.nunoccupied_beta) for b in
                 range(a + 1, system.nunoccupied_beta)
                 for i in range(system.noccupied_beta) for j in range(i + 1, system.noccupied_beta)])

    def fill_triples(self, system):
        setattr(self, 'aaa', self.__getattribute__('aaa') +
                [Determinant([a, b, c, i, j, k], 'aaa', system) for a in range(system.nunoccupied_alpha)
                 for b in range(a + 1, system.nunoccupied_alpha)
                 for c in range(b + 1, system.nunoccupied_alpha)
                 for i in range(system.noccupied_alpha)
                 for j in range(i + 1, system.noccupied_alpha)
                 for k in range(j + 1, system.noccupied_alpha)])
        setattr(self, 'aab', self.__getattribute__('aab') +
                [Determinant([a, b, c, i, j, k], 'aab', system) for a in range(system.nunoccupied_alpha)
                 for b in range(a + 1, system.nunoccupied_alpha)
                 for c in range(system.nunoccupied_beta)
                 for i in range(system.noccupied_alpha)
                 for j in range(i + 1, system.noccupied_alpha)
                 for k in range(system.noccupied_beta)])
        setattr(self, 'abb', self.__getattribute__('abb') +
                [Determinant([a, b, c, i, j, k], 'abb', system) for a in range(system.nunoccupied_alpha)
                 for b in range(system.nunoccupied_beta)
                 for c in range(b + 1, system.nunoccupied_beta)
                 for i in range(system.noccupied_alpha)
                 for j in range(system.noccupied_beta)
                 for k in range(j + 1, system.noccupied_beta)])
        setattr(self, 'bbb', self.__getattribute__('bbb') +
                [Determinant([a, b, c, i, j, k], 'bbb', system) for a in range(system.nunoccupied_beta)
                 for b in range(a + 1, system.nunoccupied_beta)
                 for c in range(b + 1, system.nunoccupied_beta)
                 for i in range(system.noccupied_beta)
                 for j in range(i + 1, system.noccupied_beta)
                 for k in range(j + 1, system.noccupied_beta)])


if __name__ == "__main__":
    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -1.2491088779
                F 0.0 0.0  1.2491088779""",
        basis="ccpvdz",
        charge=1,
        spin=1,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = load_pyscf_integrals(mf, nfrozen)

    system.print_info()

    d = Determinant([2, 3, 1, 2], 'ab', system)
    print(d.spinorbital_occupation)
    occ_a, unocc_a, occ_b, unocc_b = d.get_orbital_partitioning(system)
    print(occ_a)
    print(unocc_a)
    print(occ_b)
    print(unocc_b)

    # pspace = DeterminantalSubspace(system, 3, fill_level=2)
    #
    # pspace.add_determinant(Determinant([4, 5, 6, 1, 2, 3], 'abb', system))
    #
    # print(pspace.abb[0].excitation)
    # print(pspace.abb[0].spincase)
    # print(pspace.abb[0].symmetry)
    # print(pspace.abb[0].bitstring)
    #
    # # pspace.remove_determinant(Determinant([4, 5, 6, 1, 2, 3], 'abb', system))
    #
    # print(pspace.abb)
