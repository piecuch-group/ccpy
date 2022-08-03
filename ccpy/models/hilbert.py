import numpy as np

class Excitation:

    def __init__(self, particles_alpha, holes_alpha, particles_beta, holes_beta, system, bit_length=32):

        assert(len(particles_alpha) == len(holes_alpha))
        assert(len(particles_beta) == len(holes_beta))

        self.to_alpha = particles_alpha
        self.to_beta = particles_beta
        self.from_alpha = holes_alpha
        self.from_beta = holes_beta

        self.spincase = 'a' * len(particles_alpha) + 'b' * len(particles_beta)
        self.degree = len(self.spincase)

        self.N_int = int(np.floor((system.norbitals - 1) / bit_length) + 1)
        self.bitstring = tuple(([0 for i in range(self.N_int)],  # hole alpha
                                [0 for i in range(self.N_int)],  # hole beta
                                [0 for i in range(self.N_int)],  # particle alpha
                                [0 for i in range(self.N_int)]))  # particle beta
        self.get_bitstring(bit_length)

        self.symmetry = ''
        self.get_symmetry(system)

    def __repr__(self):

        if self.from_alpha and self.to_alpha:
            print([str(i) + 'A' for i in self.from_alpha], "->", [str(i) + 'A' for i in self.to_alpha])
        if self.from_beta and self.to_beta:
            print([str(i) + 'B' for i in self.from_beta], "->", [str(i) + 'B' for i in self.to_beta])
        print("Spincase:", self.spincase, ",", "Symmetry:", self.symmetry)
        return ""

    def get_bitstring(self, bit_length):

        def _get_int_index(x):
            return int(np.floor(x / bit_length))

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')
        for n in range(num_alpha):
            a = self.to_alpha[n] - 1
            i = self.from_alpha[n] - 1
            self.bitstring[0][_get_int_index(i)] += 2 ** (i % bit_length)  # hole alpha
            self.bitstring[2][_get_int_index(a)] += 2 ** (a % bit_length)  # particle alpha
        for n in range(num_beta):
            a = self.to_beta[n] - 1
            i = self.from_beta[n] - 1
            self.bitstring[1][_get_int_index(i)] += 2 ** (i % bit_length)  # hole beta
            self.bitstring[3][_get_int_index(a)] += 2 ** (a % bit_length)  # particle beta

    def get_symmetry(self, system):
        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')

        # this assumes that excitation originates from HF. This may not always be the case!
        #sym_val = system.point_group_irrep_to_number[system.reference_symmetry]
        sym_val = 0

        for n in range(num_alpha):
            a = self.to_alpha[n]
            i = self.from_alpha[n]
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i - 1]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a - 1]]
            sym_val = sym_val ^ orb_sym_val

        for n in range(num_beta):
            a = self.to_beta[n]
            i = self.from_beta[n]
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i - 1]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a - 1]]
            sym_val = sym_val ^ orb_sym_val

        self.symmetry = system.point_group_number_to_irrep[sym_val]

    def get_relative_excitation(self, other_excitation, system):

        return self.get_determinant(system).get_excitation(other_excitation.get_determinant(system), system)

    def get_determinant(self, system, reference=None):

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')

        if reference is None:
            reference = [i for i in range(1, system.nelectrons + 1)] # Hartree-Fock

        occupation = [i for i in reference]

        for n in range(num_alpha):
            a = self.to_alpha[n] + system.noccupied_alpha
            i = self.from_alpha[n]
            idx = occupation.index(2 * i - 1)
            occupation[idx] = 2 * a - 1

        for n in range(num_beta):
            a = self.to_beta[n] + system.noccupied_beta
            i = self.from_beta[n]
            idx = occupation.index(2 * i)
            occupation[idx] = 2 * a

        return Determinant([self.spin_to_spatial(i) for i in occupation if i % 2 == 1],
                           [self.spin_to_spatial(i) for i in occupation if i % 2 == 0],
                           system)

    @staticmethod
    def spin_to_spatial(x):
        if x % 2 == 1:
            return x // 2 + 1
        else:
            return x // 2

class Determinant:

    def __init__(self, occupied_alpha, occupied_beta, system):

        assert(len(occupied_alpha) + len(occupied_beta) == system.nelectrons)
        assert(len(occupied_alpha) == system.noccupied_alpha)
        assert(len(occupied_beta) == system.noccupied_beta)

        self.sz = 0.5 * (len(occupied_alpha) - len(occupied_beta))
        self.multiplicity = int(2 * abs(self.sz)) + 1

        self.occupation = sorted([2 * i - 1 for i in occupied_alpha]
                                +[2 * i for i in occupied_beta]
                                 )

        self.occ_a = np.array(occupied_alpha)
        self.unocc_a = np.array([i for i in range(system.norbitals) if i not in occupied_alpha])
        self.occ_b = np.array(occupied_beta)
        self.unocc_b = np.array([i for i in range(system.norbitals) if i not in occupied_beta])

        self.symmetry = ''
        self.get_symmetry(system)

    def __repr__(self):
        print("Occupation:", self.occupation)
        print("Occupied alpha:", self.occ_a)
        print("Unoccupied alpha:", self.unocc_a)
        print("Occupied beta:", self.occ_b)
        print("Unoccupied beta:", self.unocc_b)
        print("Symmetry:", self.symmetry, ",", "Multiplicity:", self.multiplicity)
        return ""

    def get_symmetry(self, system):
        sym_val = 0 # identity in binary
        for p in self.occupation:
            sym_val = sym_val ^ system.point_group_irrep_to_number[system.orbital_symmetries[self.spin_to_spatial(p) - 1]]
        self.symmetry = system.point_group_number_to_irrep[sym_val]

    def get_excitation(self, other_det, system):
        # returns an excitation that expresses d1 (this determinant) as an excitation relative
        # to d2 (other determinant)
        # | d1 > = E_d2^d1 | d2 >

        holes = []
        particles = []
        num_alpha_1 = 0
        num_beta_1 = 0
        num_alpha_2 = 0
        num_beta_2 = 0
        for p in self.occupation:
            if p not in other_det.occupation:
                particles.append(p)
                if p % 2 == 0:
                    num_beta_1 += 1
                else:
                    num_alpha_1 += 1

        for q in other_det.occupation:
            if q not in self.occupation:
                holes.append(q)
                if q % 2 == 0:
                    num_beta_2 += 1
                else:
                    num_alpha_2 += 1

        assert(num_alpha_1 == num_alpha_2)
        assert(num_beta_1 == num_beta_2)

        holes_alpha = [self.spin_to_spatial(i) for i in holes if i % 2 == 1]
        holes_beta = [self.spin_to_spatial(i) for i in holes if i % 2 == 0]
        particles_alpha = [self.spin_to_spatial(a) for a in particles if a % 2 == 1]
        particles_beta = [self.spin_to_spatial(a) for a in particles if a % 2 == 0]

        return Excitation(particles_alpha, holes_alpha, particles_beta, holes_beta, system)

    @staticmethod
    def spin_to_spatial(x):
        if x % 2 == 1:
            return x // 2 + 1
        else:
            return x // 2


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

    d1 = Determinant([1, 2, 3, 4, 5, 6, 11], [1, 2, 9, 4, 22, 6], system)
    print(d1)
    d2 = Determinant([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 22, 6], system)
    print(d2)

    exc = d1.get_excitation(d2, system)
    print(exc)