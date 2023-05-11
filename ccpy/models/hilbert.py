import numpy as np
from ccpy.energy.hf_energy import calc_hf_energy_unsorted


class Excitation:

    def __init__(self, particles_alpha, holes_alpha, particles_beta, holes_beta, system, phase = 1.0, bit_length=32):

        assert(len(particles_alpha) == len(holes_alpha))
        assert(len(particles_beta) == len(holes_beta))

        self.to_alpha = [a - 1 for a in particles_alpha]
        self.to_beta = [a - 1 for a in particles_beta]
        self.from_alpha = [i - 1 for i in holes_alpha]
        self.from_beta = [i - 1 for i in holes_beta]

        self.phase = phase

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
            print([str(i + 1) + 'A' for i in self.from_alpha], "->", [str(i) + 'A' for i in self.to_alpha])
        if self.from_beta and self.to_beta:
            print([str(i + 1) + 'B' for i in self.from_beta], "->", [str(i) + 'B' for i in self.to_beta])
        print("Spincase:", self.spincase, ",", "Symmetry:", self.symmetry)
        print("Phase = ", self.phase)
        return ""

    def get_bitstring(self, bit_length):

        def _get_int_index(x):
            return int(np.floor(x / bit_length))

        num_alpha = self.spincase.count('a')
        num_beta = self.spincase.count('b')
        for n in range(num_alpha):
            a = self.to_alpha[n]
            i = self.from_alpha[n]
            self.bitstring[0][_get_int_index(i)] += 2 ** (i % bit_length)  # hole alpha
            self.bitstring[2][_get_int_index(a)] += 2 ** (a % bit_length)  # particle alpha
        for n in range(num_beta):
            a = self.to_beta[n]
            i = self.from_beta[n]
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
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a]]
            sym_val = sym_val ^ orb_sym_val

        for n in range(num_beta):
            a = self.to_beta[n]
            i = self.from_beta[n]
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[i]]
            sym_val = sym_val ^ orb_sym_val
            orb_sym_val = system.point_group_irrep_to_number[system.orbital_symmetries[a]]
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
            a = self.to_alpha[n] + 1 + system.noccupied_alpha
            i = self.from_alpha[n] + 1
            idx = occupation.index(2 * i - 1)
            occupation[idx] = 2 * a - 1

        for n in range(num_beta):
            a = self.to_beta[n]  + 1 + system.noccupied_beta
            i = self.from_beta[n] + 1
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

        self.occ_a = np.array(sorted([i - 1 for i in occupied_alpha]))
        self.unocc_a = np.array(sorted([i - 1 for i in range(1, system.norbitals + 1) if i not in occupied_alpha]))
        self.occ_b = np.array(sorted([i - 1 for i in occupied_beta]))
        self.unocc_b = np.array(sorted([i - 1 for i in range(1, system.norbitals + 1) if i not in occupied_beta]))

        self.symmetry = ''
        self.get_symmetry(system)

    def __repr__(self):
        print("Occupation:", self.occupation)
        print("Occupied alpha:", [i +1 for i in self.occ_a])
        print("Unoccupied alpha:", [a + 1 for a in self.unocc_a])
        print("Occupied beta:", [i + 1 for i in self.occ_b])
        print("Unoccupied beta:", [a + 1 for a in self.unocc_b])
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
        # is there a phase associated with this excitation???

        holes = []
        particles = []
        num_alpha_1 = 0
        num_beta_1 = 0
        num_alpha_2 = 0
        num_beta_2 = 0

        nperm = 0

        for p in self.occupation:
            if p not in other_det.occupation:
                particles.append(p)
                if p % 2 == 0:
                    idx, = np.where(self.occ_b == self.spin_to_spatial(p) - 1)
                    nperm += len(self.occ_b) - idx[0] + 1 + num_beta_1
                    num_beta_1 += 1
                else:
                    idx, = np.where(self.occ_a == self.spin_to_spatial(p) - 1)
                    nperm += len(self.occ_a) - idx[0] + 1 + num_alpha_1
                    num_alpha_1 += 1

        for q in other_det.occupation:
            if q not in self.occupation:
                holes.append(q)
                if q % 2 == 0:
                    idx, = np.where(other_det.occ_b == self.spin_to_spatial(q) - 1)
                    nperm += len(other_det.occ_b) - idx[0] + 1 + num_beta_2
                    num_beta_2 += 1
                else:
                    idx, = np.where(other_det.occ_a == self.spin_to_spatial(q) - 1)
                    nperm += len(other_det.occ_a) - idx[0] + 1 + num_alpha_2
                    num_alpha_2 += 1

        assert(num_alpha_1 == num_alpha_2)
        assert(num_beta_1 == num_beta_2)

        holes_alpha = [self.spin_to_spatial(i) for i in holes if i % 2 == 1]
        holes_beta = [self.spin_to_spatial(i) for i in holes if i % 2 == 0]
        particles_alpha = [self.spin_to_spatial(a) for a in particles if a % 2 == 1]
        particles_beta = [self.spin_to_spatial(a) for a in particles if a % 2 == 0]

        return Excitation(particles_alpha, holes_alpha, particles_beta, holes_beta, system, (-1.0)**nperm)

    @staticmethod
    def spin_to_spatial(x):
        if x % 2 == 1:
            return x // 2 + 1
        else:
            return x // 2

def slater_eval(H, idet, jdet, system):

    exc = idet.get_excitation(jdet, system)

    val = 0.0
    if exc.degree > 2:
        return val
    elif exc.degree == 2:
        if exc.spincase == 'aa':
            val = H.aa[exc.to_alpha[0], exc.to_alpha[1], exc.from_alpha[0], exc.from_alpha[1]]
        elif exc.spincase == 'ab':
            val = H.ab[exc.to_alpha[0], exc.to_beta[0], exc.from_alpha[0], exc.from_beta[0]]
        elif exc.spincase == 'bb':
            val = H.bb[exc.to_beta[0], exc.to_beta[1], exc.from_beta[0], exc.from_beta[1]]
    elif exc.degree == 1:
        if exc.spincase == 'a':
            val = H.a[exc.to_alpha[0], exc.from_alpha[0]]
            val += np.einsum("ii->", np.squeeze(H.aa[np.ix_(np.array([exc.to_alpha[0]]), jdet.occ_a, np.array([exc.from_alpha[0]]), jdet.occ_a)]))
            val += np.einsum("ii->", np.squeeze(H.ab[np.ix_(np.array([exc.to_alpha[0]]), jdet.occ_b, np.array([exc.from_alpha[0]]), jdet.occ_b)]))
        elif exc.spincase == 'b':
            val = H.b[exc.to_beta[0], exc.from_beta[0]]
            val += np.einsum("ii->", np.squeeze(H.bb[np.ix_(np.array([exc.to_beta[0]]), jdet.occ_b, np.array([exc.from_beta[0]]), jdet.occ_b)]))
            val += np.einsum("ii->", np.squeeze(H.ab[np.ix_(jdet.occ_a, np.array([exc.to_beta[0]]), jdet.occ_a, np.array([exc.from_beta[0]]))]))
    else:
        val = np.einsum("ii->", H.a[np.ix_(jdet.occ_a, jdet.occ_a)])
        val += np.einsum("ii->", H.b[np.ix_(jdet.occ_b, jdet.occ_b)])
        val += 0.5 * np.einsum("ijij->", H.aa[np.ix_(jdet.occ_a, jdet.occ_a, jdet.occ_a, jdet.occ_a)])
        val += np.einsum("ijij->", H.ab[np.ix_(jdet.occ_a, jdet.occ_b, jdet.occ_a, jdet.occ_b)])
        val += 0.5 * np.einsum("ijij->", H.bb[np.ix_(jdet.occ_b, jdet.occ_b, jdet.occ_b, jdet.occ_b)])

    return val * exc.phase


if __name__ == "__main__":
    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -1.2491088779
                F 0.0 0.0  1.2491088779""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = load_pyscf_integrals(mf, nfrozen, sorted=False, normal_ordered=False)

    system.print_info()

    # d1 = Determinant([1, 2, 3, 4, 5, 6, 11], [1, 2, 23, 4, 22, 6], system)
    # print(d1)
    # d2 = Determinant([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 22, 6], system)
    # print(d2)
    #
    # exc = d1.get_excitation(d2, system)
    # print(exc)

    singles = {'a' : [], 'b' : []}
    doubles = {'aa' : [], 'ab' : [], 'bb' : []}

    HF = Determinant([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], system)
    e_ref = calc_hf_energy_unsorted(H, HF.occ_a, HF.occ_b)

    for a in range(system.noccupied_alpha, system.norbitals):
        for i in range(system.noccupied_alpha):
            occ_a = [1, 2, 3, 4, 5, 6, 7]
            occ_b = [1, 2, 3, 4, 5, 6, 7]
            occ_a[i] = a + 1
            singles['a'].append(Determinant(occ_a, occ_b, system))

    for a in range(system.noccupied_beta, system.norbitals):
        for i in range(system.noccupied_beta):
            occ_a = [1, 2, 3, 4, 5, 6, 7]
            occ_b = [1, 2, 3, 4, 5, 6, 7]
            occ_b[i] = a + 1
            singles['b'].append(Determinant(occ_a, occ_b, system))


    H_SASA = np.zeros((len(singles['a']), len(singles['a'])))
    for idx, idet in enumerate(singles['a']):
        for jdx, jdet in enumerate(singles['a']):
            H_SASA[idx, jdx] = slater_eval(H, idet, jdet, system)

    H_SASB = np.zeros((len(singles['a']), len(singles['b'])))
    for idx, idet in enumerate(singles['a']):
        for jdx, jdet in enumerate(singles['b']):
            H_SASB[idx, jdx] = slater_eval(H, idet, jdet, system)

    H_SBSA = np.zeros((len(singles['b']), len(singles['a'])))
    for idx, idet in enumerate(singles['b']):
        for jdx, jdet in enumerate(singles['a']):
            H_SBSA[idx, jdx] = slater_eval(H, idet, jdet, system)

    H_SBSB = np.zeros((len(singles['b']), len(singles['b'])))
    for idx, idet in enumerate(singles['b']):
        for jdx, jdet in enumerate(singles['b']):
            H_SBSB[idx, jdx] = slater_eval(H, idet, jdet, system)

    H_CIS = np.vstack((np.hstack((H_SASA, H_SASB)),
                       np.hstack((H_SBSA, H_SBSB))))
    E, V = np.linalg.eigh(H_CIS)
    idx = np.argsort(E)
    e_test = E[idx]
    V = V[:, idx]

    system, H = load_pyscf_integrals(mf, nfrozen, sorted=True, normal_ordered=True)

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta

    H_SASA = np.zeros((n1a, n1a))
    idx = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            jdx = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    H_SASA[idx, jdx] = (i == j) * H.a.vv[a, b] - (a == b) * H.a.oo[j, i] + H.aa.voov[a, j, i, b]
                    jdx += 1
            idx += 1

    H_SASB = np.zeros((n1a, n1b))
    idx = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            jdx = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    H_SASB[idx, jdx] = H.ab.voov[a, j, i, b]
                    jdx += 1
            idx += 1

    H_SBSA = np.zeros((n1b, n1a))
    idx = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            jdx = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    H_SBSA[idx, jdx] = H.ab.ovvo[j, a, b, i]
                    jdx += 1
            idx += 1

    H_SBSB = np.zeros((n1b, n1b))
    idx = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            jdx = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    H_SBSB[idx, jdx] = (i == j) * H.b.vv[a, b] - (a == b) * H.b.oo[j, i] + H.bb.voov[a, j, i, b]
                    jdx += 1
            idx += 1


    H_CIS = np.vstack((np.hstack((H_SASA, H_SASB)),
                       np.hstack((H_SBSA, H_SBSB))))

    E, V = np.linalg.eigh(H_CIS)
    idx = np.argsort(E)
    e_true = E[idx]
    V = V[:, idx]

    for iroot in range(len(e_true)):
        print("eigenvalue", iroot + 1, "Expected = ", e_true[iroot], "Got = ", e_test[iroot])


