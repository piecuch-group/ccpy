import numpy as np
from itertools import combinations_with_replacement

class ActiveOperator:

    def __init__(self, system, order, spincase, num_active, data_type=np.float64):
        self.order = order
        self.spincase = spincase
        self.num_active = num_active
        self.slices = []
        self.dimensions = []
        self.ndim = 0

        dims_alpha = {"o": system.noccupied_alpha - system.num_act_occupied_alpha,
                      "O": system.noccupied_alpha - (system.noccupied_alpha - system.num_act_occupied_alpha),
                      "V": system.noccupied_alpha + system.num_act_unoccupied_alpha - system.noccupied_alpha,
                      "v": system.norbitals - (system.noccupied_alpha + system.num_act_unoccupied_alpha)
        }
        dims_beta =  {"o": system.noccupied_beta - system.num_act_occupied_beta,
                      "O": system.noccupied_beta - (system.noccupied_beta - system.num_act_occupied_beta),
                      "V": system.noccupied_beta + system.num_act_unoccupied_beta - system.noccupied_beta,
                      "v": system.norbitals - (system.noccupied_beta + system.num_act_unoccupied_beta)
        }
        double_spin_string = list(self.spincase) * 2
        num_alpha_hole = double_spin_string[self.order : 2*self.order].count("a")
        num_alpha_particle = double_spin_string[: self.order].count("a")
        num_beta_particle = double_spin_string[: self.order].count("b")
        num_beta_hole = double_spin_string[self.order: 2 * self.order].count("b")
        for pa in self.get_particle_combinations(num_alpha_particle):
            dim_pa = [dims_alpha[x] for x in pa]
            for pb in self.get_particle_combinations(num_beta_particle):
                dim_pb = [dims_beta[x] for x in pb]
                for ha in self.get_hole_combinations( num_alpha_hole):
                    dim_ha = [dims_alpha[x] for x in ha]
                    for hb in self.get_hole_combinations(num_beta_hole):
                        dim_hb = [dims_beta[x] for x in hb]

                        dimensions = tuple(dim_pa + dim_pb + dim_ha + dim_hb)
                        temp = ''.join(pa + pb + ha + hb)
                        num_act_holes = temp.count('O')
                        num_act_particles = temp.count('V')
                        if num_act_holes >= num_active and num_act_particles >= num_active:
                            self.slices.append(temp)
                            self.dimensions.append(dimensions)
                            self.ndim += np.prod(dimensions)
                            setattr(self, temp, np.zeros(dimensions, dtype=data_type))


    def get_hole_combinations(self, n):
        combs = []
        for i in range(n + 1):
            temp = ['O'] * n
            for j in range(i):
                temp[j] = 'o'
            combs.append(temp)
        return combs

    def get_particle_combinations(self, n):
        combs = []
        for i in range(n + 1):
            temp = ['V'] * n
            for j in range(i):
                temp[n - 1 - j] = 'v'
            combs.append(temp)
        return combs

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.slices]
        )

    def unflatten(self, T_flat):
        prev = 0
        for dims, name in zip(self.dimensions, self.slices):
            ndim = np.prod(dims)
            setattr(self, name, np.reshape(T_flat[prev : ndim + prev], dims))
            prev += ndim


class ClusterOperator:
    def __init__(self, system, order, active_orders=[None], num_active=[None], data_type=np.float64):
        self.order = order
        self.spin_cases = []
        self.dimensions = []

        # [TODO]: think of a nicer way to handle the active order cases
        ndim = 0
        act_cnt = 0
        for i in range(1, order + 1):

            if i in active_orders:
                nact = num_active[act_cnt]
                act_cnt += 1

            for j in range(i + 1):
                name = get_operator_name(i, j)
                dimensions = get_operator_dimension(i, j, system)

                if i in active_orders:

                    active_t = ActiveOperator(system, i, name, nact, data_type=data_type)
                    setattr(self, name, active_t)
                    for dim in active_t.dimensions:
                        self.dimensions.append(dim)
                    ndim += active_t.ndim

                else:

                    setattr(self, name, np.zeros(dimensions, dtype=data_type))
                    self.dimensions.append(dimensions)
                    ndim += np.prod(dimensions)

                self.spin_cases.append(name)

        self.ndim = ndim

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.spin_cases]
        )

    def unflatten(self, T_flat):
        prev = 0
        for dims, name in zip(self.dimensions, self.spin_cases):
            ndim = np.prod(dims)
            setattr(self, name, np.reshape(T_flat[prev: ndim + prev], dims))
            prev += ndim


class FockOperator:
    """Builds generalized particle-nonconserving operators of the EA/IP-type and
    higher-order extensions, such as DEA/DIP, etc.
    Naming convention goes as follows:
        Suppose R.* has n letters. The first n - num_add indices, where num_add is the number of
        added particles (EA) or holes (IP), denote the particle-conserving
        part, while the last num_add refer to the added particles or holes. For example,
        R.aab in the DIP case (num_add=2) means that the first R.a part is an alpha-alpha singles
        block of dimension (nua, noa), and the last R.*ab part refers to the added holes,
        which are of alpha and beta character, respectively. Thus, the total dimension of
        R.aab in the DIP case is (nua, noa, noa, nob)."""
    def __init__(self, system, num_particles, num_holes, data_type=np.float64):
        self.num_particles = num_particles
        self.num_holes = num_holes
        self.spin_cases = []
        self.dimensions = []

        assert num_particles != num_holes

        order = min(num_particles, num_holes)
        if num_particles > num_holes: # EA operator
            num_add = num_particles - num_holes
            single_particle_dims = {'a' : system.nunoccupied_alpha, 'b' : system.nunoccupied_beta}
            single_particle_spins = ['a', 'b']
        elif num_particles < num_holes: # IP operator
            num_add = num_holes - num_particles
            single_particle_dims = {'a' : system.noccupied_alpha, 'b' : system.noccupied_beta}
            single_particle_spins = ['a', 'b']

        add_spin = []
        add_dims = []
        for comb in combinations_with_replacement(single_particle_spins, num_add):
            add_spin.append(''.join(list(comb)))
            add_dims.append([single_particle_dims[x] for x in comb])

        ndim = 0
        # initial iteration for the purely ionizing/attaching operators (e.g., R_1p/1h or R_2p/2h)
        for spin, dim in zip(add_spin, add_dims):
            name = spin
            dimensions = dim

            setattr(self, name, np.zeros(dimensions, dtype=data_type))
            self.spin_cases.append(name)
            self.dimensions.append(dimensions)
            ndim += np.prod(dimensions)
        # now add ionizing/attaching operators to np-nh particle-conserving operators
        for i in range(1, order + 1):
            for j in range(i + 1):
                name_base = get_operator_name(i, j)
                dimension_base = get_operator_dimension(i, j, system)

                for spin, dim in zip(add_spin, add_dims):
                    name = name_base + spin
                    dimensions = dimension_base + dim

                    setattr(self, name, np.zeros(dimensions, dtype=data_type))
                    self.spin_cases.append(name)
                    self.dimensions.append(dimensions)
                    ndim += np.prod(dimensions)

        self.ndim = ndim

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.spin_cases]
        )

    def unflatten(self, T_flat):
        prev = 0
        for dims, name in zip(self.dimensions, self.spin_cases):
            ndim = np.prod(dims)
            setattr(self, name, np.reshape(T_flat[prev : ndim + prev], dims))
            prev += ndim

def get_operator_name(i, j):
    return "a" * (i - j) + "b" * j

def get_operator_dimension(i, j, system):

    nocc_a = system.noccupied_alpha
    nocc_b = system.noccupied_beta
    nunocc_a = system.nunoccupied_alpha
    nunocc_b = system.nunoccupied_beta

    ket = [nunocc_a] * (i - j) + [nunocc_b] * j
    bra = [nocc_a] * (i - j) + [nocc_b] * j

    return ket + bra


if __name__ == "__main__":

    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvdz",
        charge=1,
        spin=1,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = load_pyscf_integrals(mf, nfrozen,
                                     num_act_holes_alpha = 2,
                                     num_act_particles_alpha = 1,
                                     num_act_holes_beta = 1,
                                     num_act_particles_beta = 2)

    print(system)

    print('Active t3 aaa')
    print('----------------')
    t3 = ActiveOperator(system, 3, 'aaa', 1)
    for slice, dim in zip(t3.slices, t3.dimensions):
        print(slice, "->", dim)
    print("Flattened dimension = ", t3.ndim)
    print(t3.flatten().shape)

    print('Active t3 aab')
    print('----------------')
    t3 = ActiveOperator(system, 3, 'aab', 1)
    for slice, dim in zip(t3.slices, t3.dimensions):
        print(slice, "->", dim)
    print("Flattened dimension = ", t3.ndim)
    print(t3.flatten().shape)

    order = 3
    T = ClusterOperator(system, order, active_orders=[3], num_active=[1])
    print("Cluster operator order", order)
    print("---------------------------")
    for spin in T.spin_cases:
        try:
            print(spin, "->", getattr(T, spin).shape)
        except:
            for slice, dim in zip(getattr(T, spin).slices, getattr(T, spin).dimensions):
                print(spin, "->", slice, "->", dim)
    print("Flattened dimension = ", T.ndim)
    print(T.flatten().shape)
    #
    # num_particles = 1
    # num_holes = 3
    # R = FockOperator(system, num_particles, num_holes)
    # print("IP operator", num_particles, 'p-', num_holes, 'h')
    # print("---------------------------")
    # for key in R.spin_cases:
    #     print(key, "->", getattr(R, key).shape)
    # print("Flattened dimension = ", R.ndim)
    # print(R.flatten().shape)
    #
    # num_particles = 4
    # num_holes = 2
    # R = FockOperator(system, num_particles, num_holes)
    # print("EA operator", num_particles, 'p-', num_holes, 'h')
    # print("---------------------------")
    # for key in R.spin_cases:
    #     print(key, "->", getattr(R, key).shape)
    # print("Flattened dimension = ", R.ndim)
    # print(R.flatten().shape)
