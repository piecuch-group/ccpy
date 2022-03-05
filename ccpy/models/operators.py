import numpy as np
from itertools import combinations_with_replacement

class ClusterOperator:
    def __init__(self, system, order, data_type=np.float64):
        self.order = order
        self.spin_cases = []
        self.dimensions = []
        ndim = 0
        for i in range(1, order + 1):
            for j in range(i + 1):
                name = get_operator_name(i, j)
                dimensions = get_operator_dimension(i, j, system)
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


class FockOperator:
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
    system, H = load_pyscf_integrals(mf, nfrozen)

    print(system)

    order = 3
    T = ClusterOperator(system, order)
    print("Cluster operator order", order)
    print("---------------------------")
    for key in T.spin_cases:
        print(key, "->", getattr(T, key).shape)
    print("Flattened dimension = ", T.ndim)
    print(T.flatten().shape)

    num_particles = 2
    num_holes = 3
    R = FockOperator(system, num_particles, num_holes)
    print("IP operator", num_particles, 'p-', num_holes, 'h')
    print("---------------------------")
    for key in R.spin_cases:
        print(key, "->", getattr(R, key).shape)
    print("Flattened dimension = ", R.ndim)
    print(R.flatten().shape)

    num_particles = 4
    num_holes = 2
    R = FockOperator(system, num_particles, num_holes)
    print("EA operator", num_particles, 'p-', num_holes, 'h')
    print("---------------------------")
    for key in R.spin_cases:
        print(key, "->", getattr(R, key).shape)
    print("Flattened dimension = ", R.ndim)
    print(R.flatten().shape)
