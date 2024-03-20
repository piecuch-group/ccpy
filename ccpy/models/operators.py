import numpy as np

# [TODO]: Allow Cluster Operators to load in the values from another Cluster Operator of lower rank (or different P space)

class PspaceOperator:

    def __init__(self, n_amps, data_type=np.float64):
        if n_amps == 0:
            self.amplitudes = np.zeros(shape=(1,), dtype=data_type, order="F")
            self.excitations = np.ones((1, 6), data_type=np.int32, order="F")
        self.amplitudes = np.zeros(n_amps, dtype=data_type, order="F")
        self.excitations = np.zeros((n_amps, 6), data_type=np.int32, order="F")

    def flatten(self):
        return self.amplitudes

class ActiveOperator:

    def __init__(self, system, order, spincase, num_active, matrix=None, data_type=np.float64):
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
                            setattr(self, temp, np.zeros(dimensions, dtype=data_type, order="F"))


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
    def __init__(self, system, order, p_orders=[None], pspace_sizes=[None], active_orders=[None], num_active=[None], data_type=np.float64):
        self.order = order
        self.spin_cases = []
        self.dimensions = []

        # [TODO]: think of a nicer way to handle the active order cases
        ndim = 0
        act_cnt = 0
        p_cnt = 0
        for i in range(1, order + 1):

            if i in active_orders:
                nact = num_active[act_cnt]
                act_cnt += 1

            if i in p_orders:
                excitation_count = pspace_sizes[p_cnt]
                p_cnt += 1

            for j in range(i + 1):
                name = get_operator_name(i, j)
                dimensions = get_operator_dimension(i, j, system)

                if i in active_orders:
                    active_t = ActiveOperator(system, i, name, nact, data_type=data_type)
                    setattr(self, name, active_t)
                    for dim in active_t.dimensions:
                        self.dimensions.append(dim)
                    ndim += active_t.ndim

                # This is trying to set a zero 1D vector for the P space components
                elif i in p_orders:
                    setattr(self, name, np.zeros(excitation_count[j], dtype=data_type, order="F"))
                    if excitation_count[j] == 0:
                         setattr(self, name, np.zeros(shape=(1,), dtype=data_type, order="F"))
                    self.dimensions.append((excitation_count[j],))
                    ndim += excitation_count[j]
                   # developmental, for the PspaceOperator
                   #setattr(self, name, PspaceOperator(excitation_count[j], data_type=data_type))
                   #self.dimensions.append((excitation_count[j],))
                   #ndim += excitation_count[j]
                else:
                    setattr(self, name, np.zeros(dimensions, dtype=data_type, order="F"))
                    self.dimensions.append(dimensions)
                    ndim += np.prod(dimensions)

                self.spin_cases.append(name)

        self.ndim = ndim

    def extend_pspace_t3_operator(self, excitation_count_spincase):
        assert len(excitation_count_spincase) == 4
        for i, spincase in enumerate(["aaa", "aab", "abb", "bbb"]):
            num_old = len(getattr(self, spincase))
            num_extend = excitation_count_spincase[i] - num_old
            if num_extend > 0:
                setattr(self, spincase, np.hstack((getattr(self, spincase), np.zeros(num_extend, dtype=np.float64, order="F"))))
                self.dimensions[i + 5] = (num_old + num_extend,)
                self.ndim += num_extend

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.spin_cases]
        )

    def unflatten(self, T_flat, order=0):
        prev = 0

        # allows unflattening of up to a specified order which may be less than
        # the order of the cluster operator.
        if order == 0: order = self.order

        for dims, name in zip(self.dimensions, self.spin_cases):

            if len(name) > order: continue

            if isinstance(getattr(self, name), ActiveOperator):
                getattr(self, name).unflatten(T_flat[prev : prev + getattr(self, name).ndim])
                prev += getattr(self, name).ndim
            # Developing for the PspaceOperator
            #elif isinstance(getattr(self, name), ActiveOperator):
            #    ndim = np.prod(dims)
            #    setattr(getattr(self, name), "amplitudes", np.reshape(T_flat[prev: ndim + prev], dims))
            #    prev += ndim
            else:
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
    def __init__(self, system, num_particles, num_holes, p_orders=[None], pspace_sizes=[None], data_type=np.float64):
        self.num_particles = num_particles
        self.num_holes = num_holes
        self.spin_cases = []
        self.dimensions = []
        self.num_diff = abs(num_particles - num_holes)
        self.num_excit = min(num_particles, num_holes)
        self.order = self.num_diff + self.num_excit

        assert num_particles != num_holes

        #order = min(num_particles, num_holes)
        if num_particles > num_holes: # EA operator
            num_add = num_particles - num_holes
            single_particle_dims = {'a' : system.nunoccupied_alpha, 'b' : system.nunoccupied_beta}
            single_particle_spins = ['a', 'b']
        elif num_particles < num_holes: # IP operator
            num_add = num_holes - num_particles
            single_particle_dims = {'a' : system.noccupied_alpha, 'b' : system.noccupied_beta}
            single_particle_spins = ['a', 'b']

        # Here, we can cut down the number of cases using RHF symmetry, e.g., for
        # EA/IP -> only add/remove a
        # DEA/DIP -> only add/remove ab
        # TEA/TIP -> only add/remove aaa for quartet, aab for doublet/triplet (let's not do this case)
        add_spin = []
        add_dims = []
        if num_add == 1: # EA/IP
            add_spin.append("a")
            add_dims.append([single_particle_dims[x] for x in ["a"]])
        elif num_add == 2: # DEA/DIP
            add_spin.append("ab")
            add_dims.append([single_particle_dims[x] for x in ["a", "b"]])

        # add_spin = []
        # add_dims = []
        # for comb in combinations_with_replacement(single_particle_spins, num_add):
        #     add_spin.append(''.join(list(comb)))
        #     add_dims.append([single_particle_dims[x] for x in comb])

        ndim = 0
        p_cnt = 0
        # initial iteration for the purely ionizing/attaching operators (e.g., R_1p/1h or R_2p/2h)
        for spin, dim in zip(add_spin, add_dims):
            name = spin
            dimensions = dim

            setattr(self, name, np.zeros(dimensions, dtype=data_type, order="F"))
            self.spin_cases.append(name)
            self.dimensions.append(dimensions)
            ndim += np.prod(dimensions)

        # now add ionizing/attaching operators to np-nh particle-conserving operators
        for i in range(1, self.num_excit + 1):

            if i in p_orders:
                excitation_count = pspace_sizes[p_cnt]
                p_cnt += 1

            for j in range(i + 1):
                name_base = get_operator_name(i, j)
                dimension_base = get_operator_dimension(i, j, system)

                for spin, dim in zip(add_spin, add_dims):
                    name = spin + name_base
                    dimensions = dim + dimension_base

                    if i in p_orders:
                        # add a P space vector for this spin case
                        setattr(self, name, np.zeros(excitation_count[j], dtype=data_type, order="F"))
                        if excitation_count[j] == 0:
                            setattr(self, name, np.zeros(shape=(1,), dtype=data_type, order="F"))
                        self.spin_cases.append(name)
                        self.dimensions.append((excitation_count[j],))
                        ndim += excitation_count[j]
                    else:
                        # use the normal arrays
                        setattr(self, name, np.zeros(dimensions, dtype=data_type, order="F"))
                        self.spin_cases.append(name)
                        self.dimensions.append(dimensions)
                        ndim += np.prod(dimensions)
        self.ndim = ndim

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.spin_cases]
        )

    def unflatten(self, T_flat, order=0):
        prev = 0

        # allows unflattening of up to a specified order which may be less than
        # the order of the cluster operator.
        if order == 0: order = self.order

        for dims, name in zip(self.dimensions, self.spin_cases):

            if len(name) > order: continue

            ndim = np.prod(dims)
            setattr(self, name, np.reshape(T_flat[prev: ndim + prev], dims))
            prev += ndim

class SpinFlipOperator:
    """Builds generalized alpha-to-beta spin-flipping operators"""
    def __init__(self, system, order, Ms, data_type=np.float64):
        self.Ms = Ms
        self.spin_cases = []
        self.dimensions = []
        self.num_flip = abs(self.Ms)
        self.order = self.num_flip + order

        # Assert that we follow convention of a->b spin flips, not b->a
        assert self.Ms < 0

        # For the time being, hard-code this to support single, double, and triple spin-flip operators for Ms = -1
        noa = system.noccupied_alpha
        nob = system.noccupied_beta
        nua = system.nunoccupied_alpha
        nub = system.nunoccupied_beta
        if self.Ms == -1:
            if self.order == 1: # single spin-flip
                setattr(self, "b", np.zeros((nub, noa), order="F")) # r1(a~|i)
                setattr(self, "spin_cases", ["b"])
                setattr(self, "dimensions", [(nub, noa)])
            elif self.order == 2: # singles and doubles spin-flip
                setattr(self, "b", np.zeros((nub, noa), order="F")) # r1(a~|i)
                setattr(self, "ab", np.zeros((nua, nub, noa, noa), order="F")) # r2(ab~|ij)
                setattr(self, "bb", np.zeros((nub, nub, nob, noa), order="F")) # r2(a~b~|i~j)
                setattr(self, "spin_cases", ["b", "ab", "bb"])
                setattr(self, "dimensions", [(nub, noa), (nua, nub, noa, noa), (nub, nub, nob, noa)])
            elif self.order == 3: # singles, doubles, and triples spin-flip
                setattr(self, "b", np.zeros((nub, noa), order="F")) # r1(a~|i)
                setattr(self, "ab", np.zeros((nua, nub, noa, noa), order="F")) # r2(ab~|ij)
                setattr(self, "bb", np.zeros((nub, nub, nob, noa), order="F")) # r2(a~b~|i~j)
                setattr(self, "aab", np.zeros((nua, nua, nub, noa, noa, noa), order="F")) # r3(abc~|ijk)
                setattr(self, "abb", np.zeros((nua, nub, nub, noa, nob, noa), order="F")) # r3(ab~c~|ij~k)
                setattr(self, "bbb", np.zeros((nub, nub, nub, nob, nob, noa), order="F")) # r3(a~b~c~|i~j~k)
                setattr(self, "spin_cases", ["b", "ab", "bb", "aab", "abb", "bbb"])
                setattr(self, "dimensions", [(nub, noa), (nua, nub, noa, noa), (nub, nub, nob, noa),
                                             (nua, nua, nub, noa, noa, noa),
                                             (nua, nub, nub, noa, nob, noa),
                                             (nub, nub, nub, nob, nob, noa)])
            self.ndim = 0
            for dim in self.dimensions:
                self.ndim += np.prod(dim)

    def flatten(self):
        return np.hstack(
            [getattr(self, key).flatten() for key in self.spin_cases]
        )

    def unflatten(self, T_flat, order=0):
        prev = 0

        # allows unflattening of up to a specified order which may be less than
        # the order of the cluster operator.
        if order == 0: order = self.order

        for dims, name in zip(self.dimensions, self.spin_cases):

            if len(name) > order: continue

            ndim = np.prod(dims)
            setattr(self, name, np.reshape(T_flat[prev: ndim + prev], dims))
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
