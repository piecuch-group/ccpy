from ccpy.utilities.pspace import count_excitations_in_pspace

class Calculation:

    def __init__(self, calculation_type,
                       order=0,
                       num_particles=0,
                       num_holes=0,
                       multiplicity=1,
                       maximum_iterations=80,
                       convergence_tolerance=1.0e-07,
                       energy_shift=0.0,
                       diis_size=6,
                       low_memory=False,
                       RHF_symmetry=False,
                       adaptive_percentages=None,
                       active_orders=None,
                       num_active=None,
                       p_orders=None,
                       pspace_sizes=None):

        if adaptive_percentages is None:
            adaptive_percentages = [None]
        if active_orders is None:
            active_orders = [None]
        if num_active is None:
            num_active = [None]
        if p_orders is None:
            p_orders = [None]
        if pspace_sizes is None:
            pspace_sizes = [None]

        self.calculation_type = calculation_type.lower()
        self.multiplicity = multiplicity
        self.maximum_iterations = maximum_iterations
        self.convergence_tolerance = convergence_tolerance
        self.energy_shift = energy_shift
        self.diis_size = diis_size
        self.low_memory = low_memory
        self.RHF_symmetry = RHF_symmetry

        self.active_orders = active_orders
        self.num_active = num_active
        self.adaptive_percentages = adaptive_percentages

        self.order = order
        self.num_particles = num_particles
        self.num_holes = num_holes

        if calculation_type.lower() in ["ccsd", "eomccsd", "left_ccsd", "eccc2"]:
            self.order = 2
            self.num_particles = 2
            self.num_holes = 2
        elif calculation_type.lower() in ["ccsdt", "eomccsdt", "left_ccsdt", "left_ccsdt_p", "ccsdt_p_slow", "left_ccsdt_p_slow"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 3
        elif calculation_type.lower() in ["ccsdt_p", "ccsdt_p_linear", "ccsdt_p_linear_omp", "ccsdt_p_quadratic", "adapt_ccsdt"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 3
        elif calculation_type.lower() in ["ccsdtq"]:
            self.order = 4
            self.num_particles = 4
            self.num_holes = 4
        elif calculation_type.lower() in ["ccsdt1", "eomccsdt1"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 3
            self.num_active = [1]
            self.active_orders = [3]
        elif calculation_type.lower() in ["ipeom2", "left_ipeom2"]:
            self.order = 2
            self.num_particles = 1
            self.num_holes = 2
        elif calculation_type.lower() in ["ipeom3", "left_ipeom3"]:
            self.order = 3
            self.num_particles = 2
            self.num_holes = 3
        elif calculation_type.lower() in ["eaeom2", "left_eaeom2"]:
            self.order = 2
            self.num_particles = 2
            self.num_holes = 1
        elif calculation_type.lower() in ["eaeom3", "left_eaeom3"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 2
        elif calculation_type.lower() in ["dipeom3", "left_dipeom3"]:
            self.order = 3
            self.num_particles = 1
            self.num_holes = 3
        elif calculation_type.lower() in ["dipeom4", "left_dipeom4"]:
            self.order = 4
            self.num_particles = 2
            self.num_holes = 4
        elif calculation_type.lower() in ["deaeom3", "left_deaeom3"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 1
        elif calculation_type.lower() in ["deaeom4", "left_deaeom4"]:
            self.order = 4
            self.num_particles = 4
            self.num_holes = 2

# from dataclasses import dataclass, field
#
# @dataclass
# class Calculation:
#
#     calculation_type: str
#     order: int
#     num_particles: int = 0
#     num_holes: int = 0
#     maximum_iterations: int = 60
#     convergence_tolerance: float = 1.0e-07
#     energy_shift: float = 0.0
#     diis_size: int = 6
#     low_memory: bool=False
#     RHF_symmetry: bool = False
#     multiplicity: int = 1
#
#     # default value list parameters
#     active_orders: list = field(default_factory=lambda: [None])
#     num_active : list = field(default_factory=lambda: [None])
#     adaptive_percentages : list = field(default_factory=lambda: [None])
#
#     def __post_init__(self):
#         self.num_particles = self.order if self.num_particles == 0 else self.num_particles
#         self.num_holes = self.order if self.num_holes == 0 else self.num_holes
