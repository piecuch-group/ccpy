from dataclasses import dataclass, field


# TODO: This is where I still need to think more about how we should proceed
# KG: My thinking is that we create a separate Calculation object for each type of
#     calculation we might execute (CC, EOMCC, CC(P;Q), CI, etc.). Then, our paradigm
#     as far as the calculation driver is concerned is
#           system, hamiltonian = initialize()
#           calc = Calculation(method=, maxit=, tol=, shift=, ...)
#           T, energy = driver(system, hamiltonian, calc)
@dataclass
class Calculation:

    calculation_type: str
    order: int
    maximum_iterations: int = 60
    convergence_tolerance: float = 1.0e-07
    energy_shift: float = 0.0
    diis_size: int = 6
    diis_out_of_core: bool=False
    RHF_symmetry: bool = False

    # default value list parameters
    active_orders: list = field(default_factory=lambda: [None])
    num_active : list = field(default_factory=lambda: [None])