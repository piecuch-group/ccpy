from dataclasses import dataclass, field


@dataclass
class Calculation:

    calculation_type: str
    order: int
    maximum_iterations: int = 60
    convergence_tolerance: float = 1.0e-07
    energy_shift: float = 0.0
    diis_size: int = 6
    low_memory: bool=False
    RHF_symmetry: bool = False
    multiplicity: int = 1

    # default value list parameters
    active_orders: list = field(default_factory=lambda: [None])
    num_active : list = field(default_factory=lambda: [None])