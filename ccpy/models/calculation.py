from dataclasses import dataclass


# TODO: This is where I still need to think more about how we should proceed
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
