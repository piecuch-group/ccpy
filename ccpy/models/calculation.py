from dataclasses import dataclass

@dataclass
class Calculation:

    calculation_type: str
    maximum_iterations: int = 100
    convergence_tolerance: float = 1.0e-07
    level_shift: float = 0.0
    diis_size: int = 6
    RHF_symmetry: bool = False
