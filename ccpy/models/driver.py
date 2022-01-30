from typing import Callable
from pydantic import BaseModel

from ccpy.models.operators import ClusterOperator


class CCDriver(BaseModel):

    method_name: str # should correlate to order for ClusterOperator
    update_function: Callable
    max_iterations: int = 60
    energy_threshold: float = 1e-7
    residuum_threshold: float = 1e-7
    energy_shift: float = 0.0
    diis_size: int = 6

    def update(self):
        return self.update_function(self)

    def is_converged(self):
        pass

    def kernel(self, T = ClusterOperator()):
        if
        T, energy = solve_cc_jacobi(update_t, T, dT, H, calculation, diis_out_of_core=False)
