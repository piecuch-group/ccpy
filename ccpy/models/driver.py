from typing import Callable
from pydantic import BaseModel


class CCDriver(BaseModel):

    method_name: str
    update_function: Callable
    max_iterations: int = 60
    energy_threshold: float = 1e-6
    residuum_threshold: float = 1e-4
    energy_shift: float = 0.0
    diis_size: int = 6

    def update(self):
        return self.update_function(self)

    def is_converged(self):
        pass