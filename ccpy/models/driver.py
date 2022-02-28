#from typing import Callable

#from pydantic import BaseModel

from ccpy.models import System
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
from ccpy.drivers.diis import DIIS

# from functools import partial
#
# class CCDriver:
#
#     def __init__(self, order,
#                        update_function,
#                        system,
#                        hamiltonian,
#                        T_init=None,
#                        max_iterations=60,
#                        convergence=1.0e-07,
#                        energy_shift=0.0,
#                        diis_size=6,
#                        use_RHF_symmetry=False,
#                        diis_out_of_core=False):
#
#         if T_init is None:
#             T = ClusterOperator(system, order)
#         else:
#             T = T_init
#
#         dT = ClusterOperator(system, order)
#
#         self.order = order
#         self.update_function = partial(update_function,
#                                        dT=dT,
#                                        H=hamiltonian,
#                                        shift=energy_shift,
#                                        flag_RHF=use_RHF_symmetry)
#         self.diis_engine = DIIS(T, diis_size, diis_out_of_core)
#
#         self.max_iterations = max_iterations
#         self.convergence = convergence
#
#     def update(self):
#         return self.update_function(self)
#
#     def is_converged(self):
#         if
#
#     # def kernel(self, system, H, T=ClusterOperator(self.system, self.order)):
#     #     T, energy = solve_cc_jacobi(
#     #         update_t, T, dT, H, calculation, diis_out_of_core=False
#     #     )


class CCMethod(BaseModel):

    order: int
    update_function: Callable
    system: System
    hamiltonian: Integral
    max_iterations: int = 60
    energy_threshold: float = 1e-7
    residuum_threshold: float = 1e-7
    energy_shift: float = 0.0
    diis_size: int = 6
