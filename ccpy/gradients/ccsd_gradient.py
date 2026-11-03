"""Analytic nuclear gradient of CCSD (RHF or UHF reference, with or without frozen core),
factorized with density fitting or Cholesky vectors: two-electron integrals and densities are
only ever handled through the three-index vectors B_x[p, q] (<pq|rs> = sum_x B_x[p,r] B_x[q,s])
and nothing with four virtual indices is stored.

Theory
------
The CCSD energy is stationary in the cluster amplitudes once the CCSD equations are added with
the left-CCSD multipliers, so the gradient is that of the Lagrangian

    E = <0|(1 + Lambda) e^{-T} H e^{T}|0> = <0|(1 + Lambda) (H(T1) e^{T2})_C|0>,

with H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol), a one- plus two-body operator with
T1-transformed Cholesky vectors. Its densities with respect to the blocks of H(T1) are those of
ccpy.density.ccsd_rdm; the gradient takes them

  1. to the bare frame, D1 = dE/dh and Y_x = dE/dB_x (ccpy.gradients.density_gradient.build_DY),
  2. through the orbital response of the reference (ccpy.gradients.orbital_response),
  3. to the derivative integrals (ccpy.gradients.derivative_integrals).
"""

import time

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.gradients.density_gradient import build_DY, get_bare_integrals
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol


def calc_ccsd_gradient(T, L, reference, rdm=None, zvector_tol=1.0e-10, perturbation=None, verbose=True):
    """Analytic CCSD nuclear gradient, (natoms, 3) in Eh/bohr, from converged CCSD (T) and
    left-CCSD (L) amplitudes on the SCF reference (ccpy.methods.Reference).

    rdm : (rdm1, rdm2) of ccpy.density.ccsd_rdm.calc_ccsd_rdm(T, L, system), if available
    perturbation : AO one-electron operator(s) (see contract_derivative_integrals); dE/dlambda is
        returned instead of the nuclear gradient
    """
    t0 = time.perf_counter()
    if T.order != 2:
        raise NotImplementedError("Only CCSD gradients are implemented.")
    system = reference.system
    H = build_hbar_ccs_chol(T, reference.H, system)
    _, B = get_bare_integrals(reference)

    # densities of <0|(1 + Lambda) (H(T1) e^{T2})_C|0> with respect to the blocks of H(T1),
    # reference determinant included, in the bare frame
    rdm1, rdm2 = rdm if rdm is not None else calc_ccsd_rdm(T, L, system)
    D1, Y = build_DY(rdm1, rdm2, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B, reference,
                     ref_weight=True)

    # orbital relaxation and derivative integrals
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference, perturbation)
    if verbose:
        print_gradient(reference.molecule, grad, "CCSD Analytic Gradient", time.perf_counter() - t0)
    return grad
