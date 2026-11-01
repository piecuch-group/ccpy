"""Analytic nuclear gradient of CCSD (RHF or UHF reference, with or without frozen core),
factorized with density fitting or Cholesky vectors: two-electron integrals and densities are
only ever handled through the three-index vectors B_x[p, q] (<pq|rs> = sum_x B_x[p,r] B_x[q,s])
and nothing with four virtual indices is stored.

Theory
------
With the CCS-transformed Hamiltonian H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol: again
one- plus two-body, with T1-transformed Cholesky vectors), the CCSD Lagrangian is that of
T1 = 0,

    L = E_ref(H(T1)) + E(T2; H(T1)) + sum_mu lambda_mu R_mu(T2; H(T1)),

so the densities are those of ccpy.density.ccsd_rdm (dL/d(blocks of H(T1))). The gradient is
assembled in three stages, each per spin:

  1. calc_ccsd_density: amplitude-relaxed D1[s] = dL/dh[s], Y[s] = dL/dB[s] in the bare MO
     frame (ccpy.gradients.mo_density.build_DY);
  2. ccpy.gradients.orbital_response.orbital_relaxation: Z-vector of the reference conditions,
     giving the relaxed D[s], Y[s] and the energy-weighted W[s];
  3. ccpy.gradients.derivative_integrals.contract_derivative_integrals.
"""

import time

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.mo_density import build_bare_hamiltonian, build_DY, get_mo_integrals
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol


def ccsd_gradient(driver, meanfield=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic CCSD nuclear gradient, shape (natoms, 3), in Eh/bohr.

    Parameters
    ----------
    driver : Driver
        Built with ``Driver.from_scf`` from a ccpy.scf RHF or UHF object (density fitting or
        Cholesky), after ``run_cc``, ``run_hbar`` and ``run_leftcc`` (ccsd or ccsd_chol variants).
    meanfield : ccpy.scf.RHF or ccpy.scf.UHF, optional
        Defaults to the SCF object stored on the driver by ``from_scf``.
    zvector_tol : float
        Convergence threshold of the Z-vector (conjugate-gradient) solver.
    """
    mf = meanfield if meanfield is not None else getattr(driver, "scf", None)
    if driver.T is None or driver.L[0] is None:
        raise RuntimeError("Run CCSD, the CCSD HBar, and left-CCSD before the gradient.")
    return ccsd_gradient_from_amplitudes(mf, driver.system, driver.T, driver.L[0], zvector_tol=zvector_tol,
                                         verbose=verbose)


def ccsd_gradient_from_amplitudes(meanfield, system, T, L, H0=None, rdm=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic CCSD nuclear gradient from converged CCSD (T) and left-CCSD (L) amplitudes.

    Parameters
    ----------
    meanfield : ccpy.scf.RHF or ccpy.scf.UHF
    system : System of the correlated calculation
    H0 : bare Cholesky Hamiltonian (with H0.chol; built from the MO integrals if not given)
    rdm : (rdm1, rdm2) of ccpy.density.ccsd_rdm.calc_ccsd_rdm(T, L, system), if available
    """
    t0 = time.perf_counter()
    if T.order != 2:
        raise NotImplementedError("Only CCSD gradients are implemented.")
    mo = get_mo_integrals(meanfield, system)
    if H0 is None or getattr(H0, "chol", None) is None:
        H0 = build_bare_hamiltonian(mo, system)
    H = build_hbar_ccs_chol(T, H0, system)
    D1, Y = calc_ccsd_density(T, L, H, mo, system, rdm=rdm)
    D, Y, W = orbital_relaxation(D1, Y, mo, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(meanfield.mol, D, Y, W, mo)
    if verbose:
        print_gradient(meanfield.mol, grad, "CCSD Analytic Gradient", time.perf_counter() - t0)
    return grad


def calc_ccsd_density(T, L, H, mo, system, rdm=None):
    """Amplitude-relaxed (D1, Y) of the CCSD Lagrangian (reference included), per spin, in the
    bare MO frame (H: build_hbar_ccs_chol)."""
    rdm1, rdm2 = rdm if rdm is not None else calc_ccsd_rdm(T, L, system)
    return build_DY(rdm1, rdm2, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s),
                    H.chol, T, mo["B"], mo, system, reference=True)
