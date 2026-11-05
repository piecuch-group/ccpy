"""Analytic nuclear gradient of CC(P) with P = all singles and doubles plus a list of triples
(CCSDT(P); CCSDT when the list holds every triple), RHF or UHF reference, factorized with
density fitting or Cholesky vectors.

Theory
------
The base functional of ccpq_gradients_revised.pdf, Sec. 2, L_P = <Phi|(1 + Lambda) Hbar|Phi>.
For P = SD + triples the omitted triples cannot be raised into a retained bra (Eq. 2.9), so
the gradient multiplier is the left-CC(P) Lambda of ccpy.methods.left.LeftCCSDT_P. With the
T1-dressed H(T1) = e^{-T1} H e^{T1} the densities are those of
<Phi|(1 + Lambda) e^{-T2-T3} H(T1) e^{T2+T3}|Phi>, and the stages are those of the CCSD gradient:

  1. calc_ccsdt_p_density: rdm1, rdm2 conjugate to the stored blocks of H(T1)
     (ccpy.density.ccsdt_p_rdm) taken to the bare MO frame (ccpy.gradients.density_gradient.build_DY);
  2. ccpy.gradients.orbital_response.orbital_relaxation;
  3. ccpy.gradients.derivative_integrals.contract_derivative_integrals.

The density module must provide

    calc_ccsdt_p_rdm(T, L, t3_excitations, system) -> (rdm1, rdm2)
        Integral containers (Integral.from_empty(system, n, use_none=True)), each block the
        derivative of the functional with respect to that stored block of H(T1), reference
        energy and vvvv blocks excluded, as in ccpy.density.ccsd_rdm.calc_ccsd_rdm;
    calc_ccsdt_p_rdm2_vvvv_slice(T, L, t3_excitations, a, spin)
        for spin "a" (dF/dH.aa.vvvv[a], dF/dH.ab.vvvv[a]), for spin "b" dF/dH.bb.vvvv[a],
        as in ccpy.density.ccsd_rdm.calc_rdm2_vvvv_slice.

T and L are the CC(P) and left-CC(P) ClusterOperators, their triples aligned with the rows of
t3_excitations. tests/scf/ccsdt_p_harness.py checks both functions block by block against an
exact determinant-space reference.
"""

import time

from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.density_gradient import build_DY, get_bare_integrals
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol


def calc_ccsdt_p_gradient(T, L, t3_excitations, reference, rdm=None, vvvv_slice=None, zvector_tol=1.0e-10,
                          perturbation=None, properties=None, verbose=True):
    """Analytic CC(P) nuclear gradient from converged CC(P) (T) and left-CC(P) (L) amplitudes.

    rdm, vvvv_slice : (rdm1, rdm2) and vvvv_slice(a, spin) to use instead of
        ccpy.density.ccsdt_p_rdm (e.g. reference densities, or a partial implementation)
    perturbation : AO one-electron operator(s) (see contract_derivative_integrals); dE/dlambda is
        returned instead of the nuclear gradient
    properties : AO one-electron operator(s) V; (gradient, dE/dlambda) are returned from the same
        relaxed density
    """
    t0 = time.perf_counter()
    H = build_hbar_ccs_chol(T, reference.H, reference.system)
    D1, Y = calc_ccsdt_p_density(T, L, t3_excitations, H, reference, rdm=rdm, vvvv_slice=vvvv_slice)
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference, perturbation)
    if verbose:
        print_gradient(reference.molecule, grad, "CC(P) Analytic Gradient", time.perf_counter() - t0)
    if properties is not None:
        return grad, contract_derivative_integrals(D, Y, W, reference, properties)
    return grad


def calc_ccsdt_p_density(T, L, t3_excitations, H, reference, rdm=None, vvvv_slice=None):
    """Amplitude-relaxed (D1, Y) of the CC(P) functional (reference included), per spin, in the
    bare MO frame (H: build_hbar_ccs_chol)."""
    system = reference.system
    if rdm is None or vvvv_slice is None:
        from ccpy.density.ccsdt_p_rdm import calc_ccsdt_p_rdm, calc_ccsdt_p_rdm2_vvvv_slice

        if rdm is None:
            rdm = calc_ccsdt_p_rdm(T, L, t3_excitations, system)
        if vvvv_slice is None:
            vvvv_slice = lambda a, s: calc_ccsdt_p_rdm2_vvvv_slice(T, L, t3_excitations, a, s)  # noqa: E731
    rdm1, rdm2 = rdm
    _, B = get_bare_integrals(reference)
    return build_DY(rdm1, rdm2, vvvv_slice, H.chol, T, B, reference, ref_weight=True)
