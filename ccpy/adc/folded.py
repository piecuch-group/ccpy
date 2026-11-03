"""Energy-dependent primary-space ADC(2) using CCpy Davidson.

Only primary vectors enter the Davidson subspace. Satellite work arrays are
reconstructed from the diagonal resolvent and retained only for final states.
"""
import numpy as np

from ccpy.adc.eeadc2 import update
from ccpy.drivers.solvers import eomcc_davidson


class PrimaryVector:
    def __init__(self, size):
        self.ndim = size
        self.data = np.zeros(size)

    def flatten(self):
        return self.data.copy()

    def unflatten(self, vector):
        self.data = np.array(vector, copy=True)


class FoldedOperator:
    def __init__(self, kernel, omega, mask):
        self.kernel, self.omega = kernel, omega
        self.nprimary = kernel.space.slices[1 if getattr(kernel, "sector", "EE") != "EE" else 2].start
        self.mask = mask[:self.nprimary]
        self.diagonal = kernel.diagonal[:self.nprimary]
        self.denominator = omega - kernel.diagonal[self.nprimary:]
        if np.min(np.abs(self.denominator), initial=np.inf) < 1.e-10:
            raise ValueError("Folded ADC(2) encountered a satellite pole; use formulation='full'.")
        self.shift = 100 + 10 * np.max(np.abs(kernel.diagonal), initial=0)

    def reconstruct(self, primary):
        vector = np.zeros(self.kernel.space.ndim)
        vector[:self.nprimary] = primary * self.mask
        coupling = self.kernel.matvec(vector)[self.nprimary:]
        vector[self.nprimary:] = coupling / self.denominator
        return vector

    def matvec(self, primary):
        vector = self.reconstruct(primary)
        return (self.kernel.matvec(vector)[:self.nprimary] * self.mask
                + self.shift * primary * ~self.mask)


def solve_folded(kernel, seed, omega, mask, system, options):
    """Newton iteration of lambda(omega)-omega with overlap-followed Davidson.

    d lambda/d omega = -||R2||^2 for unit-normalized R1. No orthogonal
    projection of primary roots is used: different folded roots are orthogonal
    in their reconstructed full-space metric, not in the primary metric.
    """
    nprimary = kernel.space.slices[1 if getattr(kernel, "sector", "EE") != "EE" else 2].start
    primary = seed[:nprimary] * mask[:nprimary]
    if np.linalg.norm(primary) < 1.e-8:
        raise ValueError("Folded ADC(2) requires primary character; use formulation='full' for satellites/doubles.")
    primary /= np.linalg.norm(primary)
    for iteration in range(options["maximum_iterations"]):
        operator = FoldedOperator(kernel, omega, mask)
        R, dR = PrimaryVector(nprimary), PrimaryVector(nprimary)

        def action(dR, trial, T, H, RHF_symmetry, system):
            return H.matvec(trial.flatten())

        R, eigenvalue, ok = eomcc_davidson(action, update, primary, R, dR, np.asarray(omega),
                                          None, operator, system, options)
        if not ok:
            raise RuntimeError("Folded ADC(2) primary Davidson solve did not converge.")
        primary = R.flatten()
        primary /= np.linalg.norm(primary)
        vector = operator.reconstruct(primary)
        norm = np.linalg.norm(vector)
        residual = np.linalg.norm(kernel.matvec(vector) - omega * vector) / norm
        change = float(eigenvalue - omega)
        print(f"   Folded ADC(2) outer {iteration}: omega={omega:.12f}, residual={residual:.3e}")
        if residual < options["amp_convergence"] * 10 and abs(change) < options["energy_convergence"]:
            return vector / norm, float(omega)
        omega = float(omega + change / (norm * norm))
    raise RuntimeError("Folded ADC(2) nonlinear iteration did not converge; use formulation='full' near satellites.")
