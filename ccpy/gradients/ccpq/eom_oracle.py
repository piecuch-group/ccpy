"""Independent determinant-space EOMCC(P;Q) derivative oracle.

Uses implicit-function adjoints and complex-step differentiation rather than
assuming any proposed response contractions. This is intentionally an oracle,
not a production EOMCC tensor implementation. Spin-orbital RDMs follow from
pulling the stationary Hbar coefficient through e^-T H e^T.
"""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from .amplitudes import SelectedCC, similarity_transform
from .densities import transition_densities


def csjac(fun, x, h=1.e-25):
    x = np.asarray(x, dtype=float)
    y = np.asarray(fun(x))
    out = np.empty((y.size, x.size))
    for j in range(x.size):
        z = x.astype(complex); z[j] += 1j*h
        out[:, j] = np.imag(fun(z)).reshape(-1)/h
    return out


def csgrad(fun, x):
    return csjac(lambda y: np.atleast_1d(fun(y)), x).reshape(-1)


class EOMOracle:
    """Compatibility adapter for historical 0-based *excited-state* indices.

    New code should use StateOracle(state=0 for ground, >=1 for excited).
    The published full-projected-Hbar moment definition is always used; the
    old commutator-moment model was not the standard EOMCC(P;Q) correction.
    """
    def __init__(self, space, P, Q, state=0, eom_matrix='projected'):
        if eom_matrix != 'projected':
            raise ValueError("commutator EOM moment model removed; use projected full-Hbar CC(P;Q)")
        from .state_oracle import StateOracle
        self.impl=StateOracle(space,P,Q,state=state+1)
    def solve(self,H,check_fd=False):
        res=self.impl.solve(H)
        from types import SimpleNamespace
        checks=dict(res.checks)
        checks['stationarity']=max(checks['stationarity_base'],checks['stationarity_corr'])
        checks['J_condition']=checks['jacobian_cond']
        return SimpleNamespace(E=res.E,E0=res.E_base-res.omega,omega=res.omega,
            delta=res.delta,t=res.t,r=res.r,l=res.l,z=res.arrays['adj_base'],
            adjoint=res.arrays['adj_corr'],Abar=None,gradient_H=res.gradient_H,
            gamma=res.gamma,Gamma=res.Gamma,checks=checks)
    def directional_gradient(self,res,dH):
        return np.sum(res.gradient_H*dH)
