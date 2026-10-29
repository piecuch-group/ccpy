import itertools
import math

import numpy as np

# Table 1 of Stoychev et al. The published table omits f for l = 1; the value 7.0
# follows Basis Set Exchange and PySCF.
_F_LAUX = [20.0, 7.0, 4.0, 4.0, 3.5, 2.5, 2.0, 2.0]
_BETA_BIG = [1.8, 2.0, 2.2, 2.2, 2.2, 2.3, 3.0, 3.0]
_BETA_SMALL = 1.8
_TIE_RTOL = 1e-10


def autoaux_electron_shells(Z, electron_shells):
    """
    Generates the AutoAux auxiliary shells of one element from its orbital shells.

    AutoAux spans the one-center product space of the orbital basis with
    even-tempered, uncontracted shells. The implementation follows the Basis Set
    Exchange port of the algorithm and reproduces its output.

    Parameters
    ----------
    Z : int
        The atomic number. Ghost atoms (Z = 0) are treated like hydrogen.
    electron_shells : list[dict]
        The orbital shells in Basis Set Exchange format, with coefficients that
        multiply normalized primitives.

    Returns
    -------
    list[dict]
        The auxiliary shells in Basis Set Exchange format.

    Notes
    -----
    G. L. Stoychev, A. A. Auer, and F. Neese, J. Chem. Theory Comput. 13, 554 (2017).
    """
    amin, amax_prim, amax_eff = {}, {}, {}
    for shell in electron_shells:
        angular_momentum = list(map(int, shell["angular_momentum"]))
        exponents = np.array(list(map(float, shell["exponents"])))
        # same expansion of sp and general contractions as the orbital basis build
        for l, coefficients in itertools.zip_longest(
            angular_momentum, shell["coefficients"], fillvalue=angular_momentum[-1]
        ):
            coefficients = np.array(list(map(float, coefficients)))
            amin[l] = min(amin.get(l, np.inf), exponents.min())
            amax_prim[l] = max(amax_prim.get(l, 0.0), exponents.max())
            amax_eff[l] = max(
                amax_eff.get(l, 0.0),
                _effective_exponent(l, exponents, coefficients),
            )

    # exponent bounds of the product space, for each auxiliary angular momentum
    minaux, maxaux_prim, maxaux_eff = {}, {}, {}
    ls = sorted(amin)
    for i, l1 in enumerate(ls):
        for l2 in ls[i:]:
            for laux in range(l2 - l1, l1 + l2 + 1):
                minaux[laux] = min(minaux.get(laux, np.inf), amin[l1] + amin[l2])
                maxaux_prim[laux] = max(
                    maxaux_prim.get(laux, 0.0), amax_prim[l1] + amax_prim[l2]
                )
                maxaux_eff[laux] = max(
                    maxaux_eff.get(laux, 0.0), amax_eff[l1] + amax_eff[l2]
                )

    # highest occupied angular momentum
    lval = 0 if Z <= 2 else 1 if Z <= 20 else 2 if Z <= 56 else 3
    linc = 1 if Z <= 18 else 2
    lmax = max(ls)
    lmax_aux = min(max(2 * lval, lmax + linc), 2 * lmax)

    aux = []
    for laux in range(lmax_aux + 1):
        if laux not in minaux:
            continue
        if laux <= 2 * lval:
            # the published formula has max where min is meant
            alpha_max = min(_F_LAUX[laux] * maxaux_eff[laux], maxaux_prim[laux])
            beta = _BETA_SMALL
        else:
            alpha_max = maxaux_eff[laux]
            beta = _BETA_BIG[min(laux, len(_BETA_BIG) - 1)]
        alpha = minaux[laux]
        while True:
            # seven significant figures, as Basis Set Exchange stores them
            aux.append(
                {
                    "angular_momentum": [laux],
                    "exponents": [f"{alpha:.6e}"],
                    "coefficients": [["1.0"]],
                }
            )
            # tolerate rounding error, as bounds from the same primitives tie exactly
            if alpha >= alpha_max * (1 - _TIE_RTOL):
                break
            alpha *= beta
    return aux


def _effective_exponent(l, exponents, coefficients):
    """
    A contracted Gaussian is replaced with a single Gaussian with an
    effective exponent with the same radial expectation.
    See eqs 8-10 in JCTC 13, 554 (2017). Eq. 9 should have r^2_exp instead of r_exp
    """
    # the exponent-dependent part of N_i
    c = coefficients * (2 * exponents) ** (l / 2 + 0.75)
    asum = exponents[:, None] + exponents[None, :]
    cc = np.outer(c, c)
    # <r> = \int chi^2(r) r dr
    r_exp = np.sum(cc * math.gamma(l + 2) / asum ** (l + 2)) / np.sum(
        cc * math.gamma(l + 1.5) / asum ** (l + 1.5)
    )
    k = 2 ** (2 * l + 1) * math.factorial(l + 1) ** 2 / math.factorial(2 * l + 2)
    return 2 * k**2 / (np.pi * r_exp**2)
