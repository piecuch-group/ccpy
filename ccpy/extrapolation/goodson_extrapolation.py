import math

def goodson_extrapolation(e_ref, e_ccsd, e_ccsdt, approximant):
    delta1 = e_ref
    delta2 = e_ccsd - e_ref
    delta3 = e_ccsdt - e_ccsd

    if approximant == "ccq":
        return quadratic_pade_approximant(delta1, delta2, delta3)
    elif approximant == "ccr":
        return rational_pade_approximant(delta1, delta2, delta3)
    elif approximant == "cccf":
        return continued_fraction_approximant(delta1, delta2, delta3)
    else:
        return 0.0

def quadratic_pade_approximant(delta1, delta2, delta3):

    extrap_ccq = (
        delta1
        + 0.5 * delta2**2/delta3 * (1.0 - math.sqrt(1.0 - 4 * delta3/delta2))
    )
    return extrap_ccq

def rational_pade_approximant(delta1, delta2, delta3):

    extrap_ccr = (
        delta1 * (1.0 + delta2/delta1 - delta3/delta2)/(1 - delta3/delta2)
    )
    return extrap_ccr

def continued_fraction_approximant(delta1, delta2, delta3):

    extrap_cccf = (
        delta1/(1.0 - delta2/delta1/(1.0 - delta3/delta2))
    )
    return extrap_cccf
