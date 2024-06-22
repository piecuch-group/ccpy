from itertools import combinations

import numpy as np

from ccpy.models.operators import get_operator_name

# Example: Alternative constructor that avoids using parent __init__
#
# class MyClass(set):
#
#     def __init__(self, filename):
#         self._value = load_from_file(filename)
#
#     @classmethod
#     def from_somewhere(cls, somename):
#         obj = cls.__new__(cls)  # Does not call __init__
#         super(MyClass, obj).__init__()  # Don't forget to call any polymorphic base class initializers
#         obj._value = load_from_somewhere(somename)
#         return obj
#
# class UnsortedIntegralArray(np.ndarray):
#
#     def __new__(cls, input_array):
#         # Input array is an already formed ndarray instance
#         # We first cast to be our class type
#         obj = np.asarray(input_array).view(cls)
#         # Finally, we must return the newly created object:
#         return obj
#
#     def __getitem__(self, item):
#         #if isinstance(item, np.ndarray):
#         #print(np.ix_(*item))
#         # print(item)
#         # print(*item)
#         #print(item)
#         return super().__getitem__(self, np.ix_(item))
#         #return super().__getitem__(np.ix_(*item))
#         #return super().__getitem__(item)
#
#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None: return
#         self.info = getattr(obj, 'info', None)


class SortedIntegral:
    def __init__(self, system, name, matrix, use_none=False):

        order = len(name)
        double_spin_string = list(name) * 2
        slice_table = {
            "a": {
                "o": slice(0, system.noccupied_alpha),
                "v": slice(system.noccupied_alpha, system.norbitals),
            },
            "b": {
                "o": slice(0, system.noccupied_beta),
                "v": slice(system.noccupied_beta, system.norbitals),
            },
        }

        self.slices = []
        for i in range(2 * order + 1):
            for combs in combinations(range(2 * order), i):
                attr = ["o"] * (2 * order)
                for k in combs:
                    attr[k] = "v"
                slicearr = [slice(None)] * (2 * order)
                for k in range(2 * order):
                    slicearr[k] = slice_table[double_spin_string[k]][attr[k]]
                if use_none:
                    self.__dict__["".join(attr)] = None
                else:
                    # make array F_CONTIGUOUS
                    self.__dict__["".join(attr)] = np.asfortranarray(matrix[tuple(slicearr)])
                self.slices.append(''.join(attr))

class Integral:
    def __init__(self, system, order, matrices, sorted=True, use_none=False):
        self.order = order
        for i in range(1, order + 1):  # Loop over many-body ranks
            for j in range(i + 1):  # Loop over distinct spin cases per rank
                name = get_operator_name(i, j)
                if sorted:
                    sorted_integral = SortedIntegral(system, name, matrices[name], use_none)
                    self.__dict__[name] = sorted_integral
                else:
                    self.__dict__[name] = matrices[name]

    @classmethod
    def from_empty(cls, system, order, data_type=np.float64, use_none=False):
        matrices = {}
        for i in range(1, order + 1):  # Loop over many-body ranks
            for j in range(i + 1):  # Loop over distinct spin cases per rank
                name = get_operator_name(i, j)
                dimension = [system.norbitals] * (2 * order)
                if use_none:
                    matrices[name] = None
                else:
                    matrices[name] = np.zeros(dimension, dtype=data_type, order="F")
        return cls(system, order, matrices, sorted=True, use_none=use_none)

    # @classmethod
    # def from_none(cls, system, order):
    #     matrices = {}
    #     for i in range(1, order + 1):  # Loop over many-body ranks
    #         for j in range(i + 1):  # Loop over distinct spin cases per rank
    #             name = get_operator_name(i, j)
    #             dimension = [system.norbitals] * (2 * order)
    #             matrices[name] = None
    #     return cls(system, order, matrices, use_none=True)

    @classmethod
    def from_none(cls, system, order):
        return cls(system, order, matrix=None, sorted=True, use_none=True)


def getHamiltonian(e1int, e2int, system, normal_ordered, sorted=True):

    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)

    twobody = build_v(e2int)
    if normal_ordered:
        onebody = build_f(e1int, twobody, system)
    else:
        onebody = {"a": e1int, "b": e1int}
    # Keep only correlated spatial orbitals in the one- and two-body matrices
    onebody["a"] = onebody["a"][corr_slice, corr_slice]
    onebody["b"] = onebody["b"][corr_slice, corr_slice]
    twobody["aa"] = twobody["aa"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["ab"] = twobody["ab"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["bb"] = twobody["bb"][corr_slice, corr_slice, corr_slice, corr_slice]

    return Integral(system, 2, {**onebody, **twobody}, sorted=sorted)

def getCholeskyHamiltonian(e1int, R_chol, system, normal_ordered, sorted=True):

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    oa = slice(system.nfrozen, system.nfrozen + noa)
    ob = slice(system.nfrozen, system.nfrozen + nob)
    va = slice(system.nfrozen + noa, system.nfrozen + system.norbitals)
    vb = slice(system.nfrozen + nob, system.nfrozen + system.norbitals)

    H = Integral.from_none(system, 2)
    #H.a.oo =

    # corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)
    #
    # twobody = build_v(e2int)
    # if normal_ordered:
    #     onebody = build_f(e1int, twobody, system)
    # else:
    #     onebody = {"a": e1int, "b": e1int}
    # # Keep only correlated spatial orbitals in the one- and two-body matrices
    # onebody["a"] = onebody["a"][corr_slice, corr_slice]
    # onebody["b"] = onebody["b"][corr_slice, corr_slice]
    # twobody["aa"] = twobody["aa"][corr_slice, corr_slice, corr_slice, corr_slice]
    # twobody["ab"] = twobody["ab"][corr_slice, corr_slice, corr_slice, corr_slice]
    # twobody["bb"] = twobody["bb"][corr_slice, corr_slice, corr_slice, corr_slice]
    #
    # return Integral(system, 2, {**onebody, **twobody}, sorted=sorted)

def build_v(e2int):
    """Generate the antisymmetrized version of the twobody matrix.

    Parameters
    ----------
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody MO integral array

    Returns
    -------
    v : dict
        Dictionary with v['A'], v['B'], and v['C'] containing the
        antisymmetrized twobody MO integrals.
    """
    v = {
        "aa": e2int - np.einsum("pqrs->pqsr", e2int),
        "ab": e2int,
        "bb": e2int - np.einsum("pqrs->pqsr", e2int),
    }
    return v


def build_f(e1int, v, system):
    """This function generates the Fock matrix using the formula
    F = Z + G where G is \sum_{i} <pi|v|qi>_A split for different
    spin cases.

    Parameters
    ----------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody MO integrals
    v : dict
        Twobody integral dictionary
    sys : dict
        System information dictionary

    Returns
    -------
    f : dict
        Dictionary containing the Fock matrices for the aa and bb cases
    """
    Nocc_a = system.noccupied_alpha + system.nfrozen
    Nocc_b = system.noccupied_beta + system.nfrozen

    # <p|f|q> = <p|z|q> + <pi|v|qi> + <pi~|v|qi~>
    f_a = (
        e1int
        + np.einsum("piqi->pq", v["aa"][:, :Nocc_a, :, :Nocc_a])
        + np.einsum("piqi->pq", v["ab"][:, :Nocc_b, :, :Nocc_b])
    )

    # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    f_b = (
        e1int
        + np.einsum("piqi->pq", v["bb"][:, :Nocc_b, :, :Nocc_b])
        + np.einsum("ipiq->pq", v["ab"][:Nocc_a, :, :Nocc_a, :])
    )

    f = {"a": f_a, "b": f_b}

    return f

def build_f_chol(e1int, R_chol, system):
    """This function generates the Fock matrix using the formula
    F = Z + G where G is \sum_{i} <pi|v|qi>_A split for different
    spin cases.

    Parameters
    ----------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody MO integrals
    v : dict
        Twobody integral dictionary
    sys : dict
        System information dictionary

    Returns
    -------
    f : dict
        Dictionary containing the Fock matrices for the aa and bb cases
    """
    Nocc_a = system.noccupied_alpha + system.nfrozen
    Nocc_b = system.noccupied_beta + system.nfrozen

    # <p|f|q> = <p|z|q> + <pi|v|qi> + <pi~|v|qi~>
    # f_a = (
    #     e1int
    #     + np.einsum("piqi->pq", v["aa"][:, :Nocc_a, :, :Nocc_a])
    #     + np.einsum("piqi->pq", v["ab"][:, :Nocc_b, :, :Nocc_b])
    # )
    #
    # # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    # f_b = (
    #     e1int
    #     + np.einsum("piqi->pq", v["bb"][:, :Nocc_b, :, :Nocc_b])
    #     + np.einsum("ipiq->pq", v["ab"][:Nocc_a, :, :Nocc_a, :])
    # )

    f = {"a": f_a, "b": f_b}

    return f
