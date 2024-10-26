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
    def __init__(self, system, order, matrices, sorted=True, use_none=False, chol=None):
        self.order = order
        self.chol = chol
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

    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)
    oa = slice(noa)
    ob = slice(nob)
    va = slice(noa, noa + nua)
    vb = slice(nob, nob + nub)

    if normal_ordered:
        onebody = build_f_chol(e1int, R_chol, system)
    else:
        onebody = {"a": e1int, "b": e1int}
    # Keep only correlated spatial orbitals in the one- and two-body matrices
    onebody["a"] = onebody["a"][corr_slice, corr_slice]
    onebody["b"] = onebody["b"][corr_slice, corr_slice]
    R_chol = R_chol[:, corr_slice, corr_slice]

    # Initialize Hamiltonian Integral object
    H = Integral.from_empty(system, 2, data_type=np.float64, use_none=True)
    # Allocate empty 1-body operator for the Cholesky vectors
    H.chol = Integral.from_empty(system, 1, data_type=np.float64, use_none=True)
    # Populate the sliced Cholesky components
    H.chol.a.oo = R_chol[:, oa, oa]
    H.chol.a.vv = R_chol[:, va, va]
    H.chol.a.ov = R_chol[:, oa, va]
    H.chol.a.vo = R_chol[:, va, oa]
    #
    H.chol.b.oo = R_chol[:, ob, ob]
    H.chol.b.vv = R_chol[:, vb, vb]
    H.chol.b.ov = R_chol[:, ob, vb]
    H.chol.b.vo = R_chol[:, vb, ob]
    # a
    H.a.oo = onebody["a"][oa, oa]
    H.a.vv = onebody["a"][va, va]
    H.a.ov = onebody["a"][oa, va]
    H.a.vo = onebody["a"][va, oa]
    # b
    H.b.oo = onebody["b"][ob, ob]
    H.b.vv = onebody["b"][vb, vb]
    H.b.ov = onebody["b"][ob, vb]
    H.b.vo = onebody["b"][vb, ob]
    # ---
    # aa
    # ---
    H.aa.oooo = (
                    np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
                    - np.einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
    ) # h(mnij)
    H.aa.ooov = (
                    np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov, optimize=True)
                    - np.einsum("xni,xme->mnie", H.chol.a.oo, H.chol.a.ov, optimize=True)
    ) # h(mnie)
    H.aa.vooo = (
                    np.einsum("xai,xmj->amij", H.chol.a.vo, H.chol.a.oo, optimize=True)
                    - np.einsum("xaj,xmi->amij", H.chol.a.vo, H.chol.a.oo, optimize=True)
    ) # h(amij)
    H.aa.voov = (
                    np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov, optimize=True)
                    - np.einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo, optimize=True)
    ) # h(amie)
    H.aa.oovv = (
                    np.einsum("xme,xnf->mnef", H.chol.a.ov, H.chol.a.ov, optimize=True)
                    - np.einsum("xmf,xne->mnef", H.chol.a.ov, H.chol.a.ov, optimize=True)
    ) # h(mnef)
    H.aa.vvoo = (
                    np.einsum("xai,xbj->abij", H.chol.a.vo, H.chol.a.vo, optimize=True)
                    - np.einsum("xaj,xbi->abij", H.chol.a.vo, H.chol.a.vo, optimize=True)
    ) # h(abij)
    # H.aa.vovv = (
    #                 np.einsum("xae,xmf->amef", H.chol.a.vv, H.chol.a.ov, optimize=True)
    #                 - np.einsum("xaf,xme->amef", H.chol.a.vv, H.chol.a.ov, optimize=True)
    # ) # h(amef)
    # H.aa.vvov = (
    #                 np.einsum("xai,xbe->abie", H.chol.a.vo, H.chol.a.vv, optimize=True)
    #                 - np.einsum("xbi,xae->abie", H.chol.a.vo, H.chol.a.vv, optimize=True)
    # ) # h(abie)
    # H.aa.vvvv = (
    #                 np.einsum("xae,xbf->abef", H.chol.a.vv, H.chol.a.vv, optimize=True)
    #                 - np.einsum("xaf,xbe->abef", H.chol.a.vv, H.chol.a.vv, optimize=True)
    # ) # h(abef)
    # ---
    # bb
    # ---
    H.bb.oooo = (
                    np.einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
                    - np.einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
    ) # h(mnij)
    H.bb.ooov = (
                    np.einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov, optimize=True)
                    - np.einsum("xni,xme->mnie", H.chol.b.oo, H.chol.b.ov, optimize=True)
    ) # h(mnie)
    H.bb.vooo = (
                    np.einsum("xai,xmj->amij", H.chol.b.vo, H.chol.b.oo, optimize=True)
                    - np.einsum("xaj,xmi->amij", H.chol.b.vo, H.chol.b.oo, optimize=True)
    ) # h(amij)
    H.bb.voov = (
                    np.einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov, optimize=True)
                    - np.einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo, optimize=True)
    ) # h(amie)
    H.bb.oovv = (
                    np.einsum("xme,xnf->mnef", H.chol.b.ov, H.chol.b.ov, optimize=True)
                    - np.einsum("xmf,xne->mnef", H.chol.b.ov, H.chol.b.ov, optimize=True)
    ) # h(mnef)
    H.bb.vvoo = (
                    np.einsum("xai,xbj->abij", H.chol.b.vo, H.chol.b.vo, optimize=True)
                    - np.einsum("xaj,xbi->abij", H.chol.b.vo, H.chol.b.vo, optimize=True)
    ) # h(abij)
    # H.bb.vovv = (
    #                 np.einsum("xae,xmf->amef", H.chol.b.vv, H.chol.b.ov, optimize=True)
    #                 - np.einsum("xaf,xme->amef", H.chol.b.vv, H.chol.b.ov, optimize=True)
    # ) # h(amef)
    # H.bb.vvov = (
    #                 np.einsum("xai,xbe->abie", H.chol.b.vo, H.chol.b.vv, optimize=True)
    #                 - np.einsum("xbi,xae->abie", H.chol.b.vo, H.chol.b.vv, optimize=True)
    # ) # h(abie)
    # H.bb.vvvv = (
    #                 np.einsum("xae,xbf->abef", H.chol.b.vv, H.chol.b.vv, optimize=True)
    #                 - np.einsum("xaf,xbe->abef", H.chol.b.vv, H.chol.b.vv, optimize=True)
    # ) # h(abef)
    # ---
    # ab
    # ---
    H.ab.oooo = (
                    np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo, optimize=True)
    ) # h(mnij)
    H.ab.ooov = (
                    np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov, optimize=True)
    ) # h(mnie)
    H.ab.oovo = (
                    np.einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo, optimize=True)
    ) # h(mnei)
    H.ab.vooo = (
                    np.einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo, optimize=True)
    ) # h(amij)
    H.ab.ovoo = (
                    np.einsum("xmj,xai->maji", H.chol.a.oo, H.chol.b.vo, optimize=True)
    ) # h(maji)
    H.ab.voov = (
                    np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov, optimize=True)
    ) # h(amie)
    H.ab.ovvo = (
                    np.einsum("xme,xai->maei", H.chol.a.ov, H.chol.b.vo, optimize=True)
    ) # h(maei)
    H.ab.vovo = (
                    np.einsum("xae,xmi->amei", H.chol.a.vv, H.chol.b.oo, optimize=True)
    ) # h(amei)
    H.ab.ovov = (
                    np.einsum("xmj,xbe->mbje", H.chol.a.oo, H.chol.b.vv, optimize=True)
    ) # h(mbje)
    H.ab.oovv = (
                    np.einsum("xme,xnf->mnef", H.chol.a.ov, H.chol.b.ov, optimize=True)
    ) # h(mnef)
    H.ab.vvoo = (
                    np.einsum("xai,xbj->abij", H.chol.a.vo, H.chol.b.vo, optimize=True)
    ) # h(abij)
    # H.ab.vovv = (
    #                 np.einsum("xae,xmf->amef", H.chol.a.vv, H.chol.b.ov, optimize=True)
    # ) # h(amef)
    # H.ab.ovvv = (
    #                 np.einsum("xmf,xae->mafe", H.chol.a.ov, H.chol.b.vv, optimize=True)
    # ) # h(mafe)
    # H.ab.vvov = (
    #                 np.einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv, optimize=True)
    # ) # h(abie)
    # H.ab.vvvo = (
    #                 np.einsum("xae,xbi->abei", H.chol.a.vv, H.chol.b.vo, optimize=True)
    # ) # h(abei)
    # H.ab.vvvv = (
    #                 np.einsum("xae,xbf->abef", H.chol.a.vv, H.chol.b.vv, optimize=True)
    # ) # h(abef)
    return H

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
    F = Z + G where G is sum_{i} <pi|v|qi>_A split for different
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
    F = Z + G where G is sum_{i} <pi|v|qi>_A split for different
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
    noa = system.noccupied_alpha + system.nfrozen
    nob = system.noccupied_beta + system.nfrozen

    # <p|f|q> = <p|z|q> + <pi|v|qi> + <pi~|v|qi~>
    f_a = e1int + (
        np.einsum("xpq,xii->pq", R_chol, R_chol[:, :noa, :noa], optimize=True) # <pi|v|qi>
        - np.einsum("xpi,xiq->pq", R_chol[:, :, :noa], R_chol[:, :noa, :], optimize=True) # <pi|v|iq>
        + np.einsum("xpq,xii->pq", R_chol, R_chol[:, :nob, :nob], optimize=True) # <pi~|v|qi~>
    )
    f_b = e1int + (
        np.einsum("xpq,xii->pq", R_chol, R_chol[:, :nob, :nob], optimize=True)  # <p~i~|v|q~i~>
        - np.einsum("xpi,xiq->pq", R_chol[:, :, :nob], R_chol[:, :nob, :], optimize=True)  # <p~i~|v|i~q~>
        + np.einsum("xii,xpq->pq", R_chol[:, :noa, :noa], R_chol, optimize=True)  # <ip~|v|iq~>
    )
    f = {"a": f_a, "b": f_b}
    return f
