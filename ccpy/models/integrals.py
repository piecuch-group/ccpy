import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from itertools import combinations
from ccpy.models.operators import get_operator_name

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

    @classmethod
    def from_none(cls, system, order):
        return cls(system, order, matrix=None, sorted=True, use_none=True)

def getHamiltonian(e1int, e2int, system, normal_ordered, sorted=True):

    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)

    twobody = build_v(e2int)
    if normal_ordered:
        onebody = build_f({"a": e1int, "b": e1int}, twobody, system)
    else:
        onebody = {"a": e1int, "b": e1int}
    # Keep only correlated spatial orbitals in the one- and two-body matrices
    onebody["a"] = onebody["a"][corr_slice, corr_slice]
    onebody["b"] = onebody["b"][corr_slice, corr_slice]
    twobody["aa"] = twobody["aa"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["ab"] = twobody["ab"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["bb"] = twobody["bb"][corr_slice, corr_slice, corr_slice, corr_slice]

    return Integral(system, 2, {**onebody, **twobody}, sorted=sorted)

def getUHFHamiltonian(coeff_a, coeff_b, z_ao, v_ao, system, normal_ordered, sorted=True):
    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)
    # Transform 1-electron integrals
    z_a = ccpy_einsum("ij,ip,jq->pq", z_ao, coeff_a, coeff_a)
    z_b = ccpy_einsum("ij,ip,jq->pq", z_ao, coeff_b, coeff_b)
    z = {"a": z_a, "b": z_b}
    # Transform 2-electron integrals
    v_aa = ccpy_einsum("ijkl,ip,jq,kr,ls->pqrs", v_ao, coeff_a, coeff_a, coeff_a, coeff_a)
    v_aa -= np.transpose(v_aa, (0, 1, 3, 2))
    v_ab = ccpy_einsum("ijkl,ip,jq,kr,ls->pqrs", v_ao, coeff_a, coeff_b, coeff_a, coeff_b)
    v_bb = ccpy_einsum("ijkl,ip,jq,kr,ls->pqrs", v_ao, coeff_b, coeff_b, coeff_b, coeff_b)
    v_bb -= np.transpose(v_bb, (0, 1, 3, 2))
    v = {"aa": v_aa, "ab": v_ab, "bb": v_bb}
    # Build Fock matrix
    if normal_ordered:
        fock = build_f(z, v, system)
    else:
        fock = z
    # Keep only correlated spatial orbitals in the one- and two-body matrices
    onebody = {}
    twobody = {}
    onebody["a"] = fock["a"][corr_slice, corr_slice]
    onebody["b"] = fock["b"][corr_slice, corr_slice]
    twobody["aa"] = v["aa"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["ab"] = v["ab"][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody["bb"] = v["bb"][corr_slice, corr_slice, corr_slice, corr_slice]
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
                    ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo)
                    - ccpy_einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo)
    ) # h(mnij)
    H.aa.ooov = (
                    ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov)
                    - ccpy_einsum("xni,xme->mnie", H.chol.a.oo, H.chol.a.ov)
    ) # h(mnie)
    H.aa.vooo = (
                    ccpy_einsum("xai,xmj->amij", H.chol.a.vo, H.chol.a.oo)
                    - ccpy_einsum("xaj,xmi->amij", H.chol.a.vo, H.chol.a.oo)
    ) # h(amij)
    H.aa.voov = (
                    ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov)
                    - ccpy_einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo)
    ) # h(amie)
    H.aa.oovv = (
                    ccpy_einsum("xme,xnf->mnef", H.chol.a.ov, H.chol.a.ov)
                    - ccpy_einsum("xmf,xne->mnef", H.chol.a.ov, H.chol.a.ov)
    ) # h(mnef)
    H.aa.vvoo = (
                    ccpy_einsum("xai,xbj->abij", H.chol.a.vo, H.chol.a.vo)
                    - ccpy_einsum("xaj,xbi->abij", H.chol.a.vo, H.chol.a.vo)
    ) # h(abij)
    # H.aa.vovv = (
    #                 ccpy_einsum("xae,xmf->amef", H.chol.a.vv, H.chol.a.ov)
    #                 - ccpy_einsum("xaf,xme->amef", H.chol.a.vv, H.chol.a.ov)
    # ) # h(amef)
    # H.aa.vvov = (
    #                 ccpy_einsum("xai,xbe->abie", H.chol.a.vo, H.chol.a.vv)
    #                 - ccpy_einsum("xbi,xae->abie", H.chol.a.vo, H.chol.a.vv)
    # ) # h(abie)
    # H.aa.vvvv = (
    #                 ccpy_einsum("xae,xbf->abef", H.chol.a.vv, H.chol.a.vv)
    #                 - ccpy_einsum("xaf,xbe->abef", H.chol.a.vv, H.chol.a.vv)
    # ) # h(abef)
    # ---
    # bb
    # ---
    H.bb.oooo = (
                    ccpy_einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo)
                    - ccpy_einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo)
    ) # h(mnij)
    H.bb.ooov = (
                    ccpy_einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov)
                    - ccpy_einsum("xni,xme->mnie", H.chol.b.oo, H.chol.b.ov)
    ) # h(mnie)
    H.bb.vooo = (
                    ccpy_einsum("xai,xmj->amij", H.chol.b.vo, H.chol.b.oo)
                    - ccpy_einsum("xaj,xmi->amij", H.chol.b.vo, H.chol.b.oo)
    ) # h(amij)
    H.bb.voov = (
                    ccpy_einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov)
                    - ccpy_einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo)
    ) # h(amie)
    H.bb.oovv = (
                    ccpy_einsum("xme,xnf->mnef", H.chol.b.ov, H.chol.b.ov)
                    - ccpy_einsum("xmf,xne->mnef", H.chol.b.ov, H.chol.b.ov)
    ) # h(mnef)
    H.bb.vvoo = (
                    ccpy_einsum("xai,xbj->abij", H.chol.b.vo, H.chol.b.vo)
                    - ccpy_einsum("xaj,xbi->abij", H.chol.b.vo, H.chol.b.vo)
    ) # h(abij)
    # H.bb.vovv = (
    #                 ccpy_einsum("xae,xmf->amef", H.chol.b.vv, H.chol.b.ov)
    #                 - ccpy_einsum("xaf,xme->amef", H.chol.b.vv, H.chol.b.ov)
    # ) # h(amef)
    # H.bb.vvov = (
    #                 ccpy_einsum("xai,xbe->abie", H.chol.b.vo, H.chol.b.vv)
    #                 - ccpy_einsum("xbi,xae->abie", H.chol.b.vo, H.chol.b.vv)
    # ) # h(abie)
    # H.bb.vvvv = (
    #                 ccpy_einsum("xae,xbf->abef", H.chol.b.vv, H.chol.b.vv)
    #                 - ccpy_einsum("xaf,xbe->abef", H.chol.b.vv, H.chol.b.vv)
    # ) # h(abef)
    # ---
    # ab
    # ---
    H.ab.oooo = (
                    ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo)
    ) # h(mnij)
    H.ab.ooov = (
                    ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov)
    ) # h(mnie)
    H.ab.oovo = (
                    ccpy_einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo)
    ) # h(mnei)
    H.ab.vooo = (
                    ccpy_einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo)
    ) # h(amij)
    H.ab.ovoo = (
                    ccpy_einsum("xmj,xai->maji", H.chol.a.oo, H.chol.b.vo)
    ) # h(maji)
    H.ab.voov = (
                    ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov)
    ) # h(amie)
    H.ab.ovvo = (
                    ccpy_einsum("xme,xai->maei", H.chol.a.ov, H.chol.b.vo)
    ) # h(maei)
    H.ab.vovo = (
                    ccpy_einsum("xae,xmi->amei", H.chol.a.vv, H.chol.b.oo)
    ) # h(amei)
    H.ab.ovov = (
                    ccpy_einsum("xmj,xbe->mbje", H.chol.a.oo, H.chol.b.vv)
    ) # h(mbje)
    H.ab.oovv = (
                    ccpy_einsum("xme,xnf->mnef", H.chol.a.ov, H.chol.b.ov)
    ) # h(mnef)
    H.ab.vvoo = (
                    ccpy_einsum("xai,xbj->abij", H.chol.a.vo, H.chol.b.vo)
    ) # h(abij)
    # H.ab.vovv = (
    #                 ccpy_einsum("xae,xmf->amef", H.chol.a.vv, H.chol.b.ov)
    # ) # h(amef)
    # H.ab.ovvv = (
    #                 ccpy_einsum("xmf,xae->mafe", H.chol.a.ov, H.chol.b.vv)
    # ) # h(mafe)
    # H.ab.vvov = (
    #                 ccpy_einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv)
    # ) # h(abie)
    # H.ab.vvvo = (
    #                 ccpy_einsum("xae,xbi->abei", H.chol.a.vv, H.chol.b.vo)
    # ) # h(abei)
    # H.ab.vvvv = (
    #                 ccpy_einsum("xae,xbf->abef", H.chol.a.vv, H.chol.b.vv)
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
        "aa": e2int - ccpy_einsum("pqrs->pqsr", e2int),
        "ab": e2int,
        "bb": e2int - ccpy_einsum("pqrs->pqsr", e2int),
    }
    return v


def build_f(z, v, system):
    """This function generates the Fock matrix using the formula
    F = Z + G where G is sum_{i} <pi|v|qi>_A split for different
    spin cases.

    Parameters
    ----------
    z : dict 
        Onebody integral dictionary
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
        z["a"]
        + ccpy_einsum("piqi->pq", v["aa"][:, :Nocc_a, :, :Nocc_a])
        + ccpy_einsum("piqi->pq", v["ab"][:, :Nocc_b, :, :Nocc_b])
    )

    # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    f_b = (
        z["b"]
        + ccpy_einsum("piqi->pq", v["bb"][:, :Nocc_b, :, :Nocc_b])
        + ccpy_einsum("ipiq->pq", v["ab"][:Nocc_a, :, :Nocc_a, :])
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
        ccpy_einsum("xpq,xii->pq", R_chol, R_chol[:, :noa, :noa]) # <pi|v|qi>
        - ccpy_einsum("xpi,xiq->pq", R_chol[:, :, :noa], R_chol[:, :noa, :]) # <pi|v|iq>
        + ccpy_einsum("xpq,xii->pq", R_chol, R_chol[:, :nob, :nob]) # <pi~|v|qi~>
    )
    f_b = e1int + (
        ccpy_einsum("xpq,xii->pq", R_chol, R_chol[:, :nob, :nob])  # <p~i~|v|q~i~>
        - ccpy_einsum("xpi,xiq->pq", R_chol[:, :, :nob], R_chol[:, :nob, :])  # <p~i~|v|i~q~>
        + ccpy_einsum("xii,xpq->pq", R_chol[:, :noa, :noa], R_chol)  # <ip~|v|iq~>
    )
    f = {"a": f_a, "b": f_b}
    return f
