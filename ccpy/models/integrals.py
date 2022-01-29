import numpy as np
from ccpy.models.operators import get_operator_name
from itertools import combinations

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

class SortedIntegral:

    def __init__(self, system, name, matrix):

        order = len(name)
        double_spin_string = list(name) * 2
        slice_table = {'a': {'o': slice(0, system.noccupied_alpha), 'v': slice(system.noccupied_alpha, system.norbitals)},
                       'b': {'o': slice(0, system.noccupied_beta),  'v': slice(system.noccupied_beta, system.norbitals)}}

        #self.slices = []
        for i in range(2*order+1):
            for combs in combinations(range(2*order),i):
                attr = ['o'] * (2*order)
                for k in combs:
                    attr[k] = 'v'
                slicearr = [slice(None)] * (2*order)
                for k in range(2*order):
                    slicearr[k] = slice_table[double_spin_string[k]][attr[k]]
                self.__dict__[''.join(attr)] = matrix[tuple(slicearr)]
                #self.slices.append(''.join(attr))

class Integral:

    def __init__(self, system, order, matrices):
        self.order = order
        for i in range(1, order + 1): # Loop over many-body ranks
            for j in range(i + 1): # Loop over distinct spin cases per rank
                name = get_operator_name(i, j)
                sorted_integral = SortedIntegral(system, name, matrices[name])
                self.__dict__[name] = sorted_integral

    @classmethod
    def fromEmpty(cls, system, order, data_type=np.float64):
        matrices = {}
        for i in range(1, order + 1): # Loop over many-body ranks
            for j in range(i + 1): # Loop over distinct spin cases per rank
                name = get_operator_name(i, j)
                dimension = [system.norbitals] * (2*order)
                matrices[name] = np.zeros(dimension, dtype=data_type)
        return cls(system, order, matrices)

def getHamiltonian(e1int, e2int, system, normal_ordered):

    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)

    twobody = build_v(e2int)
    if normal_ordered:
        onebody = build_f(e1int, twobody, system)
    else:
        onebody = {'a' : e1int, 'b' : e1int}
    # Keep only correlated spatial orbitals in the one- and two-body matrices
    onebody['a'] = onebody['a'][corr_slice, corr_slice]
    onebody['b'] = onebody['b'][corr_slice, corr_slice]
    twobody['aa'] = twobody['aa'][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody['ab'] = twobody['ab'][corr_slice, corr_slice, corr_slice, corr_slice]
    twobody['bb'] = twobody['bb'][corr_slice, corr_slice, corr_slice, corr_slice]

    return Integral(system, 2, {**onebody, **twobody})

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
    v =  {
        "aa": e2int - np.einsum("pqrs->pqsr", e2int),
        "ab": e2int,
        "bb": e2int - np.einsum('pqrs->pqsr', e2int)
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
    f_a = e1int + np.einsum('piqi->pq', v['aa'][:, :Nocc_a, :, :Nocc_a]) + np.einsum('piqi->pq',
                                                                                    v['ab'][:, :Nocc_b, :, :Nocc_b])

    # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    f_b = e1int + np.einsum('piqi->pq', v['bb'][:, :Nocc_b, :, :Nocc_b]) + np.einsum('ipiq->pq',
                                                                                    v['ab'][:Nocc_a, :, :Nocc_a, :])

    f = {
        "a": f_a,
        "b": f_b
    }

    return f

if __name__ == '__main__':

    from ccpy.interfaces.pyscf_tools import loadFromPyscfMolecular
    from ccpy.interfaces.gamess_tools import loadFromGamess
    from pyscf import gto, scf

    # Testing from PySCF
    mol = gto.Mole()
    mol.build(
        atom='''F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816''',
        basis='ccpvdz',
        charge=1,
        spin=1,
        symmetry='D2H',
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 0
    system, H = loadFromPyscfMolecular(mf, nfrozen, dumpIntegrals=False)
    print(system)

    # Testing HF energy using F and V, only works if nfrozen = 0
    hf_energy = np.einsum('ii->', H.a.oo, optimize=True)
    hf_energy += np.einsum('ii->', H.b.oo, optimize=True)
    hf_energy -= 0.5*np.einsum('ijij->', H.aa.oooo, optimize=True)
    hf_energy -= np.einsum('ijij->', H.ab.oooo, optimize=True)
    hf_energy -= 0.5*np.einsum('ijij->', H.bb.oooo, optimize=True)
    hf_energy += system.nuclear_repulsion
    assert(np.allclose(hf_energy, mf.energy_tot(), atol=1.0e-06, rtol=0.0))

    # Testing from GAMESS
    nfrozen = 0
    gamess_logfile = "/Users/harellab/Documents/ccpy/tests/F2+-1.0-631g/F2+-1.0-631g.log"
    onebody_file = "/Users/harellab/Documents/ccpy/tests/F2+-1.0-631g/onebody.inp"
    twobody_file = "/Users/harellab/Documents/ccpy/tests/F2+-1.0-631g/twobody.inp"

    system, H = loadFromGamess(gamess_logfile, onebody_file, twobody_file, nfrozen, normal_ordered=True, data_type=np.float64)

    print(system)

    # Testing HF energy using F and V, only works if nfrozen = 0
    hf_energy = np.einsum('ii->', H.a.oo, optimize=True)
    hf_energy += np.einsum('ii->', H.b.oo, optimize=True)
    hf_energy -= 0.5 * np.einsum('ijij->', H.aa.oooo, optimize=True)
    hf_energy -= np.einsum('ijij->', H.ab.oooo, optimize=True)
    hf_energy -= 0.5 * np.einsum('ijij->', H.bb.oooo, optimize=True)
    hf_energy += system.nuclear_repulsion
    assert (np.allclose(hf_energy, -198.0361965498, atol=1.0e-06, rtol=0.0))
