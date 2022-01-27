import numpy as np
from dataclasses import dataclass

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

@dataclass
class SortedIntegral:


class OneBodyIntegral:
    def __init__(self, sys, data_type=np.float64):
        self.oo = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha), dtype=data_type)
        self.ov = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha), dtype=data_type)
        self.vo = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha), dtype=data_type)
        self.vv = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha), dtype=data_type)

    def sortIntegrals(self, z, slice_occ, slice_unocc):
        self.oo = z[slice_occ, slice_occ]
        self.ov = z[slice_occ, slice_unocc]
        self.vo = z[slice_unocc, slice_occ]
        self.vv = z[slice_unocc, slice_unocc]


class TwoBodyIntegral:
    def __init__(self, sys, spin_type, data_type=np.float64):

        if spin_type == 'aaaa':
            self.oooo = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha),
                                 dtype=data_type)
            self.oovo = np.zeros((sys.noccupied_alpha, sys.noccupied_alpha, sys.nunoccupied_alpha, sys.noccupied_alpha),
                                 dtype=data_type)
            self.vooo = np.zeros((sys.nunoccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha),
                                 dtype=data_type)
            self.vvoo = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha),
                dtype=data_type)
            self.voov = np.zeros(
                (sys.nunoccupied_alpha, sys.noccupied_alpha, sys.noccupied_alpha, sys.nunoccupied_alpha),
                dtype=data_type)
            self.oovv = np.zeros(
                (sys.noccupied_alpha, sys.noccupied_alpha, sys.nunoccupied_alpha, sys.nunoccupied_alpha),
                dtype=data_type)
            self.vvov = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_alpha, sys.noccupied_alpha, sys.nunoccupied_alpha),
                dtype=data_type)
            self.vovv = np.zeros(
                (sys.nunoccupied_alpha, sys.noccupied_alpha, sys.nunoccupied_alpha, sys.nunoccupied_alpha),
                dtype=data_type)
            self.vvvv = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_alpha, sys.nunoccupied_alpha, sys.nunoccupied_alpha),
                dtype=data_type)

        if spin_type == 'bbbb':
            self.oooo = np.zeros((sys.noccupied_beta, sys.noccupied_beta, sys.noccupied_beta, sys.noccupied_beta),
                                 dtype=data_type)
            self.oovo = np.zeros((sys.noccupied_beta, sys.noccupied_beta, sys.nunoccupied_beta, sys.noccupied_beta),
                                 dtype=data_type)
            self.vooo = np.zeros((sys.nunoccupied_beta, sys.noccupied_beta, sys.noccupied_beta, sys.noccupied_beta),
                                 dtype=data_type)
            self.vvoo = np.zeros((sys.nunoccupied_beta, sys.nunoccupied_beta, sys.noccupied_beta, sys.noccupied_beta),
                                 dtype=data_type)
            self.voov = np.zeros((sys.nunoccupied_beta, sys.noccupied_beta, sys.noccupied_beta, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.oovv = np.zeros((sys.noccupied_beta, sys.noccupied_beta, sys.nunoccupied_beta, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.vvov = np.zeros((sys.nunoccupied_beta, sys.nunoccupied_beta, sys.noccupied_beta, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.vovv = np.zeros((sys.nunoccupied_beta, sys.noccupied_beta, sys.nunoccupied_beta, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.vvvv = np.zeros(
                (sys.nunoccupied_beta, sys.nunoccupied_beta, sys.nunoccupied_beta, sys.nunoccupied_beta),
                dtype=data_type)

        if spin_type == 'abab':
            self.oooo = np.zeros((sys.noccupied_alpha, sys.noccupied_beta, sys.noccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.oovo = np.zeros((sys.noccupied_alpha, sys.noccupied_beta, sys.nunoccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.ooov = np.zeros((sys.noccupied_alpha, sys.noccupied_beta, sys.noccupied_alpha, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.vooo = np.zeros((sys.nunoccupied_alpha, sys.noccupied_beta, sys.noccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.ovoo = np.zeros((sys.noccupied_alpha, sys.nunoccupied_beta, sys.noccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.vvoo = np.zeros((sys.nunoccupied_alpha, sys.nunoccupied_beta, sys.noccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.voov = np.zeros((sys.nunoccupied_alpha, sys.noccupied_beta, sys.noccupied_alpha, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.ovvo = np.zeros((sys.noccupied_alpha, sys.nunoccupied_beta, sys.nunoccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.vovo = np.zeros((sys.nunoccupied_alpha, sys.noccupied_beta, sys.nunoccupied_alpha, sys.noccupied_beta),
                                 dtype=data_type)
            self.ovov = np.zeros((sys.noccupied_alpha, sys.nunoccupied_beta, sys.noccupied_alpha, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.oovv = np.zeros((sys.noccupied_alpha, sys.noccupied_beta, sys.nunoccupied_alpha, sys.nunoccupied_beta),
                                 dtype=data_type)
            self.vvov = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_beta, sys.noccupied_alpha, sys.nunoccupied_beta),
                dtype=data_type)
            self.vvvo = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_beta, sys.nunoccupied_alpha, sys.noccupied_beta),
                dtype=data_type)
            self.vovv = np.zeros(
                (sys.nunoccupied_alpha, sys.noccupied_beta, sys.nunoccupied_alpha, sys.nunoccupied_beta),
                dtype=data_type)
            self.ovvv = np.zeros(
                (sys.noccupied_alpha, sys.nunoccupied_beta, sys.nunoccupied_alpha, sys.nunoccupied_beta),
                dtype=data_type)
            self.vvvv = np.zeros(
                (sys.nunoccupied_alpha, sys.nunoccupied_beta, sys.nunoccupied_alpha, sys.nunoccupied_beta),
                dtype=data_type)

        @classmethod
        def fromSliceFullMatrix(cls, v, **kwargs):
            self.oooo = v[slice_occ, slice_occ, slice_occ, slice_occ]
            self.oovo =
            self.vooo =
            self.vvoo =
            self.voov =
            self.oovv =
            self.vvov =
            self.vovv =
            self.vvvv =

            if
            self.oooo =
            self.oovo =
            self.ooov =
            self.vooo =
            self.ovoo =
            self.vvoo =
            self.voov =
            self.ovvo =
            self.vovo =
            self.ovov =
            self.oovv =
            self.vvov =
            self.vvvo =
            self.vovv =
            self.ovvv =
            self.vvvv =

class Hamiltonian:

    def __init__(self, H1A, H1B, H2A, H2B, H2C):
        self.aa = H1A
        self.bb = H1B
        self.aaaa = H2A
        self.abab = H2B
        self.bbbb = H2C

    @classmethod
    def fromPyscfMolecular(cls, meanFieldObj, nfrozen, normalOrdered=True):
        from pyscf import ao2mo

        norbitals = meanFieldObj.mo_coeff.shape[1]
        nuclearRepulsion = meanFieldObj.mol.energy_nuc()

        kineticAOIntegrals = meanFieldObj.mol.intor_symmetric('int1e_kin')
        nuclearAOIntegrals = meanFieldObj.mol.intor_symmetric('int1e_nuc')
        Z = np.einsum('pi,pq,qj->ij', meanFieldObj.mo_coeff, kineticAOIntegrals + nuclearAOIntegrals, meanFieldObj.mo_coeff)
        V = np.reshape(ao2mo.kernel(meanFieldObj.mol, meanFieldObj.mo_coeff, compact=False), (norbitals, norbitals, norbitals, norbitals))
        if notation != 'chemist':  # physics notation
            V = np.transpose(V, (0, 2, 1, 3))

        ehfCalculated = calculateHFEnergy(Z, V, meanFieldObj.mol.nelectron)
        ehfCalculated += nuclearRepulsion
        assert (np.allclose(ehfCalculated, meanFieldObj.energy_tot(), atol=1.0e-06, rtol=0.0))

        if not normalOrdered:
            return cls(Z1A, Z1B, V2A, V2B, V2C)
        else:


    @classmethod
    def fromPGFiles(cls, onebody_file, twobody_file, sys, normalOrdered=True):




    # @staticmethod
    # def dumpIntegralsToPGFiles(self):

def getNumberTotalOrbitals(onebody_file):

    with open(onebody_file) as f_in:
        lines = f_in.readlines()
        ct = 0
        for line in lines:
            ct += 1
    return int(-0.5 + np.sqrt(0.25 + 2*x))

def loadOnebodyIntegralFile(onebody_file, norbitalsTotal, data_type=np.float64):
    """This function reads the onebody.inp file from GAMESS
    and returns a numpy matrix.
    
    Parameters
    ----------
    filename : str
        Path to onebody integral file
    sys : dict
        System information dict
        
    Returns
    -------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody part of the bare Hamiltonian in the MO basis (Z)
    """
    e1int = np.zeros((norbitalsTotal, norbitalsTotal), dtype=data_type)
    try:
        with open(onebody_file) as f_in:
            lines = f_in.readlines()
            ct = 0
            for i in range(norbitalsTotal):
                for j in range(i + 1):
                    val = float(lines[ct].split()[0])
                    e1int[i, j] = val
                    e1int[j, i] = val
                    ct += 1
    except IOError:
        print('Error: {} does not appear to exist.'.format(onebody_file))
    return e1int


def loadTwobodyIntegralFile(twobody_file, norbitalsTotal, data_type=np.float64):
    """This function reads the twobody.inp file from GAMESS
    and returns a numpy matrix.
    
    Parameters
    ----------
    filename : str
        Path to twobody integral file
    sys : dict
        System information dict
        
    Returns
    -------
    e_nn : float
        Nuclear repulsion energy (in hartree)
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody part of the bare Hamiltonian in the MO basis (V)
    """
    try:
        # initialize numpy array
        e2int = np.zeros((norbitalsTotal, norbitalsTotal, norbitalsTotal, norbitalsTotal), dtype=data_type)
        # open file
        with open(twobody_file) as f_in:
            # loop over lines
            for line in f_in:
                # split fields and parse
                fields = line.split()
                indices = tuple(map(int, fields[:4]))
                val = float(fields[4])
                # check whether value is nuclear repulsion
                # fill matrix otherwise
                if sum(indices) == 0:
                    e_nn = val
                else:
                    indices = tuple(i - 1 for i in indices)
                    e2int[indices] = val
        # convert e2int from chemist notation (ia|jb) to
        # physicist notation <ij|ab>
        e2int = np.einsum('iajb->ijab', e2int)
    except IOError:
        print('Error: {} does not appear to exist.'.format(twobody_file))
    return e_nn, e2int


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
    v_aa = e2int - np.einsum("pqrs->pqsr", e2int)
    v_ab = e2int
    v_bb = e2int - np.einsum('pqrs->pqsr', e2int)

    v = {
        "A": v_aa,
        "B": v_ab,
        "C": v_bb
    }

    return v


def build_f(e1int, v, sys):
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
    Nocc_a = sys['Nocc_a'] + sys['Nfroz']
    Nocc_b = sys['Nocc_b'] + sys['Nfroz']

    # <p|f|q> = <p|z|q> + <pi|v|qi> + <pi~|v|qi~>
    f_a = e1int + np.einsum('piqi->pq', v['A'][:, :Nocc_a, :, :Nocc_a]) + np.einsum('piqi->pq',
                                                                                    v['B'][:, :Nocc_b, :, :Nocc_b])

    # <p~|f|q~> = <p~|z|q~> + <p~i~|v|q~i~> + <ip~|v|iq~>
    f_b = e1int + np.einsum('piqi->pq', v['C'][:, :Nocc_b, :, :Nocc_b]) + np.einsum('ipiq->pq',
                                                                                    v['B'][:Nocc_a, :, :Nocc_a, :])

    f = {
        "A": f_a,
        "B": f_b
    }

    return f

if __name__ == '__main__':
    Z = OneBodyIntegral()
