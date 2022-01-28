from typing import Any, List, Dict
from pydantic import BaseModel

import numpy as np

#from ccpy.drivers.solvers import diis


class Operator(BaseModel):

    name: str
    spin_type: int
    array: Any


class DIIS:

    def __init__(self, T, diis_size=6):

        dim = 0
        for key, item in T.items():
            dim += item.size

        self.diis_size = diis_size
        self.T_list = np.zeros((dim, diis_size))
        self.T_residuum_list = np.zeros((dim, diis_size))

    def push(self, T, T_residuum, iteration):
        self.T_list[:, iteration % self.diis_size] = T
        self.T_residuum_list[:, iteration % self.diis_size] = T_residuum

    def extrapolate(self):
        return diis(self.T_list, self.T_residuum_list)

class ManyBodyOperator:

    def __init__(self, system, order):
        for i in range(1, order + 1):
            for j in range(i + 1):
                name = get_operator_name(i, j)
                dimensions = get_operator_dimension(i, j, system)
                self.__dict__[name] = np.zeros(dimensions)

    def __repr__(self):
        for key,value in vars(self).items():
            print('     ',key,'->',value)
        return ''

class ClusterOperator:

    def __init__(self, system, order):
        self.order = order
        self.T = build_cluster_expansion(system, order)

    def flatten(self):
        return np.hstack([t.flatten() for t in self.T.values()])




# TODO: move this to a classmethod or something
def get_operator_name(i, j):
    return "a" * (i - j)  + "b" * j

def get_operator_dimension(i, j, system):

    nocc_a = system.noccupied_alpha
    nocc_b = system.noccupied_beta
    nunocc_a = system.nunoccupied_alpha
    nunocc_b = system.nunoccupied_beta

    ket = [nunocc_a] * (i - j) + [nunocc_b] * j
    bra = [nocc_a] * (i - j) + [nocc_b] * j

    return ket + bra

def build_cluster_expansion(system, order):
    operators = dict()

    for i in range(1, order+1):
        for j in range(i+1):
            name = get_operator_name(i, j)
            dimensions = get_operator_dimension(i, j, system)
            operators[name] = Operator(name=name, spin_type=j, array=np.zeros(dimensions))

    return operators

if __name__ == "__main__":

    from ccpy.interfaces.pyscf_tools import parsePyscfMolecularMeanField
    from pyscf import gto, scf

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

    nfrozen = 2
    system = parsePyscfMolecularMeanField(mf, nfrozen)

    print(system)

    T = ClusterOperator(system, 3)

    for key, values in T.T.items():
        print(key,'->',values.array.shape)

    M = ManyBodyOperator(system, 3)

    print(dir(M))


