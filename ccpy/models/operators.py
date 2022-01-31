from typing import Any, List, Dict
from pydantic import BaseModel

import numpy as np


class Operator(BaseModel):

    name: str
    spin_type: int
    array: Any

# class Operator:
#
#     def __init__(self, name, spin_type, array):
#         self.name = name
#         self.spin_type = spin_type
#         self.array = array



class ClusterOperator:

    def __init__(self, system, order, data_type=np.float64):
        self.order = order
        self.spin_cases = []
        self.dimensions = []
        ndim = 0
        for i in range(1, order + 1):
            for j in range(i + 1):
                name = get_operator_name(i, j)
                dimensions = get_operator_dimension(i, j, system)
                self.__dict__[name] = np.zeros(dimensions, dtype=data_type)
                self.spin_cases.append(name)
                self.dimensions.append(dimensions)
                ndim += np.prod(dimensions)
        self.ndim = ndim

    def flatten(self):
        return np.hstack([self.__getattribute__(key).flatten() for key in self.spin_cases])

    def unflatten(self, T_flat):
        prev = 0
        for dims, name in zip(self.dimensions, self.spin_cases):
            ndim = np.prod(dims)
            self.__dict__[name] = np.reshape(T_flat[prev : ndim + prev], dims)
            prev += ndim


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

    from ccpy.interfaces.pyscf_tools import loadFromPyscfMolecular
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
    system, H = loadFromPyscfMolecular(mf, nfrozen)

    print(system)

    order = 4
    T = ClusterOperator(system, order)
    print('Cluster operator order',order)
    print('---------------------------')
    for key in T.spin_cases:
        print(key,'->',T.__getattribute__(key).shape)
    print('Flattened dimension = ',T.ndim)
    print(T.flatten().shape)






