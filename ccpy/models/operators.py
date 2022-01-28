from typing import Any, List, Dict
from pydantic import BaseModel

import numpy as np
from itertools import combinations

#from ccpy.drivers.solvers import diis


class Operator(BaseModel):

    name: str
    spin_type: int
    array: Any

class SlicedOperator:

    def __init__(self, system, name):
        dimensions_table = {'a': {'o': system.noccupied_alpha, 'v': system.nunoccupied_alpha},
                            'b': {'o': system.noccupied_beta,  'v': system.nunoccupied_beta}}
        order = len(name)
        double_spin_string = list(name) * 2
        for i in range(2*order+1):
            for combs in combinations(range(2*order),i):
                attr = ['o'] * (2*order)
                dimensions = [0] * 2*order
                for k in combs:
                    attr[k] = 'v'
                for k in range(2*order):
                    dimensions[k] = dimensions_table[double_spin_string[k]][attr[k]]
                self.__dict__[''.join(attr)] = np.zeros(dimensions)

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
                self.__dict__[name] = SlicedOperator(system, name)

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
    system, e1int, e2int = parsePyscfMolecularMeanField(mf, nfrozen)

    print(system)

    T = ClusterOperator(system, 3)

    for key, values in T.T.items():
        print(key,'->',values.array.shape)

    print('Testing 2-body operator:')
    H = ManyBodyOperator(system, 2)
    print('1-body part: AA')
    print(H.a.oo.shape)
    print(H.a.ov.shape)
    print(H.a.vo.shape)
    print(H.a.vv.shape)
    print('1-body part: BB')
    print(H.b.oo.shape)
    print(H.b.ov.shape)
    print(H.b.vo.shape)
    print(H.b.vv.shape)
    print('2-body part: ABAB')
    print(H.ab.oooo.shape)
    print(H.ab.vooo.shape)
    print(H.ab.vooo.shape)




