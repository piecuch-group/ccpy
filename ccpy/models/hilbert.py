import numpy as np

from ccpy.models.operators import get_operator_name

# class for P (and Q) spaces.
# should contain:
#   - accessible list of relevant determinants in the excited Slater form (a1,a2,...,i1,i2,...)
#     organized by spincase
#   - functions that count the number of determinants of specific excitation level (S, D, T, Q, etc.)
#   - functions that sort the determinants of a given excitation level by point group symmetry
#   - easy ways of adding and removing determinants from the list

class HilbertSubspace:

    def __init__(self, system, order):
        self.order = order
        self.spin_cases = []
        self.dimensions = []
        ndim = 0
        for i in range(1, order + 1):
            for j in range(i + 1):
                name = get_operator_name(i, j)
                setattr(self, name, [])
                self.spin_cases.append(name)

        if order > 2:
            # add all singles and doubles from the get-go
            setattr(self, 'a', self.__getattribute__('a') +
                    [ [a, i] for a in range(system.nunoccupied_alpha) for i in range(system.noccupied_alpha)])
            setattr(self, 'b', self.__getattribute__('b') +
                    [ [a, i] for a in range(system.nunoccupied_beta) for i in range(system.noccupied_beta)])
            setattr(self, 'aa', self.__getattribute__('aa') +
                    [ [a, b, i, j] for a in range(system.nunoccupied_alpha) for b in range(a + 1, system.nunoccupied_alpha)
                                   for i in range(system.noccupied_alpha) for j in range(i + 1, system.noccupied_alpha)])
            setattr(self, 'ab', self.__getattribute__('ab') +
                    [ [a, b, i, j] for a in range(system.nunoccupied_alpha) for b in range(system.nunoccupied_beta)
                                   for i in range(system.noccupied_alpha) for j in range(system.noccupied_beta)])
            setattr(self, 'bb', self.__getattribute__('bb') +
                    [ [a, b, i, j] for a in range(system.nunoccupied_beta) for b in range(a + 1, system.nunoccupied_beta)
                                   for i in range(system.noccupied_beta) for j in range(i + 1, system.noccupied_beta)])

    def add_determinant(self, determinant, spincase):
        setattr(self, spincase, self.__getattribute__(spincase) + [determinant])

    def remove_determinant(self, determinant, spincase):
        setattr(self, spincase, [det for det in self.__getattribute__(spincase) if det != determinant])


if __name__ == "__main__":

    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvdz",
        charge=1,
        spin=1,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = load_pyscf_integrals(mf, nfrozen)

    pspace = HilbertSubspace(system, 3)

    print(pspace.aa)
    print('Removing determinant')
    pspace.remove_determinant([19, 20, 5, 6], 'aa')
    print(pspace.aa)