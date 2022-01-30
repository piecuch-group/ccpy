import datetime
from art import tprint

whitespace = '  '

def ccpy_header():
    # twisted, sub-zero, swampland, starwars
    tprint('   ccpy', 'twisted')
    print(whitespace,'Authors:')
    print(whitespace,' Karthik Gururangan (gururang@msu.edu)')
    print(whitespace,' J. Emiliano Deustua (edeustua@caltech.edu)')
    print(whitespace,' Affiliated with the Piecuch Group at Michigan State University')
    print('\n')

class SystemPrinter:

    def __init__(self, system):
        self.system = system

    def header(self):

        print(whitespace,'System Information:')
        print(whitespace,'----------------------------------------------------')
        print(whitespace,'  Number of correlated electrons =', self.system.nelectrons)
        print(whitespace,'  Number of correlated orbitals =', self.system.norbitals)
        print(whitespace,'  Number of frozen orbitals =', self.system.nfrozen)
        print(whitespace,'  Number of alpha occupied orbitals =', self.system.noccupied_alpha)
        print(whitespace,'  Number of alpha unoccupied orbitals =', self.system.nunoccupied_alpha)
        print(whitespace,'  Number of beta occupied orbitals =', self.system.noccupied_beta)
        print(whitespace,'  Number of beta unoccupied orbitals =', self.system.nunoccupied_beta)
        print(whitespace,'  Charge =', self.system.charge)
        print(whitespace,'  Point group =', self.system.point_group)
        print(whitespace,'  Spin multiplicity of reference =', self.system.multiplicity)
        print('')
        print(whitespace,'    MO #      Energy (a.u.)   Symmetry    Occupation')
        print(whitespace,'----------------------------------------------------')
        for i in range(self.system.norbitals + self.system.nfrozen):
            print(whitespace,'     {}       {:>6f}          {}         {}'.format(i+1,
                                                                                  self.system.mo_energies[i],
                                                                                  self.system.orbital_symmetries[i],
                                                                                  self.system.mo_occupation[i]))
        print('')
        print(whitespace,'Nuclear Repulsion Energy =', self.system.nuclear_repulsion)
        print(whitespace,'Reference Energy =', self.system.reference_energy)
        print('')



class CCPrinter:

    def __init__(self, calculation):
        self.calculation = calculation

    def header(self):
        print(whitespace,'--------------------------------------------------')
        print(whitespace,'Calculation type = ',self.calculation.calculation_type.upper())
        print(whitespace,'Maximum iterations =',self.calculation.maximum_iterations)
        print(whitespace,'Convergence tolerance =',self.calculation.convergence_tolerance)
        print(whitespace,'Energy shift =',self.calculation.level_shift)
        print(whitespace,'DIIS size = ',self.calculation.diis_size)
        print(whitespace,'RHF symmetry =',self.calculation.RHF_symmetry)
        print(whitespace,'--------------------------------------------------')
        print('')
        print(whitespace,'CC calculation started at',\
              datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),\
              '%Y-%m-%d %H:%M'),'\n')


    @staticmethod
    def calculation_summary(reference_energy, cc_energy):
        print('')
        print(whitespace,'CC Calculation Summary')
        print(whitespace,'----------------------------------------')
        print(whitespace,' Reference energy = ', reference_energy)
        print(whitespace,' CC correlation energy = ', cc_energy)
        print(whitespace,' Total CC energy = ', reference_energy + cc_energy,'\n')
        print(whitespace,'CC calculation ended at',\
              datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),\
              '%Y-%m-%d %H:%M:%S'),'\n')

