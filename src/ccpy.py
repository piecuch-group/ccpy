"""Driver module for the CCpy program suite.
Call this script with a supplied input file as
    python ccpy.py <input>
"""
import os
import argparse
from parser_module import parse_input_file
from system import build_system
from integrals import get_integrals
from calc_driver_main import calc_driver_main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='parser for Python CC implementation')
    parser.add_argument('input_file',type=str,help='Path to input file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print('ERROR: Input {} not found!'.format(args.input_file))
    else:
        input_path = os.path.abspath(args.input_file)

    inputs = parse_input_file(input_path)

    sys = build_system(inputs['gamess_file'],inputs['nfroz'])
    print('System Information:')
    print('-------------------------------------------------')
    print('  Number of correlated electrons = {}'.format(sys['Nelec']))
    print('  Number of frozen electrons = {}'.format(2*inputs['nfroz']))
    print('  Number of alpha occupied orbitals = {}'.format(sys['Nocc_a']))
    print('  Number of alpha unoccupied orbitals = {}'.format(sys['Nunocc_a']))
    print('  Number of beta occupied orbitals = {}'.format(sys['Nocc_b']))
    print('  Number of beta unoccupied orbitals = {}'.format(sys['Nunocc_b']))
    print('  Charge = {}'.format(sys['charge']))
    print('  Point group = {}'.format(sys['point_group']))
    print('  Spin multiplicity of reference = {}'.format(sys['multiplicity']))
    print('')
    print('CC Calculation Settings:')
    print('-------------------------------------------------')
    print('  Calculation Type = {}'.format(inputs['calc_type']))
    print('  CC shift = {} hartree'.format(inputs['ccshift']))
    print('  Left-CC shift = {} hartree'.format(inputs['lccshift']))
    print('  Maxit = {}'.format(inputs['maxit']))
    print('  Tolerance = {}'.format(inputs['tol']))
    if 'EOM' in inputs['calc_type']:
        print('  Number of roots = {}'.format(inputs['nroot']))
        print('  EOMCC Tolerance = {}'.format(inputs['eom_tol']))
        print('  EOMCC Maxit = {}'.format(inputs['eom_maxit']))
        print('  EOMCC Initial Guess = {}'.format(inputs['eom_init']))
        print('  Left-EOMCC Shift = {}'.format(inputs['eom_lccshift']))
        if inputs['root_select'] is not None:
            print('  Selected roots = {}'.format(inputs['root_select']))

    print('')
    print('    MO #      Energy (a.u.)   Symmetry    Occupation')
    print('-------------------------------------------------')
    for i in range(sys['Norb']):
        print('     {}       {:>6f}          {}         {}'.format(i+1,sys['mo_energy'][i],sys['sym'][i],sys['mo_occ'][i]))

    ints = get_integrals(inputs['onebody_file'],inputs['twobody_file'],sys)
    print('')
    print('  Nuclear Repulsion Energy = {} HARTREE'.format(ints['Vnuc']))
    print('  Reference Energy = {} HARTREE'.format(ints['Escf']))

    
    calc_driver_main(inputs,sys,ints)

