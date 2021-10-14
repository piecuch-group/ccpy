import os

os.environ['OPENBLAS_NUM_THREADS'] = '10'

import argparse
import numpy as np
from system import build_system
from integrals import get_integrals
from parser_module import parse_input_file
from symmetry_count import get_symmetry_count

def calc_driver_main(inputs,sys,ints):

    calc_type = inputs['calc_type']

    if calc_type == 'mp2' or calc_type == 'MP2':
        from mp2_module import mp2
        Emp2 = mp2(sys,ints)

    if calc_type == 'ccs' or calc_type == 'CCS':
        from ccs_module import ccs
        cc_t, Eccs = ccs(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        
    if calc_type == 'ccd' or calc_type == 'CCD':
        from ccd_module import ccd
        cc_t, Eccd = ccd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

    if calc_type == 'accd' or calc_type == 'ACCD':
        from accd_module import accd
        cc_t, Eaccd = accd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

    if calc_type == 'ccsd' or calc_type == 'CCSD':
        from ccsd_module import ccsd
        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

    if calc_type == 'eomccsd' or calc_type == 'EOMCCSD':
        from ccsd_module import ccsd
        from eomccsd_module import eomccsd
        from HBar_module import HBar_CCSD

        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        
        cc_t, omega = eomccsd(inputs['nroot'],H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,\
                        tol=inputs['eom_tol'],initial_guess=inputs['eom_init'],\
                        maxit=inputs['eom_maxit'])

    if calc_type == 'ccsdt' or calc_type == 'CCSDT':
        from ccsdt_module import ccsdt
        cc_t, Eccsdt = ccsdt(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'],flag_save=inputs['save_data'],save_location=inputs['work_dir'])

    if calc_type == 'eomccsdt' or calc_type == 'EOMCCSDT':
        from ccsdt_module import ccsdt
        from HBar_module import HBar_CCSDT
        from eomccsdt_module import eomccsdt, test_updates

        cc_t, Eccsdt = ccsdt(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'],flag_save=inputs['save_data'],save_location=inputs['work_dir'])

        H1A,H1B,H2A,H2B,H2C = HBar_CCSDT(cc_t,ints,sys)
        
        #matfile = inputs['work_dir']+'/Rvec-h2o-631g.mat'
        #test_updates(matfile,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        cc_t, omega = eomccsdt(inputs['nroot'],H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,\
                        tol=inputs['eom_tol'],initial_guess=inputs['eom_init'],\
                        maxit=inputs['eom_maxit'])

    if calc_type == 'crcc23' or calc_type == 'CRCC23' or calc_type == 'CR-CC(2,3)':
        from ccsd_module import ccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc23_module import crcc23
        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,\
                        shift=inputs['lccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=inputs['isRHF'])

    if calc_type == 'crcc24' or calc_type == 'CRCC24' or calc_type == 'CR-CC(2,4)':
        from ccsd_module import ccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc24_module import crcc24
        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,\
                        shift=inputs['lccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        Ecrcc24,E24 = crcc24(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=inputs['isRHF'])

    if calc_type == 'creomcc23' or calc_type == 'CREOMCC23' or calc_type == 'CR-EOMCC(2,3)':
        from ccsd_module import ccsd
        from eomccsd_module import eomccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc23_module import crcc23

        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        
        cc_t, omega = eomccsd(inputs['nroot'],H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,\
                        tol=inputs['eom_tol'],initial_guess=inputs['eom_init'],\
                        maxit=inputs['eom_maxit'])
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,\
                        shift=inputs['lccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'],nroot=inputs['nroot'],omega=omega,\
                        eom_tol=inputs['eom_tol'],eom_lccshift=inputs['eom_lccshift'],eom_maxit=inputs['eom_maxit'])
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=inputs['isRHF'],\
                        nroot=inputs['nroot'],omega=omega)

    if calc_type == 'adaptive_ccpq_relaxed':
        from adaptive_ccpq_main import calc_adaptive_ccpq
        from symmetry_count import get_symmetry_count

        nexc = 3
        _,countref = get_symmetry_count(sys,nexc)

        Eccp,Eccpq = calc_adaptive_ccpq(sys,ints,countref,inputs['work_dir'],\
                        ccshift=inputs['ccshift'],lccshift=inputs['lccshift'],
                        maxit=inputs['adaptive_maxit'],growth_percentage=inputs['adaptive_growthperc'],\
                        ccmaxit=inputs['maxit'],tol=inputs['tol'],\
                        diis_size=inputs['diis_size'],isRHF=inputs['isRHF'],\
                        restart_dir=inputs['adaptive_restart_dir'],\
                        flag_save=inputs['save_data'])


    if calc_type == 'adaptive_ccpq_unrelaxed':
        from adaptive_ccpq_main import calc_adaptive_ccpq_norelax
        from symmetry_count import get_symmetry_count

        nexc = 3
        _,countref = get_symmetry_count(sys,nexc)

        Eccp,Eccpq = calc_adaptive_ccpq_norelax(sys,ints,countref,inputs['work_dir'],\
                        inputs['adaptive_triples_percentages'],\
                        ccshift=inputs['ccshift'],lccshift=inputs['lccshift'],tol=inputs['tol'],\
                        diis_size=inputs['diis_size'],ccmaxit=inputs['maxit'],isRHF=inputs['isRHF'],\
                        flag_save=inputs['save_data'])

    return

#if __name__ == '__main__':
#
#    parser = argparse.ArgumentParser(description='parser for Python CC implementation')
#    parser.add_argument('input_file',type=str,help='Path to input file')
#    args = parser.parse_args()
#
#    if not os.path.exists(args.input_file):
#        print('ERROR: Input {} not found!'.format(args.input_file))
#    else:
#        input_path = os.path.abspath(args.input_file)
#
#    inputs = parse_input_file(input_path)
#
#    sys = build_system(inputs['gamess_file'],inputs['nfroz'])
#    print('System Information:')
#    print('-------------------------------------------------')
#    print('  Number of correlated electrons = {}'.format(sys['Nelec']))
#    print('  Number of frozen electrons = {}'.format(2*inputs['nfroz']))
#    print('  Number of alpha occupied orbitals = {}'.format(sys['Nocc_a']))
#    print('  Number of alpha unoccupied orbitals = {}'.format(sys['Nunocc_a']))
#    print('  Number of beta occupied orbitals = {}'.format(sys['Nocc_b']))
#    print('  Number of beta unoccupied orbitals = {}'.format(sys['Nunocc_b']))
#    print('  Charge = {}'.format(sys['charge']))
#    print('  Point group = {}'.format(sys['point_group']))
#    print('  Spin multiplicity of reference = {}'.format(sys['multiplicity']))
#    print('')
#    print('CC Calculation Settings:')
#    print('-------------------------------------------------')
#    print('  CC shift = {} hartree'.format(inputs['ccshift']))
#    print('  Left-CC shift = {} hartree'.format(inputs['lccshift']))
#    print('  Maxit = {}'.format(inputs['maxit']))
#    print('  Tolerance = {}'.format(inputs['tol']))
#    print('  Calculation Type = {}'.format(inputs['calc_type']))
#    print('')
#    print('    MO #    Energy (a.u.)   Symmetry    Occupation')
#    print('-------------------------------------------------')
#    for i in range(sys['Norb']):
#        print('     {}       {}          {}         {}'.format(i+1,sys['mo_energy'][i],sys['sym'][i],sys['mo_occ'][i]))
#
#    ints = get_integrals(inputs['onebody_file'],inputs['twobody_file'],sys)
#    print('')
#    print('  Nuclear Repulsion Energy = {} HARTREE'.format(ints['Vnuc']))
#    print('  Reference Energy = {} HARTREE'.format(ints['Escf']))
#
#    
#    calc_driver_main(inputs,sys,ints)
#
