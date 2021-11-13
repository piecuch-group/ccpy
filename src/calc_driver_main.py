"""Main calculation driver module of CCpy."""
import os
import argparse
import numpy as np
from system import build_system
from integrals import get_integrals
from parser_module import parse_input_file
from symmetry_count import get_symmetry_count

def calc_driver_main(inputs,sys,ints):
    """Performs the calculation specified by the user in the input.

    Parameters
    ----------
    inputs : dict
        Contains all input keyword flags obtained from parsing the user-supplied input
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals that define the bare Hamiltonian H_N

    Returns
    -------
    None
    """
    calc_type = inputs['calc_type']
    cc_t = None

    if calc_type == 'mp2' or calc_type == 'MP2':
        from mbpt_module import mp2
        Emp2 = mp2(sys,ints)

    if calc_type == 'mp3' or calc_type == 'MP3':
        from mbpt_module import mp3
        Emp3 = mp3(sys,ints)

    if calc_type == 'ip-gf2' or calc_type == 'IP-GF2':
        from mbgf_module import gf2_ip
        ip_omega_gf2 = gf2_ip(inputs['nroot'],ints,sys,maxit=inputs['maxit'],tol=inputs['tol'])

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
                        tol=inputs['eom_tol'],\
                        noact=inputs['eom_guess_noact'],nuact=inputs['eom_guess_nuact'],\
                        maxit=inputs['eom_maxit'])

    if calc_type == 'ipeom2' or calc_type == 'IPEOM2':
        from ccsd_module import ccsd
        from ipeom2_module import ipeom2
        from HBar_module import HBar_CCSD

        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        
        cc_t, omega = ipeom2(inputs['nroot'],H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,\
                        tol=inputs['eom_tol'],\
                        noact=inputs['eom_guess_noact'],nuact=inputs['eom_guess_nuact'],\
                        maxit=inputs['eom_maxit'])

    if calc_type == 'ccsdt' or calc_type == 'CCSDT':
        from ccsdt_module import ccsdt
        cc_t, Eccsdt = ccsdt(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

    if calc_type == 'eomccsdt' or calc_type == 'EOMCCSDT':
        from ccsdt_module import ccsdt
        from HBar_module import HBar_CCSDT
        from eomccsdt_module import eomccsdt, test_updates

        cc_t, Eccsdt = ccsdt(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])

        H1A,H1B,H2A,H2B,H2C = HBar_CCSDT(cc_t,ints,sys)
        
        cc_t, omega = eomccsdt(inputs['nroot'],H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,\
                        tol=inputs['eom_tol'],\
                        noact=inputs['eom_guess_noact'],nuact=inputs['eom_guess_nuact'],\
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
        from crcc23_module import crcc23
        from crcc24_module import crcc24
        cc_t, Eccsd = ccsd(sys,ints,\
                        shift=inputs['ccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,\
                        shift=inputs['lccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'])
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=inputs['isRHF'])
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
                        tol=inputs['eom_tol'],\
                        noact=inputs['eom_guess_noact'],nuact=inputs['eom_guess_nuact'],\
                        maxit=inputs['eom_maxit'])
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,\
                        shift=inputs['lccshift'],tol=inputs['tol'],maxit=inputs['maxit'],\
                        diis_size=inputs['diis_size'],nroot=len(omega),omega=omega,\
                        eom_tol=inputs['eom_tol'],eom_lccshift=inputs['eom_lccshift'],eom_maxit=inputs['eom_maxit'])
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=inputs['isRHF'],\
                        nroot=len(omega),omega=omega)

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

    # Save the cc_t dictionary if it is initialized (e.g, not None)
    save_location = inputs['work_dir'] + '/save_data'
    save_cc_vectors(inputs['save_data'],save_location,cc_t)

    return

def save_cc_vectors(if_save,save_location,cc_t=None):
    if if_save:
        if cc_t is not None:
            print('\n\n   ***** Saving final cc_t dictionary to directory {} *****'.format(save_location))
            if os.path.isdir(save_location): 
                print('   Directory {} already exists'.format(save_location))
            else:
                print('   Making directory {}'.format(save_location))
                os.mkdir(save_location)
            print('   Saving',end=' ')
            for key, value in cc_t.items():
                print(' {} '.format(key),end='')
                np.save(save_location+'/'+key,value,allow_pickle=True)
            print('')

