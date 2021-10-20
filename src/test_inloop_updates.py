import os
import argparse
import numpy as np
from system import build_system
from integrals import get_integrals
from adaptive_ccp_module_v2 import get_list_of_triples, update_t3a_inloop, update_t3b_inloop, update_t3c_inloop, update_t3d_inloop
from adaptive_ccp_module_v3 import ccsdt_p
#from adaptive_ccp_module import ccsdt_p
from ccsdt_module import ccsdt, get_ccsd_intermediates
import ccp_loops
import time

def get_file_names_with_strings(path,identifier):
    full_list = os.listdir(path)
    final_list = [str(x) for x in full_list if identifier in x]
    if len(final_list) > 0:
        return final_list[0]
    else: return ''

def main():

    flag_test_full = True

    #work_dir = os.path.abspath(args.directory)+'/'
    #nfroz = args.frozen

    #work_dir = '/home2/gururang/CCpy/tests/H2O-2Re-DZ/'
    work_dir = '/home2/gururang/CCpy/tests/F2+-1.0-631g'
    nfroz = 2

    gamess_file = work_dir + '/' + get_file_names_with_strings(work_dir,'log')
    onebody_file = work_dir + '/' + get_file_names_with_strings(work_dir,'onebody')
    twobody_file = work_dir + '/' + get_file_names_with_strings(work_dir,'twobody')

    sys = build_system(gamess_file,nfroz)
    print('System Information:')
    print('-------------------------------------------------')
    print('  Number of correlated electrons = {}'.format(sys['Nelec']))
    print('  Number of frozen electrons = {}'.format(2*nfroz))
    print('  Charge = {}'.format(sys['charge']))
    print('  Point group = {}'.format(sys['point_group']))
    print('  Spin multiplicity of reference = {}'.format(sys['multiplicity']))
    print('')
    print('    MO #    Energy (Eh)   Symmetry    Occupation')
    print('-------------------------------------------------')
    for i in range(sys['Norb']):
        print('     {}       {:>6f}          {}         {}'.format(i+1,sys['mo_energy'][i],sys['sym'][i],sys['mo_occ'][i]))

    ints = get_integrals(onebody_file,twobody_file,sys)
    print('')
    print('  Nuclear Repulsion Energy = {} Eh'.format(ints['Vnuc']))
    print('  Reference Energy = {} Eh'.format(ints['Escf']))

    # get full triples P spaces
    p_spaces = {}
    p_spaces['A'] = np.ones((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a'])) 
    p_spaces['B'] = np.ones((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b'])) 
    p_spaces['C'] = np.ones((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b'])) 
    p_spaces['D'] = np.ones((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    # test the full CCSDT routine using in loop updates
    if flag_test_full:
        cc_t, Eccsdt = ccsdt_p(sys,ints,p_spaces)
    else:
        shift = 0.0
        cc_t, Eccsdt = ccsdt(sys,ints,shift=0.0,tol=1.0e-09)
        t2a = cc_t['t2a']
        t2b = cc_t['t2b']
        t2c = cc_t['t2c']
        t3a = cc_t['t3a']
        t3b = cc_t['t3b']
        t3c = cc_t['t3c']
        t3d = cc_t['t3d']
        vA = ints['vA']
        vB = ints['vB']
        vC = ints['vC']
        fA = ints['fA']
        fB = ints['fB']
   
        # get CCSD intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)

        # vectorized updates
        #cc_t = update_t3a(cc_t,ints,p_spaces['A'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        #cc_t = update_t3b(cc_t,ints,p_spaces['B'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        #t3a_vec = cc_t['t3a']
        #t3b_vec = cc_t['t3b']

        # get lists of triples from P space
        list_of_triples = get_list_of_triples(p_spaces)
        num_triples_A = len(list_of_triples['A'])
        num_triples_B = len(list_of_triples['B'])
        num_triples_C = len(list_of_triples['C'])
        num_triples_D = len(list_of_triples['D'])

        # test t3 updates
        list_A = np.asarray(list_of_triples['A'])
        print('FORTRAN UPDATE t3a...')
        t1 = time.perf_counter()
        I2A_vvov = H2A['vvov'] + np.einsum('me,abim->abie',H1A['ov'],t2a,optimize=True)
        t3a_devec2 = f90_ccp_updates_mkl_omp.ccp_loops.update_t3a(t2a,t3a,t3b,list_A,\
                        vA['oovv'],vB['oovv'],H1A['oo'],H1A['vv'],H2A['oooo'],\
                        H2A['vvvv'],H2A['voov'],H2B['voov'],H2A['vooo'],I2A_vvov,\
                        fA['oo'],fA['vv'],shift,sys['Nocc_a'],sys['Nunocc_a'],\
                        sys['Nocc_b'],sys['Nunocc_b'],num_triples_A)
        t2 = time.perf_counter()
        print('took {} s'.format(t2-t1))

        list_B = np.asarray(list_of_triples['B'])
        print('FORTRAN UPDATE t3b...')
        t1 = time.perf_counter()
        I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True)
        I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True)
        I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
        t3b_devec2 = f90_ccp_updates_mkl_omp.ccp_loops.update_t3b(t2a,t2b,t3a,t3b,t3c,\
                        list_B,vA['oovv'],vB['oovv'],vC['oovv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['oooo'],H2A['vvvv'],\
                        H2A['voov'],H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovov'],\
                        H2B['vovo'],H2B['ovvo'],H2C['voov'],I2A_vooo,H2A['vvov'],\
                        I2B_vooo,I2B_ovoo,H2B['vvov'],H2B['vvvo'],\
                        fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift,\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'],\
                        num_triples_B)
        t2 = time.perf_counter()
        print('took {} s'.format(t2-t1))

        list_C = np.asarray(list_of_triples['C'])
        print('FORTRAN UPDATE t3c...')
        t1 = time.perf_counter()
        I2B_ovoo = H2B['ovoo'] - np.einsum('me,ebij->mbij',H1A['ov'],t2b,optimize=True)
        I2B_vooo = H2B['vooo'] - np.einsum('me,aeij->amij',H1B['ov'],t2b,optimize=True)
        I2C_vooo = H2C['vooo'] - np.einsum('me,cekj->cmkj',H1B['ov'],t2c,optimize=True)
        t3c_devec2 = f90_ccp_updates_mkl_omp.ccp_loops.update_t3c(t2b,t2c,t3b,t3c,t3d,\
                        list_C,vA['oovv'],vB['oovv'],vC['oovv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],\
                        H2A['voov'],\
                        H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovov'],\
                        H2B['vovo'],H2B['ovvo'],H2C['oooo'],H2C['vvvv'],\
                        H2C['voov'],I2C_vooo,H2C['vvov'],\
                        I2B_vooo,I2B_ovoo,H2B['vvov'],H2B['vvvo'],\
                        fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift,\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'],\
                        num_triples_C)
        t2 = time.perf_counter()
        print('took {} s'.format(t2-t1))

        list_D = np.asarray(list_of_triples['D'])
        print('FORTRAN UPDATE t3d...')
        t1 = time.perf_counter()
        I2C_vvov = H2C['vvov'] + np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)
        t3d_devec2 = f90_ccp_updates_mkl_omp.ccp_loops.update_t3d(t2c,t3c,t3d,list_D,\
                        vB['oovv'],vC['oovv'],H1B['oo'],H1B['vv'],H2C['oooo'],\
                        H2C['vvvv'],H2C['voov'],H2B['ovvo'],H2C['vooo'],I2C_vvov,\
                        fB['oo'],fB['vv'],shift,sys['Nocc_a'],sys['Nunocc_a'],\
                        sys['Nocc_b'],sys['Nunocc_b'],num_triples_D)
        t2 = time.perf_counter()
        print('took {} s'.format(t2-t1))



        print('PYTHON UPDATE t3a...')
        cc_t['t3a'] = t3a; cc_t['t3b'] = t3b; cc_t['t3c'] = t3c; cc_t['t3d'] = t3d;
        cc_t = update_t3a_inloop(cc_t,ints,list_of_triples['A'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        t3a_devec = cc_t['t3a']

        print('PYTHON UPDATE t3b...')
        cc_t['t3a'] = t3a; cc_t['t3b'] = t3b; cc_t['t3c'] = t3c; cc_t['t3d'] = t3d;
        cc_t = update_t3b_inloop(cc_t,ints,list_of_triples['B'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        t3b_devec = cc_t['t3b']

        print('PYTHON UPDATE t3c...')
        cc_t['t3a'] = t3a; cc_t['t3b'] = t3b; cc_t['t3c'] = t3c; cc_t['t3d'] = t3d;
        cc_t = update_t3c_inloop(cc_t,ints,list_of_triples['C'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        t3c_devec = cc_t['t3c']

        print('PYTHON UPDATE t3d...')
        cc_t['t3a'] = t3a; cc_t['t3b'] = t3b; cc_t['t3c'] = t3c; cc_t['t3d'] = t3d;
        cc_t = update_t3d_inloop(cc_t,ints,list_of_triples['D'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        t3d_devec = cc_t['t3d']

        error = 0.0
        for a in range(sys['Nunocc_a']):
            for b in range(sys['Nunocc_a']):
                for c in range(sys['Nunocc_a']):
                    for i in range(sys['Nocc_a']):
                        for j in range(sys['Nocc_a']):
                            for k in range(sys['Nocc_a']):
                                error += abs(t3a_devec2[a,b,c,i,j,k] - t3a_devec[a,b,c,i,j,k])
        print('Error in t3a = {}'.format(error))

        error = 0.0
        for a in range(sys['Nunocc_a']):
            for b in range(sys['Nunocc_a']):
                for c in range(sys['Nunocc_b']):
                    for i in range(sys['Nocc_a']):
                        for j in range(sys['Nocc_a']):
                            for k in range(sys['Nocc_b']):
                                error += abs(t3b_devec2[a,b,c,i,j,k] - t3b_devec[a,b,c,i,j,k])
        print('Error in t3b = {}'.format(error))

        error = 0.0
        for a in range(sys['Nunocc_a']):
            for b in range(sys['Nunocc_b']):
                for c in range(sys['Nunocc_b']):
                    for i in range(sys['Nocc_a']):
                        for j in range(sys['Nocc_b']):
                            for k in range(sys['Nocc_b']):
                                error += abs(t3c_devec2[a,b,c,i,j,k] - t3c_devec[a,b,c,i,j,k])
        print('Error in t3c = {}'.format(error))

        error = 0.0
        for a in range(sys['Nunocc_b']):
            for b in range(sys['Nunocc_b']):
                for c in range(sys['Nunocc_b']):
                    for i in range(sys['Nocc_b']):
                        for j in range(sys['Nocc_b']):
                            for k in range(sys['Nocc_b']):
                                error += abs(t3d_devec2[a,b,c,i,j,k] - t3d_devec[a,b,c,i,j,k])
        print('Error in t3d = {}'.format(error))

    return


def get_t_from_mbpt(ints,sys):

    # obtain t1 from 2nd-order MBPT estimate
    t1a = np.zeros((sys['Nunocc_a'],sys['Nocc_a']))
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            val = 0.0
            denom0 = fA['oo'][i,i]-fA['vv'][a,a]
            for e in range(sys['Nunocc_a']):
                for f in range(sys['Nunocc_a']):
                    for n in range(sys['Nocc_a']):
                        denom = fA['oo'][i,i]+fA['oo'][n,n]-fA['vv'][e,e]-fA['vv'][f,f]
                        val += 0.5*vA['vovv'][a,n,e,f]*vA['oovv'][e,f,i,n]/(denom*denom0)
                for f in range(sys['Nunocc_b']):
                    for n in range(sys['Nocc_b']):
                        denom = fA['oo'][i,i]+fB['oo'][n,n]-fA['vv'][e,e]-fB['vv'][f,f]
                        val += vB['vovv'][a,n,e,f]*vB['oovv'][e,f,i,n]/(denom*denom0)
            for m in range(sys['Nocc_a']):
                for f in range(sys['Nunocc_a']):
                    for n in range(sys['Nocc_a']):
                        denom = fA['oo'][m,m]+fA['oo'][n,n]-fA['vv'][a,a]-fA['vv'][f,f]
                        val -= 0.5*vA['ooov'][m,n,i,f]*vA['oovv'][a,f,m,n]/(denom*denom0)
                for f in range(sys['Nunocc_b']):
                    for n in range(sys['Nocc_b']):
                        denom = fA['oo'][m,m]+fB['oo'][n,n]-fA['vv'][a,a]-fB['vv'][f,f]
                        val -= vB['ooov'][m,n,i,f]*vB['oovv'][a,f,m,n]/(denom*denom0)
            t1a[a,i] = val
        
    # obtain t2 from 1st-order MBPT estimate
    t2a = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for i in range(sys['Nocc_a']):
                for j in range(i+1,sys['Nocc_a']):
                    denom = fA['oo'][i,i]+fA['oo'][j,j]-fA['vv'][a,a]-fA['vv'][b,b]
                    t2a[a,b,i,j] = vA['oovv']/denom
                    t2a[b,a,i,j] = -t2a[a,b,i,j]
                    t2a[a,b,j,i] = -t2a[a,b,i,j]
                    t2a[b,a,j,i] = t2a[a,b,i,j]

    t2b = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
    for a in range(sys['Nunocc_a']):
        for b in range(sys['Nunocc_b']):
            for i in range(sys['Nocc_a']):
                for j in range(sys['Nocc_b']):
                    denom = fA['oo'][i,i]+fB['oo'][j,j]-fA['vv'][a,a]-fB['vv'][b,b]
                    t2b[a,b,i,j] = vB['oovv']/denom

    # obtain t3 from 2nd-order MBPT estimate
    t3a = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    t3b = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']))
    t3c = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']))
    t3d = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))



    cc_t['t1a'] = t1a
    cc_t['t1b'] = t1a
    cc_t['t2a'] = t2a
    cc_t['t2b'] = t2b
    cc_t['t2c'] = t2a
    cc_t['t3a'] = t3a
    cc_t['t3b'] = t3b
    cc_t['t3c'] = t3c
    cc_t['t3d'] = t3d
    return cc_t

if __name__ == '__main__':
    main()
    #parser = argparse.ArgumentParser(description='parser for testing devectorized t3 updates')
    #parser.add_argument('directory',type=str,help='Path to working directory containing GAMESS log file and integrals')
    #parser.add_argument('-f','--frozen',type=int,help='Number of frozen spatial orbitals',default=0)

    #args = parser.parse_args()
    #main(args)
