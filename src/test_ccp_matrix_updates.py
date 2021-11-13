import os
import argparse
import numpy as np
from system import build_system
from integrals import get_integrals
from ccsdt_module import ccsdt, get_ccsd_intermediates
import ccp_matrix
import time
#from numba import njit

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

    work_dir = '/home2/gururang/CCpy/tests/H2O-2Re-DZ'
    #work_dir = '/home2/gururang/CCpy/tests/F2+-1.0-631g'
    nfroz = 0 

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

    # get lists of triples from P space
    list_of_triples = get_list_of_triples(p_spaces)
    num_triples_A = len(list_of_triples['A'])
    num_triples_B = len(list_of_triples['B'])
    num_triples_C = len(list_of_triples['C'])
    num_triples_D = len(list_of_triples['D'])
    list_A = np.asarray(list_of_triples['A'])
    list_B = np.asarray(list_of_triples['B'])
    list_C = np.asarray(list_of_triples['C'])
    list_D = np.asarray(list_of_triples['D'])

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
    D1 = -np.einsum('mk,abcijm->abcijk',H1A['oo'],t3a,optimize=True)
    D2 = np.einsum('ce,abeijk->abcijk',H1A['vv'],t3a,optimize=True)
    D3 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],t3a,optimize=True)
    D4 = 0.5*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],t3a,optimize=True)
    D5 = np.einsum('cmke,abeijm->abcijk',H2A['voov'],t3a,optimize=True)
    D6 = np.einsum('cmke,abeijm->abcijk',H2B['voov'],t3b,optimize=True)
    # A(k/ij)
    D1 += -np.einsum('abcijk->abckji',D1,optimize=True)\
            -np.einsum('abcijk->abcikj',D1,optimize=True)
    D3 += -np.einsum('abcijk->abckji',D3,optimize=True)\
            -np.einsum('abcijk->abcikj',D3,optimize=True)
    # A(c/ab)
    D2 += -np.einsum('abcijk->cbaijk',D2,optimize=True)\
            -np.einsum('abcijk->acbijk',D2,optimize=True)
    D4 += -np.einsum('abcijk->cbaijk',D4,optimize=True)\
            -np.einsum('abcijk->acbijk',D4,optimize=True)
    # A(k/ij)A(c/ab)
    D5 += -np.einsum('abcijk->abckji',D5,optimize=True)\
            -np.einsum('abcijk->abcikj',D5,optimize=True)\
            -np.einsum('abcijk->cbaijk',D5,optimize=True)\
            -np.einsum('abcijk->acbijk',D5,optimize=True)\
            +np.einsum('abcijk->cbakji',D5,optimize=True)\
            +np.einsum('abcijk->cbaikj',D5,optimize=True)\
            +np.einsum('abcijk->acbkji',D5,optimize=True)\
            +np.einsum('abcijk->acbikj',D5,optimize=True)   
    D6 += -np.einsum('abcijk->abckji',D6,optimize=True)\
            -np.einsum('abcijk->abcikj',D6,optimize=True)\
            -np.einsum('abcijk->cbaijk',D6,optimize=True)\
            -np.einsum('abcijk->acbijk',D6,optimize=True)\
            +np.einsum('abcijk->cbakji',D6,optimize=True)\
            +np.einsum('abcijk->cbaikj',D6,optimize=True)\
            +np.einsum('abcijk->acbkji',D6,optimize=True)\
            +np.einsum('abcijk->acbikj',D6,optimize=True)   
    X3A = D1 + D2 + D3 + D4
    #t3a_new_1 = cc_loops.cc_loops.update_t3a(t3a,X3A,fA['oo'],fA['vv'],shift)

   
    t3a_p = np.zeros(num_triples_A)
    X3A_p = np.zeros(num_triples_A)
    for ct,idx in enumerate(list_of_triples['A']):
        t3a_p[ct] = t3a[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]]
        X3A_p[ct] = X3A[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]]
    t3b_p = np.zeros(num_triples_B)
    X3B_p = np.zeros(num_triples_B)
    for ct,idx in enumerate(list_of_triples['B']):
        t3b_p[ct] = t3b[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]]
    #    X3B_p[ct] = X3B[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]] 
    t3c_p = np.zeros(num_triples_C)
    X3C_p = np.zeros(num_triples_C)
    for ct,idx in enumerate(list_of_triples['C']):
        t3c_p[ct] = t3c[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]]
    #    X3C_p[ct] = X3C[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]] 
    t3d_p = np.zeros(num_triples_D)
    X3D_p = np.zeros(num_triples_D)
    for ct,idx in enumerate(list_of_triples['D']):
        t3d_p[ct] = t3d[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]]
    #    X3D_p[ct] = X3D[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5]] 

    # build temporary 3-body HBars
    h3A_vooovo = np.einsum('mnef,aeij->anmifj',vA['oovv'],t2a,optimize=True)
    h3A_vovovv = -np.einsum('mnef,abim->anbife',vA['oovv'],t2a,optimize=True)

    # test t3 updates
    #print('FORTRAN UPDATE t3a...')
    #t1 = time.perf_counter()
    #HT3A,_,_,_ = ccp_matrix.ccp_matrix.build_ht3(t3a_p,t3b_p,t3c_p,t3d_p,\
    #                list_A,list_B,list_C,list_D,\
    #                fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
    #                vA['oovv'],vB['oovv'],vC['oovv'],t2a,t2b,t2c,\
    #                H1A['oo'],H1A['vv'],H1A['ov'],\
    #                H1B['oo'],H1B['vv'],H1B['ov'],\
    #                H2A['oooo'],H2A['vvvv'],H2A['voov'],H2A['vooo'],H2A['vvov'],H2A['ooov'],H2A['vovv'],\
    #                H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovvo'],H2B['vovo'],H2B['ovov'],H2B['vooo'],\
    #                H2B['ovoo'],H2B['vvov'],H2B['vvvo'],H2B['ooov'],H2B['oovo'],H2B['vovv'],H2B['ovvv'],\
    #                H2C['oooo'],H2C['vvvv'],H2C['voov'],H2C['vooo'],H2C['vvov'],H2C['ooov'],H2C['vovv'],\
    #                h3A_vooovo,h3A_vovovv)
    #t2 = time.perf_counter()
    #print('took {} s'.format(t2-t1))

    t1 = time.perf_counter()
    HT3A = build_HT3A(t3a_p,t3b_p,t3c_p,t3d_p,list_of_triples['A'],H1A,H1B,H2A,H2B,H2C)
    t2 = time.perf_counter()
    print('took {} s'.format(t2-t1))

    # Calculate error
    error = 0.0
    for ct,idx in enumerate(list_of_triples['A']):
        #print(HT3A[ct])
        error += HT3A[ct] - X3A_p[ct]
    print('Error in t3a = {}'.format(error))

    return

def build_HT3A(t3a_p,t3b_p,t3c_p,t3d_p,list_of_triples_A,H1A,H1B,H2A,H2B,H2C):

    HT3A = np.zeros(len(list_of_triples_A))
    for idet,idx1 in enumerate(list_of_triples_A):
        a1 = idx1[0]; b1 = idx1[1]; c1 = idx1[2];
        i1 = idx1[3]; j1 = idx1[4]; k1 = idx1[5];
        for jdet,idx2 in enumerate(list_of_triples_A):
            a2 = idx2[0]; b2 = idx2[1]; c2 = idx2[2];
            i2 = idx2[3]; j2 = idx2[4]; k2 = idx2[5];
            # diagram 1 -A(i/jk)h(mi)t3a(abcmjk)
            # -> A(abc)A(
            dgm1 = -(c1==c2)*(b1==b2)*(a1==a2)*(\
                    (j1==j2)*(k1==k2)*H1A['oo'][i2,i1]\
                    -(i1==j2)*(k1==k2)*H1A['oo'][i2,j1]\
                    -(j1==j2)*(i1==k2)*H1A['oo'][i2,k1]\
                    -(j1==i2)*(k1==k2)*H1A['oo'][j2,i1]\
                    -(j1==j2)*(k1==i2)*H1A['oo'][k2,i1]\
                    +(i1==i2)*(k1==k2)*H1A['oo'][j2,j1]\
                    +(i1==j2)*(k1==i2)*H1A['oo'][k2,j1]\
                    +(j1==i2)*(i1==k2)*H1A['oo'][j2,k1]\
                    +(j1==j2)*(i1==i2)*H1A['oo'][k2,k1]\
                    -(k1==j2)*(j1==k2)*H1A['oo'][i2,i1]\
                    +(i1==j2)*(j1==k2)*H1A['oo'][i2,k1]\
                    +(k1==j2)*(i1==k2)*H1A['oo'][i2,j1]\
                    +(k1==i2)*(j1==k2)*H1A['oo'][j2,i1]\
                    +(k1==j2)*(j1==i2)*H1A['oo'][k2,i1]\
                    -(i1==i2)*(j1==k2)*H1A['oo'][j2,k1]\
                    -(i1==j2)*(j1==i2)*H1A['oo'][k2,k1]\
                    -(k1==i2)*(i1==k2)*H1A['oo'][j2,j1]\
                    -(k1==j2)*(i1==i2)*H1A['oo'][k2,j1])
            # diagram 2 A(a/bc)h(ae)t3a(ebcijk)
            dgm2 = (k1==k2)*(j1==j2)*(i1==i2)*(\
                    (b1==b2)*(c1==c2)*H1A['vv'][a1,a2]\
                    -(a1==b2)*(c1==c2)*H1A['vv'][b1,a2]\
                    -(b1==a2)*(c1==c2)*H1A['vv'][a1,b2]\
                    -(b1==b2)*(a1==c2)*H1A['vv'][c1,a2]\
                    -(b1==b2)*(c1==a2)*H1A['vv'][a1,c2]\
                    +(a1==a2)*(c1==c2)*H1A['vv'][b1,b2]\
                    +(a1==b2)*(c1==a2)*H1A['vv'][b1,c2]\
                    +(b1==a2)*(a1==c2)*H1A['vv'][c1,b2]\
                    +(b1==b2)*(a1==a2)*H1A['vv'][c1,c2]\
                    -(c1==b2)*(b1==c2)*H1A['vv'][a1,a2]\
                    +(a1==b2)*(b1==c2)*H1A['vv'][c1,a2]\
                    +(c1==a2)*(b1==c2)*H1A['vv'][a1,b2]\
                    +(c1==b2)*(a1==c2)*H1A['vv'][b1,a2]\
                    +(c1==b2)*(b1==a2)*H1A['vv'][a1,c2]\
                    -(a1==a2)*(b1==c2)*H1A['vv'][c1,b2]\
                    -(a1==b2)*(b1==a2)*H1A['vv'][c1,c2]\
                    -(c1==a2)*(a1==c2)*H1A['vv'][b1,b2]\
                    -(c1==b2)*(a1==a2)*H1A['vv'][b1,c2])
            # diagram 3 0.5*A(k/ij)h(mnij)t3a(abcmnk)
            dgm3 = (a1==a2)*(b1==b2)*(c1==c2)*(\
                    (k1==k2)*H2A['oooo'][i2,j2,i1,j1]\
                    -(i1==k2)*H2A['oooo'][i2,j2,k1,j1]\
                    -(j1==k2)*H2A['oooo'][i2,j2,i1,k1]\
                    -(k1==i2)*H2A['oooo'][k2,j2,i1,j1]\
                    -(k1==j2)*H2A['oooo'][i2,k2,i1,j1]\
                    +(i1==i2)*H2A['oooo'][k2,j2,k1,j1]\
                    +(i1==j2)*H2A['oooo'][i2,k2,k1,j1]\
                    +(j1==i2)*H2A['oooo'][k2,j2,i1,k1]\
                    +(j1==j2)*H2A['oooo'][i2,k2,i1,k1])
            # diagram 4 0.5*A(c/ab)h(abef)t3a(ebcijk)
            dgm4 = (i1==i2)*(j1==j2)*(k1==k2)*(\
                    (c1==c2)*H2A['vvvv'][a1,b1,a2,b2]\
                    -(a1==c2)*H2A['vvvv'][c1,b1,a2,b2]\
                    -(c1==a2)*H2A['vvvv'][a1,b1,c2,b2]\
                    -(b1==c2)*H2A['vvvv'][a1,c1,a2,b2]\
                    -(c1==b2)*H2A['vvvv'][a1,b1,a2,c2]\
                    +(a1==a2)*H2A['vvvv'][c1,b1,c2,b2]\
                    +(a1==b2)*H2A['vvvv'][c1,b1,a2,c2]\
                    +(b1==a2)*H2A['vvvv'][a1,c1,c2,b2]\
                    +(b1==b2)*H2A['vvvv'][a1,c1,a2,c2])
            # diagram 5 A(i/jk)A(c/ab)h(amie)t3a(ebcmjk)
            # -> A(jk)A(bc)A(a/bc)A(i/jk)A(a'\b'c')A(i'\j'k') x 
            #               d(kk')d(jj')d(bb')d(cc')h(a,i',i,a')
            dgm5 = (k1==k2)*(j1==j2)*(b1==b2)*(c1==c2)*H2A['voov'][a1,i2,i1,a2]\

                   -(k1==k2)*(j1==j2)*(b1==a2)*(c1==c2)*H2A['voov'][a1,i2,i1,b2]\
                   -(k1==k2)*(j1==j2)*(b1==b2)*(c1==a2)*H2A['voov'][a1,i2,i1,c2]\

                   -(k1==k2)*(j1==i2)*(b1==b2)*(c1==c2)*H2A['voov'][a1,j2,i1,a2]\
                   -(k1==i2)*(j1==j2)*(b1==b2)*(c1==c2)*H2A['voov'][a1,k2,i1,a2]\

            HT3A[idet] = HT3A[idet] + (dgm1+dgm2+dgm3+dgm4)*t3a_p[jdet]

    return HT3A

def get_list_of_triples(p_spaces):

    noa = p_spaces['A'].shape[3]
    nua = p_spaces['A'].shape[0]
    nob = p_spaces['D'].shape[3]
    nub = p_spaces['D'].shape[0]

    list_of_triples_A = []
    for a in range(nua):
        for b in range(a+1,nua):
            for c in range(b+1,nua):
                for i in range(noa):
                    for j in range(i+1,noa):
                        for k in range(j+1,noa):
                            if p_spaces['A'][a,b,c,i,j,k] == 1:
                                list_of_triples_A.append([a, b, c, i, j, k])

    list_of_triples_B = []
    for a in range(nua):
        for b in range(a+1,nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i+1,noa):
                        for k in range(nob):
                            if p_spaces['B'][a,b,c,i,j,k] == 1:
                                list_of_triples_B.append([a, b, c, i, j, k])
                            
    list_of_triples_C = []
    for a in range(nua):
        for b in range(nub):
            for c in range(b+1,nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j+1,nob):
                            if p_spaces['C'][a,b,c,i,j,k] == 1:
                                list_of_triples_C.append([a, b, c, i, j, k])

    list_of_triples_D = []
    for a in range(nub):
        for b in range(a+1,nub):
            for c in range(b+1,nub):
                for i in range(nob):
                    for j in range(i+1,nob):
                        for k in range(j+1,nob):
                            if p_spaces['D'][a,b,c,i,j,k] == 1:
                                list_of_triples_D.append([a, b, c, i, j, k])

    list_of_triples = {'A' : list_of_triples_A,\
                        'B' : list_of_triples_B,\
                        'C' : list_of_triples_C,\
                        'D' : list_of_triples_D}
    return list_of_triples


if __name__ == '__main__':
    main()
