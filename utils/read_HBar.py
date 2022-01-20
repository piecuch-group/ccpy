import numpy as np
from scipy.io import FortranFile
import argparse 

np.set_printoptions(precision=6,threshold=np.inf,suppress=True,linewidth=200)

def read_HBar(fid,noa,nob,nua,nub):

    thresh = 0.0

    norb_a = nua + noa
    norb_b = nub + nob

    oa = slice(0,noa)
    ob = slice(0,nob)
    ua = slice(noa,norb_a)
    ub = slice(nob,norb_b)
    
    h1a = np.zeros((norb_a,norb_a))
    h1b = np.zeros((norb_b,norb_b))
    h2a = np.zeros((norb_a,norb_a,norb_a,norb_a))
    h2b = np.zeros((norb_a,norb_b,norb_a,norb_b))
    h2c = np.zeros((norb_b,norb_b,norb_b,norb_b))

    with FortranFile(fid,'r') as f_in:

        ndata=0
        h_read = f_in.read_reals(dtype=np.float64)
        for p in range(norb_a):
            for q in range(norb_a):
                h1a[p,q]=h_read[ndata]
                ndata+=1

        ndata=0
        h_read = f_in.read_reals(dtype=np.float64)
        for p in range(norb_b):
            for q in range(norb_b):
                h1b[p,q]=h_read[ndata]
                ndata+=1
    
        ndata=0
        h_read = f_in.read_reals(dtype=np.float64)
        for p in range(norb_a):
            for q in range(norb_a):
                for r in range(norb_a):
                    for s in range(norb_a):
                        h2a[p,q,r,s]=h_read[ndata]
                        ndata+=1

        ndata=0
        h_read = f_in.read_reals(dtype=np.float64)
        for p in range(norb_a):
            for q in range(norb_b):
                for r in range(norb_a):
                    for s in range(norb_b):
                        h2b[p,q,r,s]=h_read[ndata]
                        ndata+=1

        ndata=0
        h_read = f_in.read_reals(dtype=np.float64)
        for p in range(norb_b):
            for q in range(norb_b):
                for r in range(norb_b):
                    for s in range(norb_b):
                        h2c[p,q,r,s]=h_read[ndata]
                        ndata+=1

    H1A = {'oo' : h1a[oa,oa], 'ov' : h1a[oa,ua], 'vo' : h1a[ua,oa], 'vv' : h1a[ua,ua]}
    H1B = {'oo' : h1b[ob,ob], 'ov' : h1b[ob,ub], 'vo' : h1b[ub,ob], 'vv' : h1b[ub,ub]}
    H2A = {'oooo' : h2a[oa,oa,oa,oa], 'ooov' : h2a[oa,oa,oa,ua], 'oovv' : h2a[oa,oa,ua,ua],\
            'voov' : h2a[ua,oa,oa,ua], 'vovv' : h2a[ua,oa,ua,ua], 'vooo' : h2a[ua,oa,oa,oa],\
            'vvov' : h2a[ua,ua,oa,ua], 'vvvv' : h2a[ua,ua,ua,ua]}
    H2B = {'oooo' : h2b[oa,ob,oa,ob], 'ooov' : h2b[oa,ob,oa,ub], 'oovv' : h2b[oa,ob,ua,ub],\
            'oovo' : h2b[oa,ob,ua,ob], 'voov' : h2b[ua,ob,oa,ub], 'vovo' : h2b[ua,ob,ua,ob],\
            'ovov' : h2b[oa,ub,oa,ub], 'ovvo' : h2b[oa,ub,ua,ob],\
            'vovv' : h2b[ua,ob,ua,ub], 'vooo' : h2b[ua,ob,oa,ob], 'ovoo' : h2b[oa,ub,oa,ob],\
            'ovvv' : h2b[oa,ub,ua,ub], 'vvvo' : h2b[ua,ub,ua,ob],\
            'vvov' : h2b[ua,ub,oa,ub], 'vvvv' : h2b[ua,ub,ua,ub]}
    H2C = {'oooo' : h2c[ob,ob,ob,ob], 'ooov' : h2c[ob,ob,ob,ub], 'oovv' : h2c[ob,ob,ub,ub],\
            'voov' : h2c[ub,ob,ob,ub], 'vovv' : h2c[ub,ob,ub,ub], 'vooo' : h2c[ub,ob,ob,ob],\
            'vvov' : h2c[ub,ub,ob,ub], 'vvvv' : h2c[ub,ub,ub,ub]}

    return H1A,H1B,H2A,H2B,H2C

def savehbar(H1A,H1B,H2A,H2B,H2C,style='python'):
    from scipy.io import savemat

    vec_dict= {'H1A' : H1A, 'H1B' : H1B, 'H2A' : H2A, 'H2B' : H2B, 'H2C' : H2C}
    if style=='matlab':
        savemat('/Users/karthik/Dropbox/Hartree Fock/hartree_fock/v4/CC_matlab/junreadsave-HBar'+'.mat',vec_dict)
    else:
        np.save('HBar-jun.npy',vec_dict)

    return

def main(args):
            
    noa = args.noa
    nob = args.nob
    nua = args.nua
    nub = args.nub
    #frz = args.frz
    fid = args.f_read

    H1A,H1B,H2A,H2B,H2C = read_HBar(fid,noa,nob,nua,nub)

    savehbar(H1A,H1B,H2A,H2B,H2C,style='python')

    

    #tol = 1.0e-07
    #print(H1A['oo'])
    #for i in range(noa):
    #    for j in range(noa):
    #        if abs(H1A['oo'][i,j]) > tol:
    #            print('h1A_oo({},{}) = {}'.format(i+1,j+1,H1A['oo'][i,j]))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script that reads and writes L and R vectors contained in L-CC** and R-CC* files from Jun's code")
    parser.add_argument('noa',type=int,help='Number of correlated occupied alpha spatial orbitals')
    parser.add_argument('nob',type=int,help='Number of correlated occupied beta spatial orbitals')
    parser.add_argument('nua',type=int,help='Number of correlated unoccupied alpha spatial orbitals')
    parser.add_argument('nub',type=int,help='Number of correlated unoccuiped beta spatial orbitals')
    #parser.add_argument('frz',type=int,help='Number of frozen spatial orbitals')
    parser.add_argument('f_read',type=str,help='Path to L or R vector file to be read')
    args = parser.parse_args()
    main(args)
