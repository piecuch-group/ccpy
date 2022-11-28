import numpy as np
from scipy.io import FortranFile
import argparse

np.set_printoptions(precision=6,threshold=np.inf,suppress=True,linewidth=200)

def read_t_amp(tccsdt,noa,nob,nua,nub,exc):

    thresh = 0.0
    
    t1a = np.zeros((nua,noa))
    t1b = np.zeros((nub,nob))
    t2a = np.zeros((nua,nua,noa,noa))
    t2b = np.zeros((nua,nub,noa,nob))
    t2c = np.zeros((nub,nub,nob,nob))
    if exc > 2:
        t3a = np.zeros((noa,noa,noa,nua,nua,nua))
        t3b = np.zeros((noa,noa,nob,nua,nua,nub))
        t3c = np.zeros((noa,nob,nob,nua,nub,nub))
        t3d = np.zeros((nob,nob,nob,nub,nub,nub))


    ndata=0
    
    for i in range(noa):
        for a in range(nua):
            if abs(tccsdt[ndata])>=thresh:
                t1a[a,i]=tccsdt[ndata]
            ndata+=1
    
    for i in range(nob):
        for a in range(nub):
            if abs(tccsdt[ndata])>=thresh:
                t1b[a,i]=tccsdt[ndata]
            ndata+=1
            
    for i in range(noa):
        for j in range(noa):
            for a in range(nua):
                for b in range(nua):
                    if abs(tccsdt[ndata])>=thresh:
                        t2a[a,b,i,j]=tccsdt[ndata]
                    ndata+=1
    
    for i in range(nob):
        for j in range(nob):
            for a in range(nub):
                for b in range(nub):
                    if abs(tccsdt[ndata])>=thresh:
                        t2c[a,b,i,j]=tccsdt[ndata]
                    ndata+=1
    
    for i in range(noa):
        for j in range(nob):
            for a in range(nua):
                for b in range(nub):
                    if abs(tccsdt[ndata])>=thresh:
                        t2b[a,b,i,j]=tccsdt[ndata]
                    ndata+=1
    if exc > 2:
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for a in range(nua):
                        for b in range(nua):
                            for c in range(nua):
                                if abs(tccsdt[ndata])>=thresh:
                                    t3a[i,j,k,a,b,c]=tccsdt[ndata]
                                ndata+=1

        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nub):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(tccsdt[ndata])>=thresh:
                                    t3d[i,j,k,a,b,c]=tccsdt[ndata]
                                ndata+=1

        for i in range(noa):
            for j in range(nob):
                for k in range(noa):
                    for a in range(nua):
                        for b in range(nub):
                            for c in range(nua):
                                if abs(tccsdt[ndata])>=thresh:
                                    t3b[i,k,j,a,c,b]=tccsdt[ndata]
                                ndata+=1

        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nua):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(tccsdt[ndata])>=thresh:
                                    t3c[i,j,k,a,b,c]=tccsdt[ndata]
                                ndata+=1
   
    # Jun flips signs
    #t1a = -t1a.transpose(1,0)
    #t1b = -t1b.transpose(1,0)
    #t2a = -t2a.transpose(2,3,0,1)
    #t2b = -t2b.transpose(2,3,0,1)
    #t2c = -t2c.transpose(2,3,0,1)
    if exc == 2:
        return {'t1a':t1a, 't1b':t1b, 't2a':-t2a, 't2b':-t2b, 't2c':-t2c}
    else:
        t3a = -t3a.transpose(3,4,5,0,1,2)
        t3b = -t3b.transpose(3,4,5,0,1,2)
        t3c = -t3c.transpose(3,4,5,0,1,2)
        t3d = -t3d.transpose(3,4,5,0,1,2)
        return {'t1a':t1a, 't1b':t1b, 't2a':t2a, 't2b':t2b, 't2c':t2c, 't3a':t3a, 't3b':t3b, 't3c':t3c, 't3d':t3d}

def read_l_amp(lccsdt,iroot,noa,nob,nua,nub,exc):
    
    thresh = 0.0
    
    l1a = np.zeros((noa,nua))
    l1b = np.zeros((nob,nub))
    l2a = np.zeros((noa,noa,nua,nua))
    l2b = np.zeros((noa,nob,nua,nub))
    l2c = np.zeros((nob,nob,nub,nub))
    if exc > 2:
        l3a = np.zeros((noa,noa,noa,nua,nua,nua))
        l3b = np.zeros((noa,noa,nob,nua,nua,nub))
        l3c = np.zeros((noa,nob,nob,nua,nub,nub))
        l3d = np.zeros((nob,nob,nob,nub,nub,nub))

    reclen=noa*nua
    reclen+=nob*nub
    reclen+=noa*noa*nua*nua
    reclen+=noa*nob*nua*nub
    reclen+=nob*nob*nub*nub
    if exc > 2:
        reclen+=noa*noa*noa*nua*nua*nua
        reclen+=noa*noa*nob*nua*nua*nub
        reclen+=noa*nob*nob*nua*nub*nub
        reclen+=nob*nob*nob*nub*nub*nub
      
    ndata=0
    
    
    for i in range(noa):
        for a in range(nua):
            if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                l1a[i,a]=lccsdt[ndata+iroot*reclen]
            ndata+=1
    
    for i in range(nob):
        for a in range(nub):
            if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                l1b[i,a]=lccsdt[ndata+iroot*reclen]
            ndata+=1
            
    for i in range(noa):
        for j in range(noa):
            for a in range(nua):
                for b in range(nua):
                    if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                        l2a[i,j,a,b]=lccsdt[ndata+iroot*reclen]
                    ndata+=1
    
    for i in range(noa):
        for j in range(nob):
            for a in range(nua):
                for b in range(nub):
                    if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                        l2b[i,j,a,b]=lccsdt[ndata+iroot*reclen]
                    ndata+=1
    
    for i in range(nob):
        for j in range(nob):
            for a in range(nub):
                for b in range(nub):
                    if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                        l2c[i,j,a,b]=lccsdt[ndata+iroot*reclen]
                    ndata+=1
    if exc > 2:
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for a in range(nua):
                        for b in range(nua):
                            for c in range(nua):
                                if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                                    l3a[i,j,k,a,b,c]=lccsdt[ndata+iroot*reclen]
                                ndata+=1

        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for a in range(nua):
                        for b in range(nua):
                            for c in range(nub):
                                if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                                    l3b[i,j,k,a,b,c]=lccsdt[ndata+iroot*reclen]
                                ndata+=1

        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nua):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                                    l3c[i,j,k,a,b,c]=lccsdt[ndata+iroot*reclen]
                                ndata+=1

        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nub):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(lccsdt[ndata+iroot*reclen])>=thresh:
                                    l3d[i,j,k,a,b,c]=lccsdt[ndata+iroot*reclen]
                                ndata+=1
    
    l1a = l1a.transpose(1,0)
    l1b = l1b.transpose(1,0)
    l2a = l2a.transpose(2,3,0,1)
    l2b = l2b.transpose(2,3,0,1)
    l2c = l2c.transpose(2,3,0,1)
    if exc == 2:
        return {'l1a':l1a, 'l1b':l1b, 'l2a':l2a, 'l2b':l2b, 'l2c':l2c}
    else:
        l3a = l3a.transpose(3,4,5,0,1,2)
        l3b = l3b.transpose(3,4,5,0,1,2)
        l3c = l3c.transpose(3,4,5,0,1,2)
        l3d = l3d.transpose(3,4,5,0,1,2)
        return {'l1a':l1a, 'l1b':l1b, 'l2a':l2a, 'l2b': l2b, 'l2c':l2c, 'l3a':l3a, 'l3b':l3b, 'l3c':l3c, 'l3d':l3d}


def read_r_amp(rccsdt,iroot,noa,nob,nua,nub,exc):

    thresh = 0.0

    if iroot==0:
        print("R does not exist for ground state!")
        return
    
    r1a = np.zeros((nua,noa))
    r1b = np.zeros((nub,nob))
    r2a = np.zeros((nua,nua,noa,noa))
    r2b = np.zeros((nua,nub,noa,nob))
    r2c = np.zeros((nub,nub,nob,nob))
    #r1a = np.zeros((noa,nua))
    #r1b = np.zeros((nob,nub))
    #r2a = np.zeros((noa,noa,nua,nua))
    #r2b = np.zeros((noa,nob,nua,nub))
    #r2c = np.zeros((nob,nob,nub,nub))
    if exc > 2:
        r3a = np.zeros((noa,noa,noa,nua,nua,nua))
        r3b = np.zeros((noa,noa,nob,nua,nua,nub))
        r3c = np.zeros((noa,nob,nob,nua,nub,nub))
        r3d = np.zeros((nob,nob,nob,nub,nub,nub))

    reclen=noa*nua
    reclen+=nob*nub
    reclen+=noa*noa*nua*nua
    reclen+=noa*nob*nua*nub
    reclen+=nob*nob*nub*nub
    if exc > 2:
        reclen+=noa*noa*noa*nua*nua*nua
        reclen+=noa*noa*nob*nua*nua*nub
        reclen+=noa*nob*nob*nua*nub*nub
        reclen+=nob*nob*nob*nub*nub*nub
      
    ndata=0
    
    for i in range(noa):
        for a in range(nua):
            if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                r1a[a,i]=rccsdt[ndata+(iroot-1)*reclen]
            ndata+=1
    
    for i in range(nob):
        for a in range(nub):
            if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                r1b[a,i]=rccsdt[ndata+(iroot-1)*reclen]
            ndata+=1
            
    for i in range(noa):
        for j in range(noa):
            for a in range(nua):
                for b in range(nua):
                    if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                        r2a[b,a,j,i]=rccsdt[ndata+(iroot-1)*reclen]
                    ndata+=1
    
    for i in range(noa):
        for j in range(nob):
            for a in range(nua):
                for b in range(nub):
                    if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                        r2b[a,b,i,j]=rccsdt[ndata+(iroot-1)*reclen]
                    ndata+=1
    
    for i in range(nob):
        for j in range(nob):
            for a in range(nub):
                for b in range(nub):
                    if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                        r2c[b,a,j,i]=rccsdt[ndata+(iroot-1)*reclen]
                    ndata+=1
   
    if exc > 2:
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for a in range(nua):
                        for b in range(nua):
                            for c in range(nua):
                                if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                                    r3a[i,j,k,a,b,c]=rccsdt[ndata+(iroot-1)*reclen]
                                ndata+=1

        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for a in range(nua):
                        for b in range(nua):
                            for c in range(nub):
                                if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                                    r3b[i,j,k,a,b,c]=rccsdt[ndata+(iroot-1)*reclen]
                                ndata+=1

        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nua):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                                    r3c[i,j,k,a,b,c]=rccsdt[ndata+(iroot-1)*reclen]
                                ndata+=1

        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nub):
                        for b in range(nub):
                            for c in range(nub):
                                if abs(rccsdt[ndata+(iroot-1)*reclen])>=thresh:
                                    r3d[i,j,k,a,b,c]=rccsdt[ndata+(iroot-1)*reclen]
                                ndata+=1
    
        #r1a = r1a.transpose(1,0)
        #r1b = r1b.transpose(1,0)
        #r2a = r2a.transpose(2,3,1,0)
        #r2b = r2b.transpose(2,3,0,1)
        #r2c = r2c.transpose(2,3,0,1)
    if exc == 2:
        return {'r1a':r1a, 'r1b':r1b, 'r2a':r2a, 'r2b':r2b, 'r2c':r2c}
    else:
        r3a = r3a.transpose(3,4,5,0,1,2)
        r3b = r3b.transpose(3,4,5,0,1,2)
        r3c = r3c.transpose(3,4,5,0,1,2)
        r3d = r3d.transpose(3,4,5,0,1,2)
        return {'r1a':r1a, 'r1b':r1b, 'r2a':r2a, 'r2b':r2b, 'r2c':r2c, 'r3a':r3a, 'r3b':r3b, 'r3c':r3c, 'r3d':r3d}

def savevec(vec_dict,vectype,style='python'):

    if style=='matlab':
        from scipy.io import savemat
        savemat('/Users/karthik/Dropbox/Hartree Fock/hartree_fock/v4/CC_matlab/junreadsave-'+vectype+'.mat',vec_dict)
    else:
        np.save(vectype+'.npy',vec_dict)
    return

def main(args):

    noa = args.noa
    nob = args.nob
    nua = args.nua
    nub = args.nub
    frz = args.frz
    exc = args.exc
    fid = args.f_read
    iroot = args.root

    if args.vectype == 'T':
        with FortranFile(fid,'r') as f_in:
            first_line_reals = f_in.read_reals(dtype=np.float64)
            t_read = f_in.read_reals(dtype=np.float64)
        #t_read = FortranFile(fid).read_reals() #due to the structure of .moe and T-CCSDt
    if args.vectype == 'R':
        r_read = np.fromfile(fid,sep="",dtype=np.float64)
    if args.vectype == 'L':
        l_read = np.fromfile(fid,sep="",dtype=np.float64)

    if exc == 2:
        if args.vectype == 'T':
            vdict = read_t_amp(t_read,noa,nob,nua,nub,exc)
        if args.vectype == 'R':
            vdict = read_r_amp(r_read,iroot,noa,nob,nua,nub,exc)
        if args.vectype == 'L':
            vdict = read_l_amp(l_read,iroot,noa,nob,nua,nub,exc)
    if exc == 3:
        if args.vectype == 'T':
            vdict = read_t_amp(t_read,noa,nob,nua,nub,exc)
        if args.vectype == 'R':
            vdict = read_r_amp(r_read,iroot,noa,nob,nua,nub,exc)
        if args.vectype == 'L':
            vdict = read_l_amp(l_read,iroot,noa,nob,nua,nub,exc)

    savevec(vdict,args.vectype)

    tol = 1.0e-02

    for item in vdict.items():
        key = item[0]
        val = item[1]
        print('')
        if len(val.shape) == 2:
            for a in range(val.shape[0]):
                for i in range(val.shape[1]):
                    if abs(val[a,i]) > tol:
                        print('{}({},{}) = {}'.format(key,a+1,i+1,val[a,i]))
        if len(val.shape) == 4:
            for a in range(val.shape[0]):
                for b in range(val.shape[1]):
                    for i in range(val.shape[2]):
                        for j in range(val.shape[3]):
                            if abs(val[a,b,i,j]) > tol:
                                print('{}({},{},{},{}) = {}'.format(key,a+1,b+1,i+1,j+1,val[a,b,i,j]))




    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script that reads and writes L and R vectors contained in L-CC** and R-CC* files from Jun's code")
    parser.add_argument('noa',type=int,help='Number of occupied alpha spatial orbitals')
    parser.add_argument('nob',type=int,help='Number of occupied beta spatial orbitals')
    parser.add_argument('nua',type=int,help='Number of unoccupied alpha spatial orbitals')
    parser.add_argument('nub',type=int,help='Number of unoccuiped beta spatial orbitals')
    parser.add_argument('frz',type=int,help='Number of frozen spatial orbitals')
    parser.add_argument('f_read',type=str,help='Path to L or R vector file to be read')
    parser.add_argument('vectype',type=str,help="Charater 'T', 'L', or 'R' to indicate what kind of file is being read")
    parser.add_argument('-e','--exc',type=int,default=2,help='Maximum excitation rank to extract')
    parser.add_argument('-r','--root',type=int,default=0,help='Index of excited state root (starting from 1) that you want to read')
    args = parser.parse_args()
    main(args)
    


