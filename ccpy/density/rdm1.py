import numpy as np


def calc_rdm_oo(cc_t,sys,mu,nu):

    noa = sys['Nocc_a']
    nob = sys['Nocc_b']

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

    l1a = cc_t['l1a'][mu]
    l1b = cc_t['l1b'][mu]
    l2a = cc_t['l2a'][mu]
    l2b = cc_t['l2b'][mu]
    l2c = cc_t['l2c'][mu]
    l3a = cc_t['l3a'][mu]
    l3b = cc_t['l3b'][mu]
    l3c = cc_t['l3c'][mu]
    l3d = cc_t['l3d'][mu]

    if nu > 0:
        r0 = cc_t['r0'][nu-1]
        r1a = cc_t['r1a'][nu-1]
        r1b = cc_t['r1b'][nu-1]
        r2a = cc_t['r2a'][nu-1]
        r2b = cc_t['r2b'][nu-1]
        r2c = cc_t['r2c'][nu-1]
        r3a = cc_t['r3a'][nu-1]
        r3b = cc_t['r3b'][nu-1]
        r3c = cc_t['r3c'][nu-1]
        r3d = cc_t['r3d'][nu-1]

    rdm1_ooa = np.zeros((noa,noa))
    rdm1_oob = np.zeros((nob,nob))

    rdm1_ooa += (-1.0*np.einsum('ei,ej->ij', l1a, t1a, optimize=True)
                -0.5*np.einsum('efin,efjn->ij', l2a, t2a, optimize=True)
                -np.einsum('efin,efjn->ij',l2b,t2b,optimize=True)
                -(1.0/12.0)*np.einsum('efgino,efgjno->ij',l3a,t3a,optimize=True)
                -0.5*np.einsum('efgino,efgjno->ij',l3b,t3b,optimize=True)
                -0.25*np.einsum('efgino,efgjno->ij',l3c,t3c,optimize=True))

    rdm1_oob += -1.0*np.einsum('ei,ej->ij',l1b,t1b,optimize=True)\
                -0.5*np.einsum('efin,efjn->ij',l2c,t2c,optimize=True)\
                -np.einsum('feni,fenj->ij',l2b,t2b,optimize=True)\
                -(1.0/12.0)*np.einsum('efgino,efgjno->ij',l3d,t3d,optimize=True)\
                -0.5*np.einsum('efgoni,efgonj->ij',l3c,t3c,optimize=True)\
                -0.25*np.einsum('efgoni,efgonj->ij',l3b,t3b,optimize=True)

    if nu > 0: # Excited-state RDM contributions
        rdm1_ooa *= r0
        rdm1_ooa += -1.0*np.einsum('ei,ej->ij',l1a,r1a,optimize=True)\
                    -0.5*np.einsum('efin,efjn->ij',l2a,r2a,optimize=True)\
                    -np.einsum('efin,efjn->ij',l2b,r2b,optimize=True)\
                    -(1.0/12.0)*np.einsum('efgino,efgjno->ij',l3a,r3a,optimize=True)\
                    -0.5*np.einsum('efgino,efgjno->ij',l3b,r3b,optimize=True)\
                    -0.25*np.einsum('efgino,efgjno->ij',l3c,r3c,optimize=True)\
                    -np.einsum('efin,ej,fn->ij',l2a,t1a,r1a,optimize=True)\
                    -np.einsum('efin,ej,fn->ij',l2b,t1a,r1b,optimize=True)\

