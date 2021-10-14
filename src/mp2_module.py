import numpy as np

def mp2(sys,ints):

    fA = ints['fA']
    fB = ints['fB']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    Ecorr = 0.0
    for i in range(sys['Nocc_a']):
        for j in range(i+1,sys['Nocc_a']):
            for a in range(sys['Nunocc_a']):
                for b in range(a+1,sys['Nunocc_a']):
                    denom = fA['oo'][i,i]+fA['oo'][j,j]-fA['vv'][a,a]-fA['vv'][b,b]
                    Ecorr += vA['oovv'][i,j,a,b]*vA['vvoo'][a,b,i,j]/denom
    for i in range(sys['Nocc_a']):
        for j in range(sys['Nocc_b']):
            for a in range(sys['Nunocc_a']):
                for b in range(sys['Nunocc_b']):
                    denom = fA['oo'][i,i]+fB['oo'][j,j]-fA['vv'][a,a]-fB['vv'][b,b]
                    Ecorr += vB['oovv'][i,j,a,b]*vB['vvoo'][a,b,i,j]/denom
    for i in range(sys['Nocc_b']):
        for j in range(i+1,sys['Nocc_b']):
            for a in range(sys['Nunocc_b']):
                for b in range(a+1,sys['Nunocc_b']):
                    denom = fB['oo'][i,i]+fB['oo'][j,j]-fB['vv'][a,a]-fB['vv'][b,b]
                    Ecorr += vC['oovv'][i,j,a,b]*vC['vvoo'][a,b,i,j]/denom

    Emp2 = Ecorr + ints['Escf']
    print('THE MP2 CORRELATION ENERGY = {} HARTREE'.format(Ecorr))
    print('THE MP2 TOTAL ENERGY = {} HARTREE'.format(Ecorr+ints['Escf']))

    return Emp2
                                    
