import numpy as np
import mbpt_loops

def calc_mp2(sys,ints):

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

    return Ecorr

def mp2(sys,ints):

    print('\n   =============+++MBPT(2) Calculation+++=============')
    fA = ints['fA']; fB = ints['fB'];
    vA = ints['vA']; vB = ints['vB']; vC = ints['vC'];
    Ecorr = mbpt_loops.mbpt_loops.mp2(fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    vA['oovv'],vA['vvoo'],vB['oovv'],vB['vvoo'],vC['oovv'],vC['vvoo'])
    
    Emp2 = Ecorr + ints['Escf']

    print('     MBPT(2) correlation energy = {} hartree'.format(Ecorr))
    print('     Total MBPT(2) energy = {} hartree'.format(Emp2))

    return Emp2


def mp3(sys,ints):

    print('\n   =============+++MBPT(3) Calculation+++=============')
    fA = ints['fA']; fB = ints['fB'];
    vA = ints['vA']; vB = ints['vB']; vC = ints['vC'];
    Ecorr2 = mbpt_loops.mbpt_loops.mp2(fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    vA['oovv'],vA['vvoo'],vB['oovv'],vB['vvoo'],vC['oovv'],vC['vvoo'])
    Ecorr3 = mbpt_loops.mbpt_loops.mp3(fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    vA['oovv'],vA['vvoo'],vA['voov'],vA['oooo'],vA['vvvv'],\
                    vB['oovv'],vB['vvoo'],vB['voov'],vB['ovvo'],vB['vovo'],vB['ovov'],vB['oooo'],vB['vvvv'],\
                    vC['oovv'],vC['vvoo'],vC['voov'],vC['oooo'],vC['vvvv'])
    Ecorr = Ecorr2 + Ecorr3
    Emp3 = Ecorr + ints['Escf']

    print('     E(2) correlation energy = {} hartree'.format(Ecorr2))
    print('     E(3) correlation energy = {} hartree'.format(Ecorr3))
    print('     MBPT(3) correlation energy = {} hartree'.format(Ecorr))
    print('     Total MBPT(3) energy = {} hartree'.format(Emp3))

    return Emp3
                                    
