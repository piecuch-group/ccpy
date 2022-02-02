from itertools import combinations

import numpy as np

A_i1j1 = {'1' : 1.0, 'i1,j1' : -1.0}
A_j1k1 = {'1' : 1.0, 'j1,k1' : -1.0}
A_b1c1 = {'1' : 1.0, 'b1,c1' : -1.0}
A_a1b1 = {'1' : 1.0, 'a1,b1' : -1.0}
A_i1_j1k1 = {'1' : 1.0, 'i1,j1' : -1.0, 'i1,k1' : -1.0}
A_i2_j2k2 = {'1' : 1.0, 'i2,j2' : -1.0, 'i2,k2' : -1.0}
A_a1_b1c1 = {'1' : 1.0, 'a1,b1' : -1.0, 'a1,c1' : -1.0}
A_a2_b2c2 = {'1' : 1.0, 'a2,b2' : -1.0, 'a2,c2' : -1.0}
A_k1_i1j1 = {'1' : 1.0, 'k1,i1' : -1.0, 'k1,j1' : -1.0}
A_c1_a1b1 = {'1' : 1.0, 'c1,a1' : -1.0, 'c1,b1' : -1.0}

def t3a_diagram5():
    # A(jk)A(bc)A(i/jk)A(l/mn)A(a/bc)A(d/ef) delta(k,n)delta(j,m)delta(b,e)delta(c,f)*h2A(a,l,i,d)

    expr0 = "(k1==k2)*(j1==j2)*(b1==b2)*(c1==c2)*H2A['voov'][a1,i2,i1,a2]"

    for perm1,sign1 in A_j1k1.items():
        for perm2,sign2 in A_b1c1.items():
            for perm3,sign3 in A_i1_j1k1.items():
                for perm4,sign4 in A_i2_j2k2.items():
                    for perm5,sign5 in A_a1_b1c1.items():
                        for perm6,sign6 in A_a2_b2c2.items():

                            temp = expr0
                            s1,temp = apply_asym(perm1,sign1,temp)
                            s2,temp = apply_asym(perm2,sign2,temp)
                            s3,temp = apply_asym(perm3,sign3,temp)
                            s4,temp = apply_asym(perm4,sign4,temp)
                            s5,temp = apply_asym(perm5,sign5,temp)
                            s6,temp = apply_asym(perm6,sign6,temp)
            
                            sign = s1*s2*s3*s4*s5*s6
                            if sign == 1.0:
                                sign_str = '+'
                            else: 
                                sign_str = '-'

                            print(sign_str+temp+'\\')

def t3a_diagram6():
    # A(ij)A(ab)A(k/ij)A(c/ab) delta(i,l)delta(j,m)delta(b,e)delta(a,d)*h2B(c,n,k,f)
    expr0 = "(i1==i2)*(j1==j2)*(b1==b2)*(a1==a2)*H2B['voov'][c1,k2,k1,c2]"
    for perm1, sign1 in A_i1j1.items():
        for perm2, sign2 in A_a1b1.items():
            for perm3, sign3 in A_k1_i1j1.items():
                for perm4, sign4 in A_c1_a1b1.items():
                    temp = expr0
                    s1,temp = apply_asym(perm1,sign1,temp)
                    s2,temp = apply_asym(perm2,sign2,temp)
                    s3,temp = apply_asym(perm3,sign3,temp)
                    s4,temp = apply_asym(perm4,sign4,temp)
                    sign = s1*s2*s3*s4
                    if sign == 1.0:
                        sign_str = '+'
                    else: 
                        sign_str = '-'
                    print(sign_str+temp+'\\')

def t3a_diagram7():
    #

def t3a_diagram8():
    #

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")
    # if not erroring, but the index is still not in the correct range..
    if index < 0: # add it to the beginning
            return newstring + s
    if index > len(s): # add it to the end
            return s + newstring
    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index+2:]

def apply_asym(perm,sign,expr):
    import re
    if perm == '1':
        return sign, expr
    else:
        s = perm.split(',')
        result1 = [_.start() for _ in re.finditer(s[0], expr)] 
        result2 = [_.start() for _ in re.finditer(s[1], expr)]
        for idx in result1:
            expr = replacer(expr,s[1],idx)
        for idx in result2:
            expr = replacer(expr,s[0],idx)
        return sign, expr


if __name__ == '__main__':
    t3a_diagram6()
