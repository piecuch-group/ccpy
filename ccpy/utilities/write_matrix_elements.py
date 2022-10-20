

def aa_H_aaaa():
    # A(ij)A(ab)A(mn/kl)A(ef/cd) delta(i,k)*delta(j,l)*delta(a,c)*delta(b,d)*v_aa(m,n,e,f)

    expr0 = "(i == k)*(j == l)*(a == c)*(b == d)*H.aa.oovv[m, n, e, f]"

    print("hmatel = (")
    for perm1, sign1 in {'1' : 1.0, 'i,j' : -1.0}.items():
        for perm2, sign2 in {'1' : 1.0, 'a,b' : -1.0}.items():
            for perm3, sign3 in {'1' : 1.0, 'm,k' : -1.0, 'm,l' : -1.0, 'n,k' : -1.0, 'n,l' : -1.0, }.items():
                for perm4, sign4 in {'1' : 1.0, 'e,c' : -1.0, 'e,d' : -1.0}.items():

                    temp = expr0
                    s1, temp = apply_asym(perm1, sign1, temp)
                    s2, temp = apply_asym(perm2, sign2, temp)
                    s3, temp = apply_asym(perm3, sign3, temp)
                    s4, temp = apply_asym(perm4, sign4, temp)

                    sign = s1 * s2 * s3 * s4
                    if sign == 1.0:
                        sign_str = '+'
                    else:
                        sign_str = '-'

                    print("    ", sign_str + temp)
    print(")")

def aa_H_aaab():
    # A(ij)A(ab)A(m/kl)A(e/cd) delta(i,k)*delta(j,l)*delta(a,c)*delta(b,d)*v_ab(m,n,e,f)

    expr0 = "(i == k)*(j == l)*(a == c)*(b == d)*H.ab.oovv[m, n, e, f]"

    print("hmatel = (")
    for perm1, sign1 in {'1' : 1.0, 'i,j' : -1.0}.items():
        for perm2, sign2 in {'1' : 1.0, 'a,b' : -1.0}.items():
            for perm3, sign3 in {'1' : 1.0, 'm,k' : -1.0, 'm,l' : -1.0}.items():
                for perm4, sign4 in {'1' : 1.0, 'e,c' : -1.0, 'e,d' : -1.0}.items():

                    temp = expr0
                    s1, temp = apply_asym(perm1, sign1, temp)
                    s2, temp = apply_asym(perm2, sign2, temp)
                    s3, temp = apply_asym(perm3, sign3, temp)
                    s4, temp = apply_asym(perm4, sign4, temp)

                    sign = s1 * s2 * s3 * s4
                    if sign == 1.0:
                        sign_str = '+'
                    else:
                        sign_str = '-'

                    print("    ", sign_str + temp)
    print(")")



def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")
    # if not erroring, but the index is still not in the correct range..
    #if index < 0: # add it to the beginning
    #        return newstring + s
    #if index > len(s): # add it to the end
    #        return s + newstring
    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index+1:]

def apply_asym(perm,sign,expr):
    import re
    if perm == '1':
        return sign, expr
    else:
        s = perm.split(',')
        result1 = [_.start() for _ in re.finditer(s[0], expr)] 
        result2 = [_.start() for _ in re.finditer(s[1], expr)]
        for idx in result1:
            expr = replacer(expr, s[1], idx)
        for idx in result2:
            expr = replacer(expr, s[0], idx)
        return sign, expr


if __name__ == '__main__':
    aa_H_aaab()
