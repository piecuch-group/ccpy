import numpy as np

def calc_t4(t1, t2, t3, t4, H1, H2, f, g, o, v):

    # <ijklabcd | H(2) | 0 >
    X4 = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H2[v, o, o, v], t2, t2, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    X4 += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H2[o, o, o, o], t2, t2, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    X4 += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H2[v, v, v, v], t2, t2, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36
    # <ijklabcd | (H(2)*T4)_C | 0 >
    X4 -= (4.0 / 576.0) * np.einsum("mi,abcdmjkl->abcdijkl", H1[o, o], t4, optimize=True) # (l/ijk) = 4
    X4 += (4.0 / 576.0) * np.einsum("ae,ebcdijkl->abcdijkl", H1[v, v], t4, optimize=True) # (d/abc) = 4
    X4 += (6.0 / 576.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H2[o, o, o, o], t4, optimize=True) # (kl/ij) = 6
    X4 += (6.0 / 576.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H2[v, v, v, v], t4, optimize=True) # (cd/ab) = 6
    X4 += (16.0 / 576.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H2[v, o, o, v], t4, optimize=True) # (d/abc)(l/ijk) = 16
    I_vvvoov = -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", g[o, o, v, v], t4, optimize=True)
    X4 += (24.0 / 576.0) * np.einsum("bcdjke,aeil->abcdijkl", I_vvvoov, t2, optimize=True) # (a/bcd)(jk/il) = 4 * 6 = 24
    I_vvoooo = 0.5 * np.einsum("mnef,bcefjkln->bcmjkl", g[o, o, v, v], t4, optimize=True)
    X4 -= (24.0 / 576.0) * np.einsum("bcmjkl,adim->abcdijkl", I_vvoooo, t2, optimize=True) # (bc/ad)(i/jkl) = 6 * 4 = 24
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    X4 += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", H2[v, v, o, v], t3, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    X4 -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", H2[v, o, o, o], t3, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24

    I_vooooo = np.einsum("nmle,bejk->bmnjkl", H2[o, o, o, v], t2, optimize=True)
    I_vooooo -= np.transpose(I_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I_vooooo, (0, 1, 2, 3, 5, 4))
    I_vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", g[o, o, v, v], t3, optimize=True) 
    X4 += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdimn->abcdijkl", I_vooooo, t3, optimize=True) # (b/acd)(i/jkl) = 4 * 4 = 16

    I_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H2[v, o, v, v], t2, optimize=True)
    I_vvvovv -= np.transpose(I_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I_vvvovv, (0, 2, 1, 3, 4, 5))
    X4 += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I_vvvovv, t3, optimize=True) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H2[o, o, o, v], t2, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H2[v, o, v, v], t2, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", g[o, o, v, v], t3, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I_vvooov -= np.transpose(I_vvooov, (0, 1, 2, 4, 3, 5))
    I_vvooov -= np.transpose(I_vvooov, (1, 0, 2, 3, 4, 5))
    X4 += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I_vvooov, t3, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # antisymmetrize the spin-orbital residuals
    X4 -= np.transpose(X4, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    X4 -= np.transpose(X4, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(X4, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    X4 -= np.transpose(X4, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(X4, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(X4, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    X4 -= np.transpose(X4, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    X4 -= np.transpose(X4, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(X4, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    X4 -= np.transpose(X4, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(X4, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(X4, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)

    return X4, I_vvooov
