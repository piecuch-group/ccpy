import numpy as np

def calc_t4_aaaa(T, dT, H, H0):
    # <ijklabcd | H(2) | 0 >
    dT.aaaa = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    dT.aaaa += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    dT.aaaa += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36
    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaaa -= (4.0 / 576.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaaa, optimize=True) # (l/ijk) = 4
    dT.aaaa += (4.0 / 576.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaaa, optimize=True) # (d/abc) = 4
    dT.aaaa += (6.0 / 576.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaaa, optimize=True) # (kl/ij) = 6
    dT.aaaa += (6.0 / 576.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaaa, optimize=True) # (cd/ab) = 6
    dT.aaaa += (16.0 / 576.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaaa, optimize=True) # (d/abc)(l/ijk) = 16
    dT.aaaa += (16.0 / 576.0) * np.einsum("amie,bcdejklm->abcdijkl", H.ab.voov, T.aaab, optimize=True) # (d/abc)(l/ijk) = 16
    I3A_vvvoov = (
                    -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", H0.aa.oovv, T.aaaa, optimize=True)
                    -np.einsum("mnef,bcdfjkmn->bcdjke", H0.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa += (24.0 / 576.0) * np.einsum("bcdjke,aeil->abcdijkl", I3A_vvvoov, T.aa, optimize=True) # (a/bcd)(jk/il) = 4 * 6 = 24

    I3A_vvoooo = (
                    0.5 * np.einsum("mnef,bcefjkln->bcmjkl", H0.aa.oovv, T.aaaa, optimize=True)
                    +np.einsum("mnef,bcefjkln->bcmjkl", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa -= (24.0 / 576.0) * np.einsum("bcmjkl,adim->abcdijkl", I3A_vvoooo, T.aa, optimize=True) # (bc/ad)(i/jkl) = 6 * 4 = 24
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaaa += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", H.aa.vvov, T.aaa, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    dT.aaaa -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", H.aa.vooo, T.aaa, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24

    I3A_vooooo = np.einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_vooooo, (0, 1, 2, 3, 5, 4))
    I3A_vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", H.aa.oovv, T.aaa, optimize=True)
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdimn->abcdijkl", I3A_vooooo, T.aaa, optimize=True) # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa, optimize=True) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", H0.aa.oovv, T.aaa, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
                    +0.25 * np.einsum("mnef,cdfkln->cdmkle", H0.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", H0.bb.oovv, T.aab, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # antisymmetrize the spin-integrated residuals
    dT.aaaa -= np.transpose(dT.aaaa, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    dT.aaaa -= np.transpose(dT.aaaa, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(dT.aaaa, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    dT.aaaa -= np.transpose(dT.aaaa, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(dT.aaaa, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(dT.aaaa, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    dT.aaaa -= np.transpose(dT.aaaa, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    dT.aaaa -= np.transpose(dT.aaaa, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(dT.aaaa, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    dT.aaaa -= np.transpose(dT.aaaa, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(dT.aaaa, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(dT.aaaa, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)

    return dT
