import numpy as np

def calc_t4_aaab(T, dT, H, H0):
    # <ijklabcd | H(2) | 0 >
    dT.aaab = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)    # (i/jk)(c/ab) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)    # (k/ij)(a/bc) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    dT.aaab -= np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    dT.aaab += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    dT.aaab += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18
    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaab, optimize=True)  # (i/jk) = 3
    dT.aaab -= (1.0 / 36.0) * np.einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aaab, optimize=True)  # (1) = 1
    dT.aaab += (1.0 / 12.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaab, optimize=True)  # (a/bc) = 3
    dT.aaab += (1.0 / 36.0) * np.einsum("de,abceijkl->abcdijkl", H.b.vv, T.aaab, optimize=True)  # (1) = 1
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaab, optimize=True)  # (k/ij) = 3
    dT.aaab += (1.0 / 12.0) * np.einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aaab, optimize=True)  # (i/jk) = 3
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaab, optimize=True)  # (c/ab) = 3
    dT.aaab += (1.0 / 12.0) * np.einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aaab, optimize=True)  # (a/bc) = 3
    dT.aaab += (9.0 / 36.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaab, optimize=True)  # (a/bc)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("amie,bcedjkml->abcdijkl", H.ab.voov, T.aabb, optimize=True)  # (a/bc)(i/jk) = 9
    dT.aaab += (1.0 / 36.0) * np.einsum("mdel,abceijkm->abcdijkl", H.ab.ovvo, T.aaaa, optimize=True)  # (1) = 1
    dT.aaab += (1.0 / 36.0) * np.einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aaab, optimize=True)  # (1) = 1
    dT.aaab -= (1.0 / 12.0) * np.einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aaab, optimize=True)  # (a/bc) = 3
    dT.aaab -= (1.0 / 12.0) * np.einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aaab, optimize=True)  # (i/jk) = 3
    I3B_vvvvoo = (
        -0.5 * np.einsum("mnef,acfdmknl->acdekl", H.aa.oovv, T.aaab, optimize=True)
        - np.einsum("mnef,acfdmknl->acdekl", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("acdekl,ebij->abcdijkl", I3B_vvvvoo, T.aa, optimize=True)  # (b/ac)(k/ij) = 9

    I3A_vvvvoo = (
        -0.5 * np.einsum("mnef,abcfmjkn->abcejk", H.aa.oovv, T.aaaa, optimize=True)
        - np.einsum("mnef,abcfmjkn->abcejk", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaab += (1.0 / 12.0) * np.einsum("abcejk,edil->abcdijkl", I3A_vvvvoo, T.ab, optimize=True)  # (i/jk) = 3

    I3B_vvvoov = (
        - np.einsum("nmfe,abfdijnm->abdije", H.ab.oovv, T.aaab, optimize=True)
        - 0.5 * np.einsum("nmfe,abfdijnm->abdije", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("abdije,cekl->abcdijkl", I3B_vvvoov, T.ab, optimize=True)  # (c/ab)(k/ij) = 9

    I3B_vovooo = (
        0.5 * np.einsum("mnef,cefdkinl->cmdkil", H.aa.oovv, T.aaab, optimize=True)
        + np.einsum("mnef,cefdkinl->cmdkil", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aaab -= (9.0 / 36.0) * np.einsum("cmdkil,abmj->abcdijkl", I3B_vovooo, T.aa, optimize=True)  # (c/ab)(j/ik) = 9

    I3A_vovooo = (
        0.5 * np.einsum("mnef,bcefjkin->bmcjik", H.aa.oovv, T.aaaa, optimize=True)
        + np.einsum("mnef,bcefjkin->bmcjik", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaab -= (1.0 / 12.0) * np.einsum("bmcjik,adml->abcdijkl", I3A_vovooo, T.ab, optimize=True)  # (a/bc) = 3

    I3B_vvoooo = (
        np.einsum("nmfe,bcfejknl->bcmjkl", H.ab.oovv, T.aaab, optimize=True)
        + 0.5 * np.einsum("nmfe,bcfejknl->bcmjkl", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aaab -= (9.0 / 36.0) * np.einsum("bcmjkl,adim->abcdijkl", I3B_vvoooo, T.ab, optimize=True)  # (a/bc)(i/jk) = 9
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * np.einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa, optimize=True)  # (k/ij) = 3
    dT.aaab -= (9.0 / 36.0) * np.einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab, optimize=True)  # (j/ik)(a/bc) = 9
    dT.aaab -= (9.0 / 36.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    dT.aaab += (1.0 / 12.0) * np.einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa, optimize=True)  # (c/ab) = 3
    dT.aaab += (9.0 / 36.0) * np.einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab, optimize=True)  # (b/ac)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_oovooo = (
                    np.einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                   +0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa, optimize=True)  # (k/ij) = 3

    I3A_vooooo = np.einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * np.einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa, optimize=True)
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab, optimize=True)  # (c/ab) = 3

    I3B_vooooo = (
                    0.5 * np.einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa, optimize=True)
                  + np.einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab, optimize=True)
                  + 0.5 * np.einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3B_vvvvvo = -np.einsum("amef,bdml->abdefl", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa, optimize=True)  # (c/ab) = 3

    I3A_vvvvvo = -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab, optimize=True)  # (k/ij) = 3

    I3B_vvvovv = (
                    -0.5 * np.einsum("mdef,acim->acdief", H.ab.ovvv, T.aa, optimize=True)
                    - np.einsum("cmef,adim->acdief", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovovo = (
                    -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                    +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
                    -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                    +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa, optimize=True)  # (a/bc)(i/jk) = 9

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab, optimize=True)  # (c/ab)(k/ij) = 9

    I3B_vvoovo = (
                -0.5 * np.einsum("nmel,acin->acmiel", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("cmef,afil->acmiel", H.ab.vovv, T.ab, optimize=True)
                - 0.5 * np.einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    dT.aaab -= (9.0 / 36.0) * np.einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovoov = (
                0.5 * np.einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("mnke,adin->amdike", H.ab.ooov, T.ab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aaab -= (9.0 / 36.0) * np.einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3C_vvooov = (
                -np.einsum("nmie,adnl->admile", H.ab.ooov, T.ab, optimize=True)
                -np.einsum("nmle,adin->admile", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->admile", H.ab.vovv, T.ab, optimize=True)
                +np.einsum("dmfe,afil->admile", H.bb.vovv, T.ab, optimize=True)
                +np.einsum("mnef,afdinl->admile", H.bb.oovv, T.abb, optimize=True)  # added 5/2/22
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb, optimize=True)  # (c/ab)(k/ij) = 9


    dT.aaab -= np.transpose(dT.aaab, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    dT.aaab -= np.transpose(dT.aaab, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(dT.aaab, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    dT.aaab -= np.transpose(dT.aaab, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    dT.aaab -= np.transpose(dT.aaab, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(dT.aaab, (1, 0, 2, 3, 4, 5, 6, 7)) # (a/bc)

    return dT
