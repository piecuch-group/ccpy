import numpy as np

def calc_t4_aabb(T, dT, H, H0):
    # <ijklabcd | H(2) | 0 >
    dT.aabb = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (kl)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)    # (ij)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (ij)(kl)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)    # (ij)(kl)(ab) = 8
    dT.aabb -= np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
    dT.aabb -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)   # (kl)(ab) = 4
    dT.aabb += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
    dT.aabb += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
    dT.aabb += np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aabb -= (2.0 / 16.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aabb, optimize=True)  # [1]  (ij) = 2
    dT.aabb -= (2.0 / 16.0) * np.einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aabb, optimize=True)  # [2]  (kl) = 2
    dT.aabb += (2.0 / 16.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aabb, optimize=True)  # [3]  (ab) = 2
    dT.aabb += (2.0 / 16.0) * np.einsum("de,abceijkl->abcdijkl", H.b.vv, T.aabb, optimize=True)  # [4]  (cd) = 2
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aabb, optimize=True)  # [5]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aabb, optimize=True)  # [6]  (ij)(kl) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("mnkl,abcdijmn->abcdijkl", H.bb.oooo, T.aabb, optimize=True)  #  [7]  (1) = 1
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aabb, optimize=True)  #  [8]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aabb, optimize=True)  #  [9]  (ab)(cd) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("cdef,abefijkl->abcdijkl", H.bb.vvvv, T.aabb, optimize=True)  #  [10]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aabb, optimize=True)  #  [11]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("amie,becdjmkl->abcdijkl", H.ab.voov, T.abbb, optimize=True)  #  [12]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("mdel,aebcimjk->abcdijkl", H.ab.ovvo, T.aaab, optimize=True)  #  [13]  (kl)(cd) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aabb, optimize=True)  #  [14]  (kl)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * np.einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aabb, optimize=True)  #  [15]  (ij)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * np.einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aabb, optimize=True)  #  [16]  (kl)(ab) = 4
    I3C_vvvvoo = (
                -0.5 * np.einsum("mnef,afcdmnkl->acdekl", H.aa.oovv, T.aabb, optimize=True)
                -np.einsum("mnef,afcdmnkl->acdekl", H.ab.oovv, T.abbb, optimize=True)
    )
    dT.aabb += (2.0 / 16.0) * np.einsum("acdekl,beji->abcdijkl", I3C_vvvvoo, T.aa, optimize=True)  #  [17]  (ab) = 2

    I3B_vvvvoo = (
                -0.5 * np.einsum("mnef,abfcmjnk->abcejk", H.aa.oovv, T.aaab, optimize=True)
                -np.einsum("mnef,abfcmjnk->abcejk", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aabb += (8.0 / 16.0) * np.einsum("abcejk,edil->abcdijkl", I3B_vvvvoo, T.ab, optimize=True)  #  [18]  (ij)(kl)(cd) = 8

    I3C_vvvoov = (
                -np.einsum("nmfe,bfcdjnkm->bcdjke", H.ab.oovv, T.aabb, optimize=True)
                -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", H.bb.oovv, T.abbb, optimize=True)
    )
    dT.aabb += (8.0 / 16.0) * np.einsum("bcdjke,aeil->abcdijkl", I3C_vvvoov, T.ab, optimize=True)  #  [19]  (ij)(kl)(ab) = 8

    I3B_vvvoov = (
                -np.einsum("nmfe,abfdijnm->abdije", H.ab.oovv, T.aaab, optimize=True)
                -0.5 * np.einsum("mnef,abfdijnm->abdije", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aabb += (2.0 / 16.0) * np.einsum("abdije,eclk->abcdijkl", I3B_vvvoov, T.bb, optimize=True)  #  [20]  (cd) = 2

    I3C_ovvooo = (
                0.5 * np.einsum("mnef,efcdinkl->mcdikl", H.aa.oovv, T.aabb, optimize=True)
                +np.einsum("mnef,efcdinkl->mcdikl", H.ab.oovv, T.abbb, optimize=True)
    )
    dT.aabb -= (2.0 / 16.0) * np.einsum("mcdikl,abmj->abcdijkl", I3C_ovvooo, T.aa, optimize=True)  #  [21]  (ij) = 2

    I3B_vovooo = (
                0.5 * np.einsum("mnef,befcjink->bmcjik", H.aa.oovv, T.aaab, optimize=True)
                +np.einsum("mnef,befcjink->bmcjik", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aabb -= (8.0 / 16.0) * np.einsum("bmcjik,adml->abcdijkl", I3B_vovooo, T.ab, optimize=True)  #  [22]  (ab)(kl)(cd) = 8

    I3C_vovooo = (
                np.einsum("nmfe,bfecjnlk->bmcjlk", H.ab.oovv, T.aabb, optimize=True)
                +0.5 * np.einsum("mnef,bfecjnlk->bmcjlk", H.bb.oovv, T.abbb, optimize=True)
    )
    dT.aabb -= (8.0 / 16.0) * np.einsum("bmcjlk,adim->abcdijkl", I3C_vovooo, T.ab, optimize=True)  #  [23]  (ij)(ab)(cd) = 8

    I3B_vvoooo = (
                np.einsum("nmfe,abfeijnl->abmijl", H.ab.oovv, T.aaab, optimize=True)
                +0.5 * np.einsum("mnef,abfeijnl->abmijl", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aabb -= (2.0 / 16.0) * np.einsum("abmijl,cdkm->abcdijkl", I3B_vvoooo, T.bb, optimize=True)  #  [24]  (kl) = 2
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aabb -= (8.0 / 16.0) * np.einsum("mdil,abcmjk->abcdijkl", H.ab.ovoo, T.aab, optimize=True)  # [1]  (ij)(kl)(cd) = 8
    dT.aabb -= (2.0 / 16.0) * np.einsum("bmji,acdmkl->abcdijkl", H.aa.vooo, T.abb, optimize=True)  # [2]  (ab) = 2
    dT.aabb -= (2.0 / 16.0) * np.einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.aab, optimize=True)  # [3]  (cd) = 2
    dT.aabb -= (8.0 / 16.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.abb, optimize=True)  # [4]  (ij)(ab)(kl) = 8
    dT.aabb += (8.0 / 16.0) * np.einsum("adel,becjik->abcdijkl", H.ab.vvvo, T.aab, optimize=True)  # [5]  (ab)(kl)(cd) = 8
    dT.aabb += (2.0 / 16.0) * np.einsum("baje,ecdikl->abcdijkl", H.aa.vvov, T.abb, optimize=True)  # [6]  (ij) = 2
    dT.aabb += (8.0 / 16.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.abb, optimize=True)  # [7]  (ij)(ab)(cd) = 8
    dT.aabb += (2.0 / 16.0) * np.einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.aab, optimize=True)  # [8]  (kl) = 2

    I3B_oovooo = (
                np.einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab, optimize=True)
               + 0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aab, optimize=True)  # [9]  (kl)(cd) = 4

    I3B_ovoooo = (
                np.einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa, optimize=True)
                + 0.5 * np.einsum("mnef,befjil->mbnijl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("mbnijl,acdmkn->abcdijkl", I3B_ovoooo, T.abb, optimize=True)  # [10]  (kl)(ab) = 4

    I3C_vooooo = (
                np.einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,aefikl->amnikl", H.bb.oovv, T.abb, optimize=True)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("amnikl,bcdjmn->abcdijkl", I3C_vooooo, T.abb, optimize=True)  # [11]  (ij)(ab) = 4

    I3C_oovooo = (
                0.5 * np.einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb, optimize=True)
                + np.einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,efcilk->mncilk", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * np.einsum("mncilk,abdmjn->abcdijkl", I3C_oovooo, T.aab, optimize=True)  # [12]  (ij)(cd) = 4

    I3B_vvvvvo = -np.einsum("bmfe,acmk->abcefk", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3B_vvvvvo, T.aab, optimize=True)  # [13]  (kl)(cd) = 4

    I3C_vvvvov = (
                -np.einsum("mdef,acmk->acdekf", H.ab.ovvv, T.ab, optimize=True)
                - 0.5 * np.einsum("amef,cdkm->acdekf", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("acdekf,ebfijl->abcdijkl", I3C_vvvvov, T.aab, optimize=True)  # [14]  (kl)(ab) = 4

    I3B_vvvvov = (
                -0.5 * np.einsum("mdef,abmj->abdejf", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("amef,bdjm->abdejf", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("abdejf,efcilk->abcdijkl", I3B_vvvvov, T.abb, optimize=True)  # [15]  (ij)(cd) = 4

    I3C_vvvovv = -np.einsum("cmef,adim->acdief", H.bb.vovv, T.ab, optimize=True)
    I3C_vvvovv -= np.transpose(I3C_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("acdief,befjkl->abcdijkl", I3C_vvvovv, T.abb, optimize=True)  # [16]  (ij)(ab) = 4

    ## !!! ERROR IS SOMERWHERE HERE !!! -> in the h(ooov) * t3 and h(vovv) * t3 parts, t3**2 diagrams are fine ##
    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H0.aa.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H0.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3A_vvooov, T.abb, optimize=True)  # [17]  (1) = 1

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.bb.oovv, T.aab, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3B_vvooov, T.bbb, optimize=True)  # [18]  (1) = 1

    I3C_ovvvoo = (
                -0.5 * np.einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb, optimize=True)
                +0.5 * np.einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb, optimize=True)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (1.0 / 16.0) * np.einsum("mcdekl,abeijm->abcdijkl", I3C_ovvvoo, T.aaa, optimize=True)  # [19]  (1) = 1

    I3D_vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb, optimize=True)
                +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb, optimize=True)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.aab, optimize=True)  # [20]  (1) = 1

    I3B_vovovo = (
                -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True) # !!! factor 1/2 to compensate asym
                +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True) 
                -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
    )
    dT.aabb += np.einsum("amdiel,becjmk->abcdijkl", I3B_vovovo, T.aab, optimize=True)  # [21]  (ij)(kl)(ab)(cd) = 16

    I3C_vovovo = (
                -np.einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab, optimize=True)
                -np.einsum("nmle,adin->amdiel", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.bb.oovv, T.abb, optimize=True) # !!! factor 1/2 to compensate asym
    )
    dT.aabb += np.einsum("amdiel,becjmk->abcdijkl", I3C_vovovo, T.abb, optimize=True)  # [22]  (ij)(kl)(ab)(cd) = 16

    I3B_vovoov = (
                -np.einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab, optimize=True)
                +0.5 * np.einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa, optimize=True)
                -0.5 * np.einsum("mnfe,bfdjin->bmdjie", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("bmdjie,aecmlk->abcdijkl", I3B_vovoov, T.abb, optimize=True)  # [23]  (ab)(cd) = 4

    I3C_ovvoov = (
                +np.einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab, optimize=True)
                -0.5 * np.einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb, optimize=True)
                -0.5 * np.einsum("mnfe,fcdikn->mcdike", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_ovvoov -= np.transpose(I3C_ovvoov, (0, 2, 1, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("mcdike,abemjl->abcdijkl", I3C_ovvoov, T.aab, optimize=True)  # [24]  (ij)(kl) = 4

    I3B_vvovoo = (
                -0.5 * np.einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa, optimize=True)
                +np.einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("abmejl,ecdikm->abcdijkl", I3B_vvovoo, T.abb, optimize=True)  # [25]  (ij)(kl) = 4

    I3C_vovvoo = (
                -np.einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab, optimize=True)
                +0.5 * np.einsum("amef,fclk->amcelk", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb -= (4.0 / 16.0) * np.einsum("amcelk,bedjim->abcdijkl", I3C_vovvoo, T.aab, optimize=True)  # [26]  (ab)(cd) = 4

    dT.aabb -= np.transpose(dT.aabb, (1, 0, 2, 3, 4, 5, 6, 7))
    dT.aabb -= np.transpose(dT.aabb, (0, 1, 3, 2, 4, 5, 6, 7))
    dT.aabb -= np.transpose(dT.aabb, (0, 1, 2, 3, 5, 4, 6, 7))
    dT.aabb -= np.transpose(dT.aabb, (0, 1, 2, 3, 4, 5, 7, 6))

    return dT, I3A_vvooov, I3B_vvooov, I3C_ovvvoo, I3D_vvooov, I3B_vovovo, I3C_vovovo, I3B_vovoov, I3C_ovvoov, I3B_vvovoo, I3C_vovvoo

