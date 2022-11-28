import numpy as np
from ccpy.models.integrals import Integral
from ccpy.utilities.active_space import get_active_slices

def get_3body_intermediates(T, H, system, spincase):

    X = Integral.from_empty(system, 3, data_type=T.a.dtype, use_none=True)

    # Get active-space slicing objects
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    if spincase == "aaaa":

        X.aaa.vooooo = np.einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa, optimize=True)
        X.aaa.vooooo -= np.transpose(X.aaa.vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(X.aaa.vooooo,
                                                                                      (0, 1, 2, 3, 5, 4))
        X.aaa.vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", H.aa.oovv, T.aaa, optimize=True)

        X.aaa.vvoooo = 0.5 * np.einsum("amef,ebfijl->abmijl", H.aa.vovv, T.aaa, optimize=True)
        X.aaa.vvoooo -= np.transpose(X.aaa.vvoooo, (1, 0, 2, 3, 4, 5))

        X.aaa.vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa, optimize=True)
                + 0.125 * np.einsum("mnef,cdfkln->cdmkle", H.aa.oovv, T.aaa, optimize=True)
                + 0.25 * np.einsum("mnef,cdfkln->cdmkle", H.ab.oovv, T.aab, optimize=True)
        )
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (0, 1, 2, 4, 3, 5))
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (1, 0, 2, 3, 4, 5))

        X.aab.vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa, optimize=True)
                + 0.125 * np.einsum("mnef,cdfkln->cdmkle", H.bb.oovv, T.aab, optimize=True)
        )
        X.aab.vvooov -= np.transpose(X.aab.vvooov, (1, 0, 2, 3, 4, 5))
        X.aab.vvooov -= np.transpose(X.aab.vvooov, (0, 1, 2, 4, 3, 5))

        # for the moments and V*T4 terms


    elif spincase == "aaab":

        X.aab.oovooo = (
                np.einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
        )
        X.aab.oovooo -= np.transpose(X.aab.oovooo, (0, 1, 2, 4, 3, 5))

        X.aaa.vooooo = np.einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa, optimize=True)
        X.aaa.vooooo -= np.transpose(X.aaa.vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(X.aaa.vooooo,
                                                                                      (0, 1, 2, 5, 4, 3))
        X.aaa.vooooo += 0.5 * np.einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa, optimize=True)

        X.aab.vooooo = (
                0.5 * np.einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab, optimize=True)
        )
        X.aab.vooooo -= np.transpose(X.aab.vooooo, (0, 1, 2, 4, 3, 5))

        X.aaa.vvoooo = 0.5 * np.einsum("amef,efcijk->acmikj", H.aa.vovv, T.aaa, optimize=True)
        X.aaa.vvoooo -= np.transpose(X.aaa.vvoooo, (1, 0, 2, 3, 4, 5))

        X.aab.vovooo = (
                0.5 * np.einsum("amef,efdijl->amdijl", H.aa.vovv, T.aab, optimize=True)
                + np.einsum("mdef,befjkl->bmdjkl", H.ab.ovvv, T.aab, optimize=True)
        )

        X.aab.vvoooo = np.einsum("amef,ebfijl->abmijl", H.ab.vovv, T.aab, optimize=True)
        X.aab.vvoooo -= np.transpose(X.aab.vvoooo, (1, 0, 2, 3, 4, 5))

        X.aab.vovovo = (
                -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                + np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
                - np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                + np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                + np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
        )

        X.aaa.vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                + 0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
                + 0.25 * np.einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa, optimize=True)
        )
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (1, 0, 2, 3, 4, 5))
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (0, 1, 2, 4, 3, 5))

        X.aab.vvoovo = (
                -0.5 * np.einsum("nmel,acin->acmiel", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("cmef,afil->acmiel", H.ab.vovv, T.ab, optimize=True)
                - 0.5 * np.einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab, optimize=True)
        )
        X.aab.vvoovo -= np.transpose(X.aab.vvoovo, (1, 0, 2, 3, 4, 5))

        X.aab.vovoov = (
                0.5 * np.einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa, optimize=True)
                - np.einsum("mnke,adin->amdike", H.ab.ooov, T.ab, optimize=True)
        )
        X.aab.vovoov -= np.transpose(X.aab.vovoov, (0, 1, 2, 4, 3, 5))

        X.abb.vvooov = (
                -np.einsum("nmie,adnl->admile", H.ab.ooov, T.ab, optimize=True)
                - np.einsum("nmle,adin->admile", H.bb.ooov, T.ab, optimize=True)
                + np.einsum("amfe,fdil->admile", H.ab.vovv, T.ab, optimize=True)
                + np.einsum("dmfe,afil->admile", H.bb.vovv, T.ab, optimize=True)
                + np.einsum("mnef,afdinl->admile", H.bb.oovv, T.abb, optimize=True)
        )

        X.aab.vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
        )
        X.aaab.vvooov -= np.transpose(X.aaab.vvooov, (1, 0, 2, 3, 4, 5))
        X.aaab.vvooov -= np.transpose(X.aaab.vvooov, (0, 1, 2, 4, 3, 5))

    elif spincase == "aabb":

        X.aab.oovooo = (
                np.einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
        )
        X.aab.oovooo -= np.transpose(X.aab.oovooo, (0, 1, 2, 4, 3, 5))

        X.aab.ovoooo = (
                np.einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa, optimize=True)
                + 0.5 * np.einsum("mnef,befjil->mbnijl", H.ab.oovv, T.aab, optimize=True)
        )
        X.aab.ovoooo -= np.transpose(X.aab.ovoooo, (0, 1, 2, 4, 3, 5))

        X.abb.vooooo = (
                np.einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,aefikl->amnikl", H.bb.oovv, T.abb, optimize=True)
        )
        X.abb.vooooo -= np.transpose(X.abb.vooooo, (0, 1, 2, 3, 5, 4))

        X.abb.oovooo = (
                0.5 * np.einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb, optimize=True)
                + np.einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,efcilk->mncilk", H.ab.oovv, T.abb, optimize=True)
        )
        X.abb.oovooo -= np.transpose(X.abb.oovooo, (0, 1, 2, 3, 5, 4))

        X.aab.vovooo = (
                0.5 * np.einsum("amef,efdijl->amdijl", H.aa.vovv, T.aab, optimize=True)
                + np.einsum("mdef,ebfijl->bmdjil", H.ab.ovvv, T.aab, optimize=True)
        )

        X.abb.vovooo = (
                0.5 * np.einsum("dmfe,aefikl->amdikl", H.bb.vovv, T.abb, optimize=True)
                + np.einsum("amef,ecfikl->amcilk", H.ab.vovv, T.abb, optimize=True)
        )

        X.aab.vvoooo = np.einsum("amef,ebfijl->abmijl", H.ab.vovv, T.aab, optimize=True)
        X.aab.vvoooo -= np.transpose(X.aab.vvoooo, (1, 0, 2, 3, 4, 5))

        X.abb.ovvooo = np.einsum("mdef,ecfikl->mcdikl", H.ab.ovvv, T.abb, optimize=True)
        X.abb.ovvooo -= np.transpose(X.abb.ovvooo, (0, 2, 1, 3, 4, 5))

        X.aaa.vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                + 0.25 * np.einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa, optimize=True)
                + 0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
        )
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (1, 0, 2, 3, 4, 5))
        X.aaa.vvooov -= np.transpose(X.aaa.vvooov, (0, 1, 2, 4, 3, 5))

        X.aab.vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                + 0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
                + 0.25 * np.einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa, optimize=True)
                + 0.25 * np.einsum("nmfe,abfijn->abmije", H.bb.oovv, T.aab, optimize=True)
        )
        X.aab.vvooov -= np.transpose(X.aab.vvooov, (1, 0, 2, 3, 4, 5))
        X.aab.vvooov -= np.transpose(X.aab.vvooov, (0, 1, 2, 4, 3, 5))

        X.abb.ovvvoo = (
                -0.5 * np.einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb, optimize=True)
                + 0.5 * np.einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb, optimize=True)
        )
        X.abb.ovvvoo -= np.transpose(X.abb.ovvvoo, (0, 2, 1, 3, 4, 5))
        X.abb.ovvvoo -= np.transpose(X.abb.ovvvoo, (0, 1, 2, 3, 5, 4))

        X.bbb.vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb, optimize=True)
                + 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb, optimize=True)
        )
        X.bbb.vvooov -= np.transpose(X.bbb.vvooov, (1, 0, 2, 3, 4, 5))
        X.bbb.vvooov -= np.transpose(X.bbb.vvooov, (0, 1, 2, 4, 3, 5))

        X.aab.vovovo = (
                -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                + np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab,
                                  optimize=True)  # !!! factor 1/2 to compensate asym
                + np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
                - np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                + np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
        )

        X.abb.vovovo = (
                -np.einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab, optimize=True)
                + np.einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab, optimize=True)
                - np.einsum("nmle,adin->amdiel", H.bb.ooov, T.ab, optimize=True)
                + np.einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,afdinl->amdiel", H.bb.oovv, T.abb, optimize=True)
        # !!! factor 1/2 to compensate asym
        )

        X.aab.vovoov = (
                -np.einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa, optimize=True)
                - 0.5 * np.einsum("mnfe,bfdjin->bmdjie", H.ab.oovv, T.aab, optimize=True)
        )
        X.aab.vovoov -= np.transpose(X.aab.vovoov, (0, 1, 2, 4, 3, 5))

        X.abb.ovvoov = (
                -0.5 * np.einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb, optimize=True)
                + np.einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab, optimize=True)
                - 0.5 * np.einsum("mnfe,fcdikn->mcdike", H.ab.oovv, T.abb, optimize=True)
        )
        X.abb.ovvoov -= np.transpose(X.abb.ovvoov, (0, 2, 1, 3, 4, 5))

        X.aab.vvovoo = (
                -0.5 * np.einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab, optimize=True)
        )
        X.aab.vvovoo -= np.transpose(X.aab.vvovoo, (1, 0, 2, 3, 4, 5))

        X.aab.vovvoo = (
                -np.einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab, optimize=True)
                + 0.5 * np.einsum("amef,fclk->amcelk", H.ab.vovv, T.bb, optimize=True)
        )
        X.aab.vovvoo -= np.transpose(X.aab.vovvoo, (0, 1, 2, 3, 5, 4))

    return X

def get_VT4_intermediates(T, H, system):

    X = Integral.from_empty(system, 3, data_type=T.a.dtype, use_none=True)

    # Get active-space slicing objects
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    ##################### aaa ###################

    ## vvoooo ##
    # ABmIJK
    X.aaa.vvoooo[Va, Va, oa, Oa, Oa, Oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,BAfenIJK->ABmIJK', H.aa.oovv[:, oa, va, va], T.aaaa.VVvvoOOO, optimize=True)
            - 1.0 * np.einsum('mneF,FBAenIJK->ABmIJK', H.aa.oovv[:, oa, va, Va], T.aaaa.VVVvoOOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEBAnIJK->ABmIJK', H.aa.oovv[:, oa, Va, Va], T.aaaa.VVVVoOOO, optimize=True)
            + 0.5 * np.einsum('mNef,BAfeIJKN->ABmIJK', H.aa.oovv[:, Oa, va, va], T.aaaa.VVvvOOOO, optimize=True)
            + 1.0 * np.einsum('mNeF,FBAeIJKN->ABmIJK', H.aa.oovv[:, Oa, va, Va], T.aaaa.VVVvOOOO, optimize=True)
            + 0.5 * np.einsum('mNEF,FEBAIJKN->ABmIJK', H.aa.oovv[:, Oa, Va, Va], T.aaaa.VVVVOOOO, optimize=True)
    )
    X.aaa.vvoooo[Va, Va, oa, Oa, Oa, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,BAefIJKn->ABmIJK', H.ab.oovv[:, ob, va, vb], T.aaab.VVvvOOOo, optimize=True)
            - 1.0 * np.einsum('mnEf,EBAfIJKn->ABmIJK', H.ab.oovv[:, ob, Va, vb], T.aaab.VVVvOOOo, optimize=True)
            - 1.0 * np.einsum('mneF,BAeFIJKn->ABmIJK', H.ab.oovv[:, ob, va, Vb], T.aaab.VVvVOOOo, optimize=True)
            - 1.0 * np.einsum('mnEF,EBAFIJKn->ABmIJK', H.ab.oovv[:, ob, Va, Vb], T.aaab.VVVVOOOo, optimize=True)
            - 1.0 * np.einsum('mNef,BAefIJKN->ABmIJK', H.ab.oovv[:, Ob, va, vb], T.aaab.VVvvOOOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EBAfIJKN->ABmIJK', H.ab.oovv[:, Ob, Va, vb], T.aaab.VVVvOOOO, optimize=True)
            - 1.0 * np.einsum('mNeF,BAeFIJKN->ABmIJK', H.ab.oovv[:, Ob, va, Vb], T.aaab.VVvVOOOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EBAFIJKN->ABmIJK', H.ab.oovv[:, Ob, Va, Vb], T.aaab.VVVVOOOO, optimize=True)
    )
    # AbmIJK
    X.aaa.vvoooo[Va, va, oa, Oa, Oa, Oa] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,AfebnIJK->AbmIJK', H.aa.oovv[:, oa, va, va], T.aaaa.VvvvoOOO, optimize=True)
            - 0.5 * np.einsum('mNef,AfebIJKN->AbmIJK', H.aa.oovv[:, Oa, va, va], T.aaaa.VvvvOOOO, optimize=True)
            - 1.0 * np.einsum('mneF,FAebnIJK->AbmIJK', H.aa.oovv[:, oa, va, Va], T.aaaa.VVvvoOOO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEAbnIJK->AbmIJK', H.aa.oovv[:, oa, Va, Va], T.aaaa.VVVvoOOO, optimize=True)
            + 1.0 * np.einsum('mNeF,FAebIJKN->AbmIJK', H.aa.oovv[:, Oa, va, Va], T.aaaa.VVvvOOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAbIJKN->AbmIJK', H.aa.oovv[:, Oa, Va, Va], T.aaaa.VVVvOOOO, optimize=True)
    )
    X.aaa.vvoooo[Va, va, oa, Oa, Oa, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,AebfIJKn->AbmIJK', H.ab.oovv[:, ob, va, vb], T.aaab.VvvvOOOo, optimize=True)
            + 1.0 * np.einsum('mnEf,EAbfIJKn->AbmIJK', H.ab.oovv[:, ob, Va, vb], T.aaab.VVvvOOOo, optimize=True)
            - 1.0 * np.einsum('mNef,AebfIJKN->AbmIJK', H.ab.oovv[:, Ob, va, vb], T.aaab.VvvvOOOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EAbfIJKN->AbmIJK', H.ab.oovv[:, Ob, Va, vb], T.aaab.VVvvOOOO, optimize=True)
            - 1.0 * np.einsum('mneF,AebFIJKn->AbmIJK', H.ab.oovv[:, ob, va, Vb], T.aaab.VvvVOOOo, optimize=True)
            + 1.0 * np.einsum('mnEF,EAbFIJKn->AbmIJK', H.ab.oovv[:, ob, Va, Vb], T.aaab.VVvVOOOo, optimize=True)
            - 1.0 * np.einsum('mNeF,AebFIJKN->AbmIJK', H.ab.oovv[:, Ob, va, Vb], T.aaab.VvvVOOOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EAbFIJKN->AbmIJK', H.ab.oovv[:, Ob, Va, Vb], T.aaab.VVvVOOOO, optimize=True)
    )
    # aBmIJK
    #X.aaa.vvoooo[va, Va, :, Oa, Oa, Oa] = -1.0 * np.transpose(X.aaa.vvoooo[Va, va, :, Oa, Oa, Oa], (1, 0, 2, 3, 4, 5))
    # abmIJK
    X.aaa.vvoooo[va, va, oa, Oa, Oa, Oa] += (1.0/1.0) * (
            -1.0*np.einsum('mneF,FebanIJK->abmIJK', H.aa.oovv[:, oa, va, Va], T.aaaa.VvvvoOOO, optimize=True)
            +1.0*np.einsum('mNeF,FebaIJKN->abmIJK', H.aa.oovv[:, Oa, va, Va], T.aaaa.VvvvOOOO, optimize=True)
            -0.5*np.einsum('mnEF,FEbanIJK->abmIJK', H.aa.oovv[:, oa, Va, Va], T.aaaa.VVvvoOOO, optimize=True)
            +0.5*np.einsum('mNEF,FEbaIJKN->abmIJK', H.aa.oovv[:, Oa, Va, Va], T.aaaa.VVvvOOOO, optimize=True)
    )
    X.aaa.vvoooo[va, va, oa, Oa, Oa, Oa] += (1.0/1.0) * (
            -1.0*np.einsum('mnEf,EbafIJKn->abmIJK', H.ab.oovv[:, ob, Va, vb], T.aaab.VvvvOOOo, optimize=True)
            -1.0*np.einsum('mNEf,EbafIJKN->abmIJK', H.ab.oovv[:, Ob, Va, vb], T.aaab.VvvvOOOO, optimize=True)
            -1.0*np.einsum('mneF,ebaFIJKn->abmIJK', H.ab.oovv[:, ob, va, Vb], T.aaab.vvvVOOOo, optimize=True)
            -1.0*np.einsum('mNeF,ebaFIJKN->abmIJK', H.ab.oovv[:, Ob, va, Vb], T.aaab.vvvVOOOO, optimize=True)
            -1.0*np.einsum('mnEF,EbaFIJKn->abmIJK', H.ab.oovv[:, ob, Va, Vb], T.aaab.VvvVOOOo, optimize=True)
            -1.0*np.einsum('mNEF,EbaFIJKN->abmIJK', H.ab.oovv[:, Ob, Va, Vb], T.aaab.VvvVOOOO, optimize=True)
    )
    # ABmiJK
    # AbmiJK
    # abmiJK

    # ABmijK
    # AbmijK
    # abmijK

    # ABmijk
    # Abmijk
    # abmijk


