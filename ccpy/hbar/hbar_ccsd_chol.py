import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.cholesky.cholesky_builders import build_2index_batch_vvvv_aa, build_2index_batch_vvvv_bb, build_3index_batch_vvvv_ab

def build_hbar_ccsd_chol(T, H0, RHF_symmetry, *args):

    # Reference to HBar is copied from reference H (not duplicated!)
    H = H0

    # One-body part of HBar
    H.a.ov = H0.a.ov.copy() + (
            ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
            + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
    else:
        H.b.ov = H0.b.ov.copy() + (
                ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
                + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
        )

    H.a.oo = H0.a.oo.copy() + (
            ccpy_einsum("je,ei->ji", H.a.ov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
            + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.b.oo = H.a.oo.copy()
    else:
        H.b.oo = H0.b.oo.copy() + (
                ccpy_einsum("je,ei->ji", H.b.ov, T.b)
                + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
                + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
                + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
        )
    H.a.vv = H0.a.vv.copy() + (
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa)  #
            - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab)  #
    )
    bt1 = (
            ccpy_einsum("xnf,fn->x", H.chol.a.ov, T.a)
            + ccpy_einsum("xnf,fn->x", H.chol.b.ov, T.b)
    )
    bxt1 = -ccpy_einsum("xne,fn->xfe", H.chol.a.ov, T.a)
    H.a.vv += (
            ccpy_einsum("xae,x->ae", H.chol.a.vv, bt1)
            + ccpy_einsum("xaf,xfe->ae", H.chol.a.vv, bxt1)
            - ccpy_einsum("me,am->ae", H.a.ov, T.a)
    )
    if RHF_symmetry:
        H.b.vv = H.a.vv.copy()
    else:
        H.b.vv = H0.b.vv.copy() + (
                - 0.5 * ccpy_einsum("mnef,afmn->ae", H.bb.oovv, T.bb)
                - ccpy_einsum("nmfe,fanm->ae", H.ab.oovv, T.ab)
                - ccpy_einsum("me,am->ae", H.b.ov, T.b)
        )
        bt1 = (
                ccpy_einsum("xnf,fn->x", H.chol.b.ov, T.b)
                + ccpy_einsum("xnf,fn->x", H.chol.a.ov, T.a)
        )
        bxt1 = -ccpy_einsum("xne,fn->xfe", H.chol.b.ov, T.b)
        H.b.vv += (
                ccpy_einsum("xae,x->ae", H.chol.b.vv, bt1)
                + ccpy_einsum("xaf,xfe->ae", H.chol.b.vv, bxt1)
        )

    # -------------------------------------------------------------------------
    ### T1-transformation of Cholesky vectors
    print("   Performing T1-transformation of Cholesky vectors")
    ### a ###
    H.chol.a.oo = H0.chol.a.oo.copy() + ccpy_einsum("xme,ei->xmi", H.chol.a.ov, T.a)
    H.chol.a.vv = H0.chol.a.vv.copy() - ccpy_einsum("xme,am->xae", H.chol.a.ov, T.a)
    H.chol.a.vo = (
            H.chol.a.vo.copy()
            - ccpy_einsum("xmi,am->xai", H.chol.a.oo, T.a)
            + ccpy_einsum("xae,ei->xai", H.chol.a.vv, T.a)
            + ccpy_einsum("xme,ei,am->xai", H.chol.a.ov, T.a, T.a)
    )
    if RHF_symmetry:
        H.chol.b.oo = H.chol.a.oo.copy()
        H.chol.b.vv = H.chol.a.vv.copy()
        H.chol.b.vo = H.chol.a.vo.copy()
    else:
        ### b ###
        H.chol.b.oo = H0.chol.b.oo.copy() + ccpy_einsum("xme,ei->xmi", H.chol.b.ov, T.b)
        H.chol.b.vv = H0.chol.b.vv.copy() - ccpy_einsum("xme,am->xae", H.chol.b.ov, T.b)
        H.chol.b.vo = (
                H.chol.b.vo.copy()
                - ccpy_einsum("xmi,am->xai", H.chol.b.oo, T.b)
                + ccpy_einsum("xae,ei->xai", H.chol.b.vv, T.b)
                + ccpy_einsum("xme,ei,am->xai", H.chol.b.ov, T.b, T.b)
        )
    # -------------------------------------------------------------------------

    # Cholesky-based intermediates folding some T2
    x_a_vo = (
            H.chol.a.vo.copy()
            + ccpy_einsum("xnf,afin->xai", H.chol.a.ov, T.aa)
            + ccpy_einsum("xnf,afin->xai", H.chol.b.ov, T.ab)
    )
    if RHF_symmetry:
        x_b_vo = x_a_vo.copy()
    else:
        x_b_vo = (
                H.chol.b.vo.copy()
                + ccpy_einsum("xnf,afin->xai", H.chol.b.ov, T.bb)
                + ccpy_einsum("xnf,fani->xai", H.chol.a.ov, T.ab)
        )

    # H.a.vv = H0.a.vv.copy() + (
    #         - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
    #         + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
    #         + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
    #         - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
    #         - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    # )
    # if RHF_symmetry:
    #     H.b.vv = H.a.vv.copy()
    # else:
    #     H.b.vv = H0.b.vv.copy() + (
    #             - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
    #             + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
    #             + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
    #             - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
    #             - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    #     )

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo)
            - ccpy_einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo)
            + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa)
    )
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo.copy()
    else:
        H.bb.oooo = (
                ccpy_einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo)
                - ccpy_einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo)
                + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb)
        )
    H.ab.oooo = (
            ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov
    H.aa.ooov = ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov)
    H.aa.ooov -= np.transpose(H.aa.ooov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov.copy()
    else:
        H.bb.ooov = ccpy_einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov)
        H.bb.ooov -= np.transpose(H.bb.ooov, (1, 0, 2, 3))
    H.ab.ooov = (
        ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov)
    )
    H.ab.oovo = (
        ccpy_einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    H.aa.vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.a.ov)
    H.aa.vovv -= np.transpose(H.aa.vovv, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vovv = H.aa.vovv.copy()
    else:
        H.bb.vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.b.vv, H.chol.b.ov)
        H.bb.vovv -= np.transpose(H.bb.vovv, (0, 1, 3, 2))
    H.ab.vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.b.ov)
    H.ab.ovvv = ccpy_einsum("xnf,xbe->nbfe", H.chol.a.ov, H.chol.b.vv)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    H.aa.vooo = (
            ccpy_einsum("xai,xmj->amij", x_a_vo, H.chol.a.oo)
            + 0.25 * ccpy_einsum("amef,efij->amij", H.aa.vovv, T.aa)
            + 0.5 * ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            #
            # This exchange term won't go away!!!
            #
            - ccpy_einsum("xnj,xmf,afin->amij", H.chol.a.oo, H.chol.a.ov, T.aa)
    )
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo.copy()
    else:
        H.bb.vooo = (
                ccpy_einsum("xai,xmj->amij", x_b_vo, H.chol.b.oo)
                + 0.25 * ccpy_einsum("amef,efij->amij", H.bb.vovv, T.bb)
                + 0.5 * ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
                #
                # This exchange term won't go away!!!
                #
                - ccpy_einsum("xnj,xmf,afin->amij", H.chol.b.oo, H.chol.b.ov, T.bb)
        )
        H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))
    H.ab.vooo = (
            ccpy_einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo)
            + ccpy_einsum("amef,efij->amij", H.ab.vovv, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
    )
    H.ab.ovoo = (
            ccpy_einsum("xmi,xbj->mbij", H.chol.a.oo, H.chol.b.vo)
            + ccpy_einsum("mnif,fbnj->mbij", H.aa.ooov, T.ab)
            + ccpy_einsum("mnif,fbnj->mbij", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfj,fbin->mbij", H.ab.oovo, T.ab)
            + ccpy_einsum("mbef,efij->mbij", H.ab.ovvv, T.ab)
            + ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov)
            - ccpy_einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo)
            + ccpy_einsum("mnef,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov.copy()
    else:
        H.bb.voov = (
                ccpy_einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov)
                - ccpy_einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo)
                + ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.bb)
                + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
        )
    H.ab.voov = (
            ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.ab)
    )
    H.ab.ovvo = (
            ccpy_einsum("xbj,xme->mbej", H.chol.b.vo, H.chol.a.ov)
            + ccpy_einsum("mnef,fbnj->mbej", H0.aa.oovv, T.ab)
            + ccpy_einsum("mnef,fbnj->mbej", H0.ab.oovv, T.bb)
    )
    H.ab.ovov = (
            ccpy_einsum("xmi,xbe->mbie", H.chol.a.oo, H.chol.b.vv)
            - ccpy_einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab)
    )
    H.ab.vovo = (
            ccpy_einsum("xae,xmj->amej", H.chol.a.vv, H.chol.b.oo)
            - ccpy_einsum("nmef,afnj->amej", H0.ab.oovv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    H.aa.vvov = (
            ccpy_einsum("xai,xbe->abie", x_a_vo, H.chol.a.vv)
            + 0.25 * ccpy_einsum("mnie,abmn->abie", H.aa.ooov, T.aa)
            - 0.5 * ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
            #
            # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
            #
            - ccpy_einsum("xbf,xne,afin->abie", H.chol.a.vv, H.chol.a.ov, T.aa)
    )
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov.copy()
    else:
        H.bb.vvov = (
                ccpy_einsum("xai,xbe->abie", x_b_vo, H.chol.b.vv)
                + 0.25 * ccpy_einsum("mnie,abmn->abie", H.bb.ooov, T.bb)
                - 0.5 * ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
                #
                # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
                #
                - ccpy_einsum("xbf,xne,afin->abie", H.chol.b.vv, H.chol.b.ov, T.bb)
        )
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))
    H.ab.vvov = (
            ccpy_einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv)
            + ccpy_einsum("nbfe,afin->abie", H.ab.ovvv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.ab)
            - ccpy_einsum("anfe,fbin->abie", H.ab.vovv, T.ab)
            + ccpy_einsum("mnie,abmn->abie", H.ab.ooov, T.ab)
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab)
    )
    H.ab.vvvo = (
            ccpy_einsum("xae,xbj->abej", H.chol.a.vv, H.chol.b.vo)
            + ccpy_einsum("anef,fbnj->abej", H.aa.vovv, T.ab)
            + ccpy_einsum("anef,fbnj->abej", H.ab.vovv, T.bb)
            - ccpy_einsum("mbef,afmj->abej", H.ab.ovvv, T.ab)
            + ccpy_einsum("mnej,abmn->abej", H.ab.oovo, T.ab)
            - ccpy_einsum("me,abmj->abej", H.a.ov, T.ab)
    )
    # -------------------------------------------------------------------------
    return H

def get_ccsd_intermediates(T, H, H0, RHF_symmetry):

    # Cholesky-based intermediates folding some T2
    x_a_vo = (
            H.chol.a.vo.copy()
            + ccpy_einsum("xnf,afin->xai", H.chol.a.ov, T.aa)
            + ccpy_einsum("xnf,afin->xai", H.chol.b.ov, T.ab)
    )
    if RHF_symmetry:
        x_b_vo = x_a_vo.copy()
    else:
        x_b_vo = (
                H.chol.b.vo.copy()
                + ccpy_einsum("xnf,afin->xai", H.chol.b.ov, T.bb)
                + ccpy_einsum("xnf,fani->xai", H.chol.a.ov, T.ab)
        )

    # One-body part of HBar
    H.a.ov = H0.a.ov.copy() + (
            ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
            + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
    else:
        H.b.ov = H0.b.ov.copy() + (
                ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
                + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
        )

    H.a.oo = H0.a.oo.copy() + (
            ccpy_einsum("je,ei->ji", H.a.ov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
            + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.b.oo = H.a.oo.copy()
    else:
        H.b.oo = H0.b.oo.copy() + (
                ccpy_einsum("je,ei->ji", H.b.ov, T.b)
                + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
                + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
                + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
        )
    H.a.vv = H0.a.vv.copy() + (
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H0.aa.oovv, T.aa)  #
            - ccpy_einsum("mnef,afmn->ae", H0.ab.oovv, T.ab)  #
    )
    bt1 = (
            ccpy_einsum("xnf,fn->x", H0.chol.a.ov, T.a)
            + ccpy_einsum("xnf,fn->x", H0.chol.b.ov, T.b)
    )
    bxt1 = -ccpy_einsum("xne,fn->xfe", H0.chol.a.ov, T.a)
    H.a.vv += (
            ccpy_einsum("xae,x->ae", H0.chol.a.vv, bt1)
            + ccpy_einsum("xaf,xfe->ae", H0.chol.a.vv, bxt1)
            - ccpy_einsum("me,am->ae", H.a.ov, T.a)
    )
    if RHF_symmetry:
        H.b.vv = H.a.vv.copy()
    else:
        H.b.vv = H0.b.vv.copy() + (
                - 0.5 * ccpy_einsum("mnef,afmn->ae", H0.bb.oovv, T.bb)
                - ccpy_einsum("nmfe,fanm->ae", H0.ab.oovv, T.ab)
                - ccpy_einsum("me,am->ae", H.b.ov, T.b)
        )
        bt1 = (
                ccpy_einsum("xnf,fn->x", H0.chol.b.ov, T.b)
                + ccpy_einsum("xnf,fn->x", H0.chol.a.ov, T.a)
        )
        bxt1 = -ccpy_einsum("xne,fn->xfe", H0.chol.b.ov, T.b)
        H.b.vv += (
                ccpy_einsum("xae,x->ae", H0.chol.b.vv, bt1)
                + ccpy_einsum("xaf,xfe->ae", H0.chol.b.vv, bxt1)
        )
    # H.a.ov = H0.a.ov.copy() + (
    #         ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
    #         + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    # )
    # if RHF_symmetry:
    #     H.b.ov = H.a.ov.copy()
    # else:
    #     H.b.ov = H0.b.ov.copy() + (
    #             ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
    #             + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
    #     )
    #
    #
    # H.a.oo = H0.a.oo.copy() + (
    #         ccpy_einsum("je,ei->ji", H.a.ov, T.a)
    #         + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
    #         + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
    #         + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
    #         + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    # )
    # if RHF_symmetry:
    #     H.b.oo = H.a.oo.copy()
    # else:
    #     H.b.oo = H0.b.oo.copy() + (
    #             ccpy_einsum("je,ei->ji", H.b.ov, T.b)
    #             + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
    #             + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
    #             + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
    #             + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
    #     )
    #
    # H.a.vv = H0.a.vv.copy() + (
    #         - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
    #         + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
    #         + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
    #         - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
    #         - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    # )
    # if RHF_symmetry:
    #     H.b.vv = H.a.vv.copy()
    # else:
    #     H.b.vv = H0.b.vv.copy() + (
    #             - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
    #             + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
    #             + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
    #             - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
    #             - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    #     )

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo)
            - ccpy_einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo)
            + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa)
    )
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo.copy()
    else:
        H.bb.oooo = (
                ccpy_einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo)
                - ccpy_einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo)
                + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb)
        )
    H.ab.oooo = (
            ccpy_einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov
    H.aa.ooov = ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov)
    H.aa.ooov -= np.transpose(H.aa.ooov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov.copy()
    else:
        H.bb.ooov = ccpy_einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov)
        H.bb.ooov -= np.transpose(H.bb.ooov, (1, 0, 2, 3))
    H.ab.ooov = (
        ccpy_einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov)
    )
    H.ab.oovo = (
        ccpy_einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    h2a_vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.a.ov)
    h2a_vovv -= np.transpose(h2a_vovv, (0, 1, 3, 2))
    if RHF_symmetry:
        h2c_vovv = h2a_vovv.copy()
    else:
        h2c_vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.b.vv, H.chol.b.ov)
        h2c_vovv -= np.transpose(h2c_vovv, (0, 1, 3, 2))
    h2b_vovv = ccpy_einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.b.ov)
    h2b_ovvv = ccpy_einsum("xnf,xbe->nbfe", H.chol.a.ov, H.chol.b.vv)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    H.aa.vooo = (
            ccpy_einsum("xai,xmj->amij", x_a_vo, H.chol.a.oo)
            + 0.25 * ccpy_einsum("amef,efij->amij", h2a_vovv, T.aa)
            + 0.5 * ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            #
            # This exchange term won't go away!!!
            #
            - ccpy_einsum("xnj,xmf,afin->amij", H.chol.a.oo, H.chol.a.ov, T.aa)
    )
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo.copy()
    else:
        H.bb.vooo = (
                ccpy_einsum("xai,xmj->amij", x_b_vo, H.chol.b.oo)
                + 0.25 * ccpy_einsum("amef,efij->amij", h2c_vovv, T.bb)
                + 0.5 * ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
                #
                # This exchange term won't go away!!!
                #
                - ccpy_einsum("xnj,xmf,afin->amij", H.chol.b.oo, H.chol.b.ov, T.bb)
        )
        H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))
    H.ab.vooo = (
            ccpy_einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo)
            + ccpy_einsum("amef,efij->amij", h2b_vovv, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
    )
    H.ab.ovoo = (
            ccpy_einsum("xmi,xbj->mbij", H.chol.a.oo, H.chol.b.vo)
            + ccpy_einsum("mnif,fbnj->mbij", H.aa.ooov, T.ab)
            + ccpy_einsum("mnif,fbnj->mbij", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfj,fbin->mbij", H.ab.oovo, T.ab)
            + ccpy_einsum("mbef,efij->mbij", h2b_ovvv, T.ab)
            + ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov)
            - ccpy_einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo)
            + ccpy_einsum("mnef,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov.copy()
    else:
        H.bb.voov = (
                ccpy_einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov)
                - ccpy_einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo)
                + ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.bb)
                + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
        )
    H.ab.voov = (
            ccpy_einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.ab)
    )
    H.ab.ovvo = (
            ccpy_einsum("xbj,xme->mbej", H.chol.b.vo, H.chol.a.ov)
            + ccpy_einsum("mnef,fbnj->mbej", H0.aa.oovv, T.ab)
            + ccpy_einsum("mnef,fbnj->mbej", H0.ab.oovv, T.bb)
    )
    H.ab.ovov = (
            ccpy_einsum("xmi,xbe->mbie", H.chol.a.oo, H.chol.b.vv)
            - ccpy_einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab)
    )
    H.ab.vovo = (
            ccpy_einsum("xae,xmj->amej", H.chol.a.vv, H.chol.b.oo)
            - ccpy_einsum("nmef,afnj->amej", H0.ab.oovv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    H.aa.vvov = (
            ccpy_einsum("xai,xbe->abie", x_a_vo, H.chol.a.vv)
            + 0.25 * ccpy_einsum("mnie,abmn->abie", H.aa.ooov, T.aa)
            - 0.5 * ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
            #
            # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
            #
            - ccpy_einsum("xbf,xne,afin->abie", H.chol.a.vv, H.chol.a.ov, T.aa)
    )
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov.copy()
    else:
        H.bb.vvov = (
                ccpy_einsum("xai,xbe->abie", x_b_vo, H.chol.b.vv)
                + 0.25 * ccpy_einsum("mnie,abmn->abie", H.bb.ooov, T.bb)
                - 0.5 * ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
                #
                # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
                #
                - ccpy_einsum("xbf,xne,afin->abie", H.chol.b.vv, H.chol.b.ov, T.bb)
        )
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))
    H.ab.vvov = (
            ccpy_einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv)
            + ccpy_einsum("nbfe,afin->abie", H.ab.ovvv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.ab)
            - ccpy_einsum("anfe,fbin->abie", H.ab.vovv, T.ab)
            + ccpy_einsum("mnie,abmn->abie", H.ab.ooov, T.ab)
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab)
    )
    H.ab.vvvo = (
            ccpy_einsum("xae,xbj->abej", H.chol.a.vv, H.chol.b.vo)
            + ccpy_einsum("anef,fbnj->abej", H.aa.vovv, T.ab)
            + ccpy_einsum("anef,fbnj->abej", H.ab.vovv, T.bb)
            - ccpy_einsum("mbef,afmj->abej", H.ab.ovvv, T.ab)
            + ccpy_einsum("mnej,abmn->abej", H.ab.oovo, T.ab)
            - ccpy_einsum("me,abmj->abej", H.a.ov, T.ab)
    )
    # -------------------------------------------------------------------------

    # # -------------------------------------------------------------------------
    # ### TYPE: VVVV
    # H.aa.vvvv = (
    #         ccpy_einsum("xae,xbf->abef", H.chol.a.vv, H.chol.a.vv)
    #         - ccpy_einsum("xaf,xbe->abef", H.chol.a.vv, H.chol.a.vv)
    #         + 0.5 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, T.aa)
    # )
    # H.ab.vvvv = (
    #         ccpy_einsum("xae,xbf->abef", H.chol.a.vv, H.chol.b.vv)
    #         + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, T.ab)
    # )
    # if RHF_symmetry:
    #     H.bb.vvvv = H.aa.vvvv.copy()
    # else:
    #     H.bb.vvvv = (
    #             ccpy_einsum("xae,xbf->abef", H.chol.b.vv, H.chol.b.vv)
    #             - ccpy_einsum("xaf,xbe->abef", H.chol.b.vv, H.chol.b.vv)
    #             + 0.5 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, T.bb)
    #     )
    # # -------------------------------------------------------------------------
    return H

def build_hbar_ccsd_chol_bak(T, H0, RHF_symmetry, *args):

    # Reference to HBar is copied from reference H (not duplicated!) 
    H = H0

    # Orbital dimensions
    nua, nub, noa, nob = T.ab.shape

    H.a.ov += (
                ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
                + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )

    H.a.oo += (
                ccpy_einsum("je,ei->ji", H.a.ov, T.a)
                + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
                + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
                + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )

    H.a.vv += (
                - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
                + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
                + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
                - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
                - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    )

    H.b.ov += (
                ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
                + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
    )

    H.b.oo += (
                ccpy_einsum("je,ei->ji", H.b.ov, T.b)
                + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
                + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
                + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
    )

    H.b.vv += (
                - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
                + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
                + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
                - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
                - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    )

    # -------------------------------------------------------------------------
    # Make useful intermediates
    tau_aa = 0.5 * T.aa + ccpy_einsum("ai,bj->abij", T.a, T.a)
    tau_aa -= np.transpose(tau_aa, (0, 1, 3, 2))
    if RHF_symmetry:
        tau_bb = tau_aa
    else:
        tau_bb = 0.5 * T.bb + ccpy_einsum("ai,bj->abij", T.b, T.b)
        tau_bb -= np.transpose(tau_bb, (0, 1, 3, 2))
    tau_ab = T.ab + ccpy_einsum("ai,bj->abij", T.a, T.b)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            0.5 * H0.aa.oooo
            + ccpy_einsum("nmje,ei->mnij", H0.aa.ooov, T.a)
            + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, tau_aa)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo
    else:
        H.bb.oooo = (
                0.5 * H0.bb.oooo
                + ccpy_einsum("nmje,ei->mnij", H0.bb.ooov, T.b)
                + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, tau_bb)
        )
        H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))
    H.ab.oooo = (
            H0.ab.oooo
            + ccpy_einsum("mnej,ei->mnij", H0.ab.oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", H0.ab.ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, tau_ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov 
    H.aa.ooov += ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov
    else:
        H.bb.ooov += ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)
    H.ab.ooov += ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)
    H.ab.oovo += ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo = H0.aa.vooo + Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo
    else:
        Q1 = (
                ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
                + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
        )
        Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
        Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
        Q1 += Q2
        Q1 -= np.transpose(Q1, (0, 1, 3, 2))
        H.bb.vooo = H0.bb.vooo + Q1 + (
                + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
                - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
                + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
        )
    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo = H0.ab.vooo + (
            ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
            - ccpy_einsum("nmij,an->amij", H.ab.oooo, T.a)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
            + ccpy_einsum("amef,efij->amij", H0.ab.vovv, T.ab)
    )
    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    H.ab.ovoo = H0.ab.ovoo + (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            H0.aa.voov
            + ccpy_einsum("amfe,fi->amie", H0.aa.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov
    else:
        H.bb.voov = (
                H0.bb.voov
                + ccpy_einsum("amfe,fi->amie", H0.bb.vovv, T.b)
                - ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)
                + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
                + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
        )
    H.ab.voov = (
            H0.ab.voov
            + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )
    H.ab.ovvo = (
            H0.ab.ovvo
            + ccpy_einsum("maef,fi->maei", H0.ab.ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", H.ab.oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )
    H.ab.ovov = (
            H0.ab.ovov
            + ccpy_einsum("mafe,fi->maie", H0.ab.ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", H.ab.ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )
    H.ab.vovo = (
            H0.ab.vovo
            - ccpy_einsum("nmei,an->amei", H.ab.oovo, T.a)
            + ccpy_einsum("amef,fi->amei", H0.ab.vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    x2a_voov = H.aa.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a) # defined to avoid double-counting from A(ab) on [8] and [15]
    H.aa.vvov = (
            0.5 * H0.aa.vvov # [1]
            - 0.5 * ccpy_einsum("me,abim->abie", H.a.ov, T.aa) # [4]+[12]+[13]
            - ccpy_einsum("amie,bm->abie", x2a_voov, T.a) # [3]+[8']+[9]+[11]+[13]+[15']
            + 0.25 * ccpy_einsum("mnie,abmn->abie", H.aa.ooov, T.aa) # [6]+[10]
            # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
            + ccpy_einsum("bnef,afin->abie", H0.aa.vovv, T.aa) # [5]
            + ccpy_einsum("bnef,afin->abie", H0.ab.vovv, T.ab) # [7]
    )
    # Add on h0(vvvv) term using Cholesky
    for a in range(nua):
        for b in range(a + 1, nua):
            batch_ints = build_2index_batch_vvvv_aa(a, b, H0)
            H.aa.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.a)
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov
    else:
        x2c_voov = H.bb.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)  # defined to avoid double-counting from A(ab) on [8] and [15]
        H.bb.vvov = (
                0.5 * H0.bb.vvov  # [1]
                - 0.5 * ccpy_einsum("me,abim->abie", H.b.ov, T.bb)  # [4]+[12]+[13]
                - ccpy_einsum("amie,bm->abie", x2c_voov, T.b)  # [3]+[8']+[9]+[11]+[13]+[15']
                + 0.25 * ccpy_einsum("mnie,abmn->abie", H.bb.ooov, T.bb)  # [6]+[10]
                # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
                + ccpy_einsum("bnef,afin->abie", H0.bb.vovv, T.bb)  # [5]
                + ccpy_einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab)  # [7]
        )
        # Add on h0(vvvv) term using Cholesky
        for a in range(nub):
            for b in range(a + 1, nub):
                batch_ints = build_2index_batch_vvvv_bb(a, b, H0)
                H.bb.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.b)
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    # need to define x2b_voov and x2b_ovov such that [10] and [18] are not double counted
    x2b_voov = H.ab.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a) # nu2no3
    x2b_ovov = H.ab.ovov + 0.5 * ccpy_einsum("nmie,bm->nbie", H.ab.ooov, T.b) # nu2no3
    H.ab.vvov = (
            H0.ab.vvov # [1]
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("amie,bm->abie", x2b_voov, T.b) # [4] + 1/2*[10] + [11] + [12] + [17] + 1/2*[18]
            - ccpy_einsum("mbie,am->abie", x2b_ovov, T.a) # [2] + [9] + 1/2*[10] + [15] + 1/2*[18]
            + ccpy_einsum("nmie,abnm->abie", H.ab.ooov, T.ab) # [7] + [14]
            # Terms we have to deal with directly that are nu^4: [2], [5], [6], and [19]
            + ccpy_einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa) # [5]
            + ccpy_einsum("bmef,afim->abie", H0.bb.vovv, T.ab) # [6]
            - ccpy_einsum("amfe,fbim->abie", H0.ab.vovv, T.ab) # [19]
    )
    x2b_ovvo = H.ab.ovvo + 0.5 * ccpy_einsum("nmei,am->naei", H.ab.oovo, T.b)
    x2b_vovo = H.ab.vovo + 0.5 * ccpy_einsum("nmei,bn->bmei", H.ab.oovo, T.a)
    H.ab.vvvo = (
            H0.ab.vvvo # [1]
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("maei,bm->baei", x2b_ovvo, T.a) # [4] + 1/2*[10] + [11] + [12] + [14] + 1/2*[17]
            - ccpy_einsum("bmei,am->baei", x2b_vovo, T.b) # [3] + [9] + 1/2*[10] + 1/2*[17] + [18]
            + ccpy_einsum("nmei,banm->baei", H.ab.oovo, T.ab) # [6] + [15]
            # Terms we have to deal with directly that are nu^4: [2], [5], [7], and [19]
            + ccpy_einsum("bmef,afim->baei", H0.ab.vovv, T.bb) # [5]
            + ccpy_einsum("bmef,fami->baei", H0.aa.vovv, T.ab) # [7]
            - ccpy_einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab) # [19]
    )
    # add on h0(vvvv) term using Cholesky
    for a in range(nua):
        batch_ints = build_3index_batch_vvvv_ab(a, H0)
        H.ab.vvov[a, :, :, :] += ccpy_einsum("bfe,fi->bie", batch_ints, T.a)
        H.ab.vvvo[a, :, :, :] += ccpy_einsum("bef,fi->bei", batch_ints, T.b)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVVV
    ### NEEDS: H0.vovv
    #H.aa.vvvv = (
    #        0.5 * H0.aa.vvvv
    #        + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa)
    #        - ccpy_einsum("amef,bm->abef", H0.aa.vovv, T.a)
    #)
    #H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    #if RHF_symmetry:
    #    H.bb.vvvv = H.aa.vvvv
    #else:
    #    H.bb.vvvv = (
    #            0.5 * H0.bb.vvvv
    #            + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb)
    #            - ccpy_einsum("amef,bm->abef", H0.bb.vovv, T.b)
    #    )
    #    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    #H.ab.vvvv = (
    #        H0.ab.vvvv
    #        - ccpy_einsum("mbef,am->abef", H0.ab.ovvv, T.a)
    #        - ccpy_einsum("amef,bm->abef", H0.ab.vovv, T.b)
    #        + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab)
    #)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    H.aa.vovv -= ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)
    if RHF_symmetry:
        H.bb.vovv = H.aa.vovv
    else:
        H.bb.vovv -= ccpy_einsum("mnfe,an->amef", H0.bb.oovv, T.b)
    H.ab.vovv -= ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a) 
    H.ab.ovvv -= ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### T1-transformation of Cholesky vectors
    print("   Performing T1-transformation of Cholesky vectors")
    ### a ###
    H.chol.a.oo = H0.chol.a.oo.copy() + ccpy_einsum("xme,ei->xmi", H.chol.a.ov, T.a)
    H.chol.a.vv = H0.chol.a.vv.copy() - ccpy_einsum("xme,am->xae", H.chol.a.ov, T.a)
    H.chol.a.vo = (
            H.chol.a.vo.copy()
            - ccpy_einsum("xmi,am->xai", H.chol.a.oo, T.a)
            + ccpy_einsum("xae,ei->xai", H.chol.a.vv, T.a)
            + ccpy_einsum("xme,ei,am->xai", H.chol.a.ov, T.a, T.a)
    )
    ### b ###
    H.chol.b.oo = H0.chol.b.oo.copy() + ccpy_einsum("xme,ei->xmi", H.chol.b.ov, T.b)
    H.chol.b.vv = H0.chol.b.vv.copy() - ccpy_einsum("xme,am->xae", H.chol.b.ov, T.b)
    H.chol.b.vo = (
            H.chol.b.vo.copy()
            - ccpy_einsum("xmi,am->xai", H.chol.b.oo, T.b)
            + ccpy_einsum("xae,ei->xai", H.chol.b.vv, T.b)
            + ccpy_einsum("xme,ei,am->xai", H.chol.b.ov, T.b, T.b)
    )
    # -------------------------------------------------------------------------
    return H

def build_hbar_ccsd_chol_debug(T, H0, RHF_symmetry, *args):
    """Calculate the CCSD similarity-transformed Hamiltonian (H_N e^(T1+T2))_C.
    Copied as-is from original CCpy implementation."""
    from copy import deepcopy
    #from ccpy.models.integrals import Integral

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)
    #H = Integral.from_empty(system, 2, use_none=True)

    # Orbital dimensions
    nua, nub, noa, nob = T.ab.shape

    # Make useful intermediates
    tau_aa = 0.5 * T.aa + ccpy_einsum("ai,bj->abij", T.a, T.a)
    tau_aa -= np.transpose(tau_aa, (0, 1, 3, 2))
    if RHF_symmetry:
        tau_bb = tau_aa
    else:
        tau_bb = 0.5 * T.bb + ccpy_einsum("ai,bj->abij", T.b, T.b)
        tau_bb -= np.transpose(tau_bb, (0, 1, 3, 2))
    tau_ab = T.ab + ccpy_einsum("ai,bj->abij", T.a, T.b)

    H.a.ov += (
                ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
                + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )

    H.a.oo += (
                ccpy_einsum("je,ei->ji", H.a.ov, T.a)
                + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
                + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
                + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )

    H.a.vv += (
                - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
                + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
                + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
                - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
                - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    )

    H.b.ov += (
                ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
                + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
    )

    H.b.oo += (
                ccpy_einsum("je,ei->ji", H.b.ov, T.b)
                + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
                + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
                + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
                + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
    )

    H.b.vv += (
                - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
                + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
                + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
                - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
                - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    )
    
    Q1 = -ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.bb.oovv, T.b)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    #x1 = time.time()
    # H.aa.vvvv = (
    #        0.5 * H0.aa.vvvv
    #        + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa)
    #        - ccpy_einsum("amef,bm->abef", H0.aa.vovv, T.a)
    # )
    # H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    # if RHF_symmetry:
    #    H.bb.vvvv = H.aa.vvvv
    # else:
    #    H.bb.vvvv = (
    #            0.5 * H0.bb.vvvv
    #            + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb)
    #            - ccpy_einsum("amef,bm->abef", H0.bb.vovv, T.b)
    #    )
    #    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    # H.ab.vvvv = (
    #        H0.ab.vvvv
    #        - ccpy_einsum("mbef,am->abef", H0.ab.ovvv, T.a)
    #        - ccpy_einsum("amef,bm->abef", H0.ab.vovv, T.b)
    #        + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab)
    # )
    #x2 = time.time()
    #print("vvvv:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.oooo = (
            0.5 * H0.aa.oooo
            + ccpy_einsum("nmje,ei->mnij", H0.aa.ooov, T.a)
            + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, tau_aa)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo
    else:
        H.bb.oooo = (
                0.5 * H0.bb.oooo
                + ccpy_einsum("nmje,ei->mnij", H0.bb.ooov, T.b)
                + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, tau_bb)
        )
        H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))
    H.ab.oooo = (
            H0.ab.oooo
            + ccpy_einsum("mnej,ei->mnij", H0.ab.oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", H0.ab.ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, tau_ab)
    )
    #x2 = time.perf_counter()
    #print("oooo:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.voov = (
            H0.aa.voov
            + ccpy_einsum("amfe,fi->amie", H0.aa.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov
    else:
        H.bb.voov = (
                H0.bb.voov
                + ccpy_einsum("amfe,fi->amie", H0.bb.vovv, T.b)
                - ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)
                + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
                + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
        )
    H.ab.voov = (
            H0.ab.voov
            + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )
    H.ab.ovvo = (
            H0.ab.ovvo
            + ccpy_einsum("maef,fi->maei", H0.ab.ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", H.ab.oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )
    H.ab.ovov = (
            H0.ab.ovov
            + ccpy_einsum("mafe,fi->maie", H0.ab.ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", H.ab.ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )
    H.ab.vovo = (
            H0.ab.vovo
            - ccpy_einsum("nmei,an->amei", H.ab.oovo, T.a)
            + ccpy_einsum("amef,fi->amei", H0.ab.vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )
    #x2 = time.perf_counter()
    #print("voov:", x2 - x1)

    #x1 = time.perf_counter()
    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo = H0.aa.vooo + Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo
    else:
        Q1 = (
                ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
                + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
        )
        Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
        Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
        Q1 += Q2
        Q1 -= np.transpose(Q1, (0, 1, 3, 2))
        H.bb.vooo = H0.bb.vooo + Q1 + (
                + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
                - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
                + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
        )
    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo = H0.ab.vooo + (
            ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
            - ccpy_einsum("nmij,an->amij", H.ab.oooo, T.a)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
            + ccpy_einsum("amef,efij->amij", H0.ab.vovv, T.ab)
    )
    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    H.ab.ovoo = H0.ab.ovoo + (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )
    #x2 = time.perf_counter()
    #print("vooo:", x2 - x1)

    #x1 = time.perf_counter()
    x2a_voov = H.aa.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a) # defined to avoid double-counting from A(ab) on [8] and [15]
    H.aa.vvov = (
            0.5 * H0.aa.vvov # [1]
            - 0.5 * ccpy_einsum("me,abim->abie", H.a.ov, T.aa) # [4]+[12]+[13]
            - ccpy_einsum("amie,bm->abie", x2a_voov, T.a) # [3]+[8']+[9]+[11]+[13]+[15']
            + 0.25 * ccpy_einsum("mnie,abmn->abie", H.aa.ooov, T.aa) # [6]+[10]
            # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
            #+ 0.5 * ccpy_einsum("abfe,fi->abie", H0.aa.vvvv, T.a) # [2]
            + ccpy_einsum("bnef,afin->abie", H0.aa.vovv, T.aa) # [5]
            + ccpy_einsum("bnef,afin->abie", H0.ab.vovv, T.ab) # [7]
    )
    # Add on h0(vvvv) term using Cholesky
    for a in range(nua):
        for b in range(a + 1, nua):
            batch_ints = build_2index_batch_vvvv_aa(a, b, H0)
            H.aa.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.a)
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov
    else:
        x2c_voov = H.bb.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)  # defined to avoid double-counting from A(ab) on [8] and [15]
        H.bb.vvov = (
                0.5 * H0.bb.vvov  # [1]
                - 0.5 * ccpy_einsum("me,abim->abie", H.b.ov, T.bb)  # [4]+[12]+[13]
                - ccpy_einsum("amie,bm->abie", x2c_voov, T.b)  # [3]+[8']+[9]+[11]+[13]+[15']
                + 0.25 * ccpy_einsum("mnie,abmn->abie", H.bb.ooov, T.bb)  # [6]+[10]
                # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
                #+ 0.5 * ccpy_einsum("abfe,fi->abie", H0.bb.vvvv, T.b)  # [2]
                + ccpy_einsum("bnef,afin->abie", H0.bb.vovv, T.bb)  # [5]
                + ccpy_einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab)  # [7]
        )
        # Add on h0(vvvv) term using Cholesky
        for a in range(nub):
            for b in range(a + 1, nub):
                batch_ints = build_2index_batch_vvvv_bb(a, b, H0)
                H.bb.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.b)
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    # need to define x2b_voov and x2b_ovov such that [10] and [18] are not double counted
    x2b_voov = H.ab.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a) # nu2no3
    x2b_ovov = H.ab.ovov + 0.5 * ccpy_einsum("nmie,bm->nbie", H.ab.ooov, T.b) # nu2no3
    H.ab.vvov = (
            H0.ab.vvov # [1]
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("amie,bm->abie", x2b_voov, T.b) # [4] + 1/2*[10] + [11] + [12] + [17] + 1/2*[18]
            - ccpy_einsum("mbie,am->abie", x2b_ovov, T.a) # [2] + [9] + 1/2*[10] + [15] + 1/2*[18]
            + ccpy_einsum("nmie,abnm->abie", H.ab.ooov, T.ab) # [7] + [14]
            # Terms we have to deal with directly that are nu^4: [2], [5], [6], and [19]
            #+ ccpy_einsum("abfe,fi->abie", H0.ab.vvvv, T.a) # [2]
            + ccpy_einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa) # [5]
            + ccpy_einsum("bmef,afim->abie", H0.bb.vovv, T.ab) # [6]
            - ccpy_einsum("amfe,fbim->abie", H0.ab.vovv, T.ab) # [19]
    )
    x2b_ovvo = H.ab.ovvo + 0.5 * ccpy_einsum("nmei,am->naei", H.ab.oovo, T.b)
    x2b_vovo = H.ab.vovo + 0.5 * ccpy_einsum("nmei,bn->bmei", H.ab.oovo, T.a)
    H.ab.vvvo = (
            H0.ab.vvvo # [1]
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("maei,bm->baei", x2b_ovvo, T.a) # [4] + 1/2*[10] + [11] + [12] + [14] + 1/2*[17]
            - ccpy_einsum("bmei,am->baei", x2b_vovo, T.b) # [3] + [9] + 1/2*[10] + 1/2*[17] + [18]
            + ccpy_einsum("nmei,banm->baei", H.ab.oovo, T.ab) # [6] + [15]
            # Terms we have to deal with directly that are nu^4: [2], [5], [7], and [19]
            #+ ccpy_einsum("baef,fi->baei", H0.ab.vvvv, T.b) # [2]
            + ccpy_einsum("bmef,afim->baei", H0.ab.vovv, T.bb) # [5]
            + ccpy_einsum("bmef,fami->baei", H0.aa.vovv, T.ab) # [7]
            - ccpy_einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab) # [19]
    )
    #x2 = time.perf_counter()
    #print("vvov:", x2 - x1)
    # add on h0(vvvv) term using Cholesky
    for a in range(nua):
        batch_ints = build_3index_batch_vvvv_ab(a, H0)
        H.ab.vvov[a, :, :, :] += ccpy_einsum("bfe,fi->bie", batch_ints, T.a)
        H.ab.vvvo[a, :, :, :] += ccpy_einsum("bef,fi->bei", batch_ints, T.b)
    return H

