import numpy as np
from ccpy.cholesky.cholesky_builders import build_2index_batch_vvvv_aa, build_2index_batch_vvvv_bb, build_3index_batch_vvvv_ab

def build_hbar_ccsd_chol(T, H0, RHF_symmetry, *args):

    # Reference to HBar is copied from reference H (not duplicated!)
    H = H0

    # One-body part of HBar
    H.a.ov = H0.a.ov.copy() + (
            np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
            + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
    else:
        H.b.ov = H0.b.ov.copy() + (
                np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
                + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
        )

    H.a.oo = H0.a.oo.copy() + (
            np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.b.oo = H.a.oo.copy()
    else:
        H.b.oo = H0.b.oo.copy() + (
                np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
                + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
                + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
        )
    H.a.vv = H0.a.vv.copy() + (
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)  #
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)  #
    )
    bt1 = (
            np.einsum("xnf,fn->x", H.chol.a.ov, T.a, optimize=True)
            + np.einsum("xnf,fn->x", H.chol.b.ov, T.b, optimize=True)
    )
    bxt1 = -np.einsum("xne,fn->xfe", H.chol.a.ov, T.a, optimize=True)
    H.a.vv += (
            np.einsum("xae,x->ae", H.chol.a.vv, bt1, optimize=True)
            + np.einsum("xaf,xfe->ae", H.chol.a.vv, bxt1, optimize=True)
            - np.einsum("me,am->ae", H.a.ov, T.a, optimize=True)
    )
    if RHF_symmetry:
        H.b.vv = H.a.vv.copy()
    else:
        H.b.vv = H0.b.vv.copy() + (
                - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
                - np.einsum("me,am->ae", H.b.ov, T.b, optimize=True)
        )
        bt1 = (
                np.einsum("xnf,fn->x", H.chol.b.ov, T.b, optimize=True)
                + np.einsum("xnf,fn->x", H.chol.a.ov, T.a, optimize=True)
        )
        bxt1 = -np.einsum("xne,fn->xfe", H.chol.b.ov, T.b, optimize=True)
        H.b.vv += (
                np.einsum("xae,x->ae", H.chol.b.vv, bt1, optimize=True)
                + np.einsum("xaf,xfe->ae", H.chol.b.vv, bxt1, optimize=True)
        )

    # -------------------------------------------------------------------------
    ### T1-transformation of Cholesky vectors
    print("   Performing T1-transformation of Cholesky vectors")
    ### a ###
    H.chol.a.oo = H0.chol.a.oo.copy() + np.einsum("xme,ei->xmi", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vv = H0.chol.a.vv.copy() - np.einsum("xme,am->xae", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vo = (
            H.chol.a.vo.copy()
            - np.einsum("xmi,am->xai", H.chol.a.oo, T.a, optimize=True)
            + np.einsum("xae,ei->xai", H.chol.a.vv, T.a, optimize=True)
            + np.einsum("xme,ei,am->xai", H.chol.a.ov, T.a, T.a, optimize=True)
    )
    if RHF_symmetry:
        H.chol.b.oo = H.chol.a.oo.copy()
        H.chol.b.vv = H.chol.a.vv.copy()
        H.chol.b.vo = H.chol.a.vo.copy()
    else:
        ### b ###
        H.chol.b.oo = H0.chol.b.oo.copy() + np.einsum("xme,ei->xmi", H.chol.b.ov, T.b, optimize=True)
        H.chol.b.vv = H0.chol.b.vv.copy() - np.einsum("xme,am->xae", H.chol.b.ov, T.b, optimize=True)
        H.chol.b.vo = (
                H.chol.b.vo.copy()
                - np.einsum("xmi,am->xai", H.chol.b.oo, T.b, optimize=True)
                + np.einsum("xae,ei->xai", H.chol.b.vv, T.b, optimize=True)
                + np.einsum("xme,ei,am->xai", H.chol.b.ov, T.b, T.b, optimize=True)
        )
    # -------------------------------------------------------------------------

    # Cholesky-based intermediates folding some T2
    x_a_vo = (
            H.chol.a.vo.copy()
            + np.einsum("xnf,afin->xai", H.chol.a.ov, T.aa)
            + np.einsum("xnf,afin->xai", H.chol.b.ov, T.ab)
    )
    if RHF_symmetry:
        x_b_vo = x_a_vo.copy()
    else:
        x_b_vo = (
                H.chol.b.vo.copy()
                + np.einsum("xnf,afin->xai", H.chol.b.ov, T.bb)
                + np.einsum("xnf,fani->xai", H.chol.a.ov, T.ab)
        )

    # H.a.vv = H0.a.vv.copy() + (
    #         - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
    #         + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
    #         + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
    #         - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
    #         - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    # )
    # if RHF_symmetry:
    #     H.b.vv = H.a.vv.copy()
    # else:
    #     H.b.vv = H0.b.vv.copy() + (
    #             - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
    #             + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
    #             + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
    #             - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
    #             - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    #     )

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
            - np.einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
            + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    )
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo.copy()
    else:
        H.bb.oooo = (
                np.einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
                - np.einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
                + 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True)
        )
    H.ab.oooo = (
            np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov
    H.aa.ooov = np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov, optimize=True)
    H.aa.ooov -= np.transpose(H.aa.ooov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov.copy()
    else:
        H.bb.ooov = np.einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov, optimize=True)
        H.bb.ooov -= np.transpose(H.bb.ooov, (1, 0, 2, 3))
    H.ab.ooov = (
        np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov, optimize=True)
    )
    H.ab.oovo = (
        np.einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    H.aa.vovv = np.einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.a.ov, optimize=True)
    H.aa.vovv -= np.transpose(H.aa.vovv, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vovv = H.aa.vovv.copy()
    else:
        H.bb.vovv = np.einsum("xbe,xnf->bnef", H.chol.b.vv, H.chol.b.ov, optimize=True)
        H.bb.vovv -= np.transpose(H.bb.vovv, (0, 1, 3, 2))
    H.ab.vovv = np.einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.b.ov, optimize=True)
    H.ab.ovvv = np.einsum("xnf,xbe->nbfe", H.chol.a.ov, H.chol.b.vv, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    H.aa.vooo = (
            np.einsum("xai,xmj->amij", x_a_vo, H.chol.a.oo, optimize=True)
            + 0.25 * np.einsum("amef,efij->amij", H.aa.vovv, T.aa, optimize=True)
            + 0.5 * np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            #
            # This exchange term won't go away!!!
            #
            - np.einsum("xnj,xmf,afin->amij", H.chol.a.oo, H.chol.a.ov, T.aa, optimize=True)
    )
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo.copy()
    else:
        H.bb.vooo = (
                np.einsum("xai,xmj->amij", x_b_vo, H.chol.b.oo, optimize=True)
                + 0.25 * np.einsum("amef,efij->amij", H.bb.vovv, T.bb, optimize=True)
                + 0.5 * np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
                #
                # This exchange term won't go away!!!
                #
                - np.einsum("xnj,xmf,afin->amij", H.chol.b.oo, H.chol.b.ov, T.bb, optimize=True)
        )
        H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))
    H.ab.vooo = (
            np.einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo, optimize=True)
            + np.einsum("amef,efij->amij", H.ab.vovv, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
    )
    H.ab.ovoo = (
            np.einsum("xmi,xbj->mbij", H.chol.a.oo, H.chol.b.vo, optimize=True)
            + np.einsum("mnif,fbnj->mbij", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnif,fbnj->mbij", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfj,fbin->mbij", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("mbef,efij->mbij", H.ab.ovvv, T.ab, optimize=True)
            + np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov, optimize=True)
            - np.einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo, optimize=True)
            + np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov.copy()
    else:
        H.bb.voov = (
                np.einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov, optimize=True)
                - np.einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo, optimize=True)
                + np.einsum("mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
        )
    H.ab.voov = (
            np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )
    H.ab.ovvo = (
            np.einsum("xbj,xme->mbej", H.chol.b.vo, H.chol.a.ov, optimize=True)
            + np.einsum("mnef,fbnj->mbej", H0.aa.oovv, T.ab, optimize=True)
            + np.einsum("mnef,fbnj->mbej", H0.ab.oovv, T.bb, optimize=True)
    )
    H.ab.ovov = (
            np.einsum("xmi,xbe->mbie", H.chol.a.oo, H.chol.b.vv, optimize=True)
            - np.einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab, optimize=True)
    )
    H.ab.vovo = (
            np.einsum("xae,xmj->amej", H.chol.a.vv, H.chol.b.oo, optimize=True)
            - np.einsum("nmef,afnj->amej", H0.ab.oovv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    H.aa.vvov = (
            np.einsum("xai,xbe->abie", x_a_vo, H.chol.a.vv, optimize=True)
            + 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
            - 0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
            #
            # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
            #
            - np.einsum("xbf,xne,afin->abie", H.chol.a.vv, H.chol.a.ov, T.aa, optimize=True)
    )
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov.copy()
    else:
        H.bb.vvov = (
                np.einsum("xai,xbe->abie", x_b_vo, H.chol.b.vv, optimize=True)
                + 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
                - 0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
                #
                # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
                #
                - np.einsum("xbf,xne,afin->abie", H.chol.b.vv, H.chol.b.ov, T.bb, optimize=True)
        )
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))
    H.ab.vvov = (
            np.einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv, optimize=True)
            + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
            - np.einsum("anfe,fbin->abie", H.ab.vovv, T.ab, optimize=True)
            + np.einsum("mnie,abmn->abie", H.ab.ooov, T.ab, optimize=True)
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
    )
    H.ab.vvvo = (
            np.einsum("xae,xbj->abej", H.chol.a.vv, H.chol.b.vo, optimize=True)
            + np.einsum("anef,fbnj->abej", H.aa.vovv, T.ab, optimize=True)
            + np.einsum("anef,fbnj->abej", H.ab.vovv, T.bb, optimize=True)
            - np.einsum("mbef,afmj->abej", H.ab.ovvv, T.ab, optimize=True)
            + np.einsum("mnej,abmn->abej", H.ab.oovo, T.ab, optimize=True)
            - np.einsum("me,abmj->abej", H.a.ov, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------
    return H

def get_ccsd_intermediates(T, H, H0, RHF_symmetry):

    # Cholesky-based intermediates folding some T2
    x_a_vo = (
            H.chol.a.vo.copy()
            + np.einsum("xnf,afin->xai", H.chol.a.ov, T.aa)
            + np.einsum("xnf,afin->xai", H.chol.b.ov, T.ab)
    )
    if RHF_symmetry:
        x_b_vo = x_a_vo.copy()
    else:
        x_b_vo = (
                H.chol.b.vo.copy()
                + np.einsum("xnf,afin->xai", H.chol.b.ov, T.bb)
                + np.einsum("xnf,fani->xai", H.chol.a.ov, T.ab)
        )

    # One-body part of HBar
    H.a.ov = H0.a.ov.copy() + (
            np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
            + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
    else:
        H.b.ov = H0.b.ov.copy() + (
                np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
                + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
        )

    H.a.oo = H0.a.oo.copy() + (
            np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.b.oo = H.a.oo.copy()
    else:
        H.b.oo = H0.b.oo.copy() + (
                np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
                + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
                + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
        )
    H.a.vv = H0.a.vv.copy() + (
            - 0.5 * np.einsum("mnef,afmn->ae", H0.aa.oovv, T.aa, optimize=True)  #
            - np.einsum("mnef,afmn->ae", H0.ab.oovv, T.ab, optimize=True)  #
    )
    bt1 = (
            np.einsum("xnf,fn->x", H0.chol.a.ov, T.a, optimize=True)
            + np.einsum("xnf,fn->x", H0.chol.b.ov, T.b, optimize=True)
    )
    bxt1 = -np.einsum("xne,fn->xfe", H0.chol.a.ov, T.a, optimize=True)
    H.a.vv += (
            np.einsum("xae,x->ae", H0.chol.a.vv, bt1, optimize=True)
            + np.einsum("xaf,xfe->ae", H0.chol.a.vv, bxt1, optimize=True)
            - np.einsum("me,am->ae", H.a.ov, T.a, optimize=True)
    )
    if RHF_symmetry:
        H.b.vv = H.a.vv.copy()
    else:
        H.b.vv = H0.b.vv.copy() + (
                - 0.5 * np.einsum("mnef,afmn->ae", H0.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfe,fanm->ae", H0.ab.oovv, T.ab, optimize=True)
                - np.einsum("me,am->ae", H.b.ov, T.b, optimize=True)
        )
        bt1 = (
                np.einsum("xnf,fn->x", H0.chol.b.ov, T.b, optimize=True)
                + np.einsum("xnf,fn->x", H0.chol.a.ov, T.a, optimize=True)
        )
        bxt1 = -np.einsum("xne,fn->xfe", H0.chol.b.ov, T.b, optimize=True)
        H.b.vv += (
                np.einsum("xae,x->ae", H0.chol.b.vv, bt1, optimize=True)
                + np.einsum("xaf,xfe->ae", H0.chol.b.vv, bxt1, optimize=True)
        )
    # H.a.ov = H0.a.ov.copy() + (
    #         np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
    #         + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    # )
    # if RHF_symmetry:
    #     H.b.ov = H.a.ov.copy()
    # else:
    #     H.b.ov = H0.b.ov.copy() + (
    #             np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
    #             + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    #     )
    #
    #
    # H.a.oo = H0.a.oo.copy() + (
    #         np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
    #         + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
    #         + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
    #         + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
    #         + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    # )
    # if RHF_symmetry:
    #     H.b.oo = H.a.oo.copy()
    # else:
    #     H.b.oo = H0.b.oo.copy() + (
    #             np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
    #             + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
    #             + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
    #             + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
    #             + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    #     )
    #
    # H.a.vv = H0.a.vv.copy() + (
    #         - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
    #         + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
    #         + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
    #         - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
    #         - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    # )
    # if RHF_symmetry:
    #     H.b.vv = H.a.vv.copy()
    # else:
    #     H.b.vv = H0.b.vv.copy() + (
    #             - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
    #             + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
    #             + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
    #             - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
    #             - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    #     )

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
            - np.einsum("xmj,xni->mnij", H.chol.a.oo, H.chol.a.oo, optimize=True)
            + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    )
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo.copy()
    else:
        H.bb.oooo = (
                np.einsum("xmi,xnj->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
                - np.einsum("xmj,xni->mnij", H.chol.b.oo, H.chol.b.oo, optimize=True)
                + 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True)
        )
    H.ab.oooo = (
            np.einsum("xmi,xnj->mnij", H.chol.a.oo, H.chol.b.oo, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov
    H.aa.ooov = np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.a.ov, optimize=True)
    H.aa.ooov -= np.transpose(H.aa.ooov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov.copy()
    else:
        H.bb.ooov = np.einsum("xmi,xne->mnie", H.chol.b.oo, H.chol.b.ov, optimize=True)
        H.bb.ooov -= np.transpose(H.bb.ooov, (1, 0, 2, 3))
    H.ab.ooov = (
        np.einsum("xmi,xne->mnie", H.chol.a.oo, H.chol.b.ov, optimize=True)
    )
    H.ab.oovo = (
        np.einsum("xme,xni->mnei", H.chol.a.ov, H.chol.b.oo, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    h2a_vovv = np.einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.a.ov, optimize=True)
    h2a_vovv -= np.transpose(h2a_vovv, (0, 1, 3, 2))
    if RHF_symmetry:
        h2c_vovv = h2a_vovv.copy()
    else:
        h2c_vovv = np.einsum("xbe,xnf->bnef", H.chol.b.vv, H.chol.b.ov, optimize=True)
        h2c_vovv -= np.transpose(h2c_vovv, (0, 1, 3, 2))
    h2b_vovv = np.einsum("xbe,xnf->bnef", H.chol.a.vv, H.chol.b.ov, optimize=True)
    h2b_ovvv = np.einsum("xnf,xbe->nbfe", H.chol.a.ov, H.chol.b.vv, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    H.aa.vooo = (
            np.einsum("xai,xmj->amij", x_a_vo, H.chol.a.oo, optimize=True)
            + 0.25 * np.einsum("amef,efij->amij", h2a_vovv, T.aa, optimize=True)
            + 0.5 * np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            #
            # This exchange term won't go away!!!
            #
            - np.einsum("xnj,xmf,afin->amij", H.chol.a.oo, H.chol.a.ov, T.aa, optimize=True)
    )
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo.copy()
    else:
        H.bb.vooo = (
                np.einsum("xai,xmj->amij", x_b_vo, H.chol.b.oo, optimize=True)
                + 0.25 * np.einsum("amef,efij->amij", h2c_vovv, T.bb, optimize=True)
                + 0.5 * np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
                #
                # This exchange term won't go away!!!
                #
                - np.einsum("xnj,xmf,afin->amij", H.chol.b.oo, H.chol.b.ov, T.bb, optimize=True)
        )
        H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))
    H.ab.vooo = (
            np.einsum("xai,xmj->amij", H.chol.a.vo, H.chol.b.oo, optimize=True)
            + np.einsum("amef,efij->amij", h2b_vovv, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
    )
    H.ab.ovoo = (
            np.einsum("xmi,xbj->mbij", H.chol.a.oo, H.chol.b.vo, optimize=True)
            + np.einsum("mnif,fbnj->mbij", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnif,fbnj->mbij", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfj,fbin->mbij", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("mbef,efij->mbij", h2b_ovvv, T.ab, optimize=True)
            + np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.a.ov, optimize=True)
            - np.einsum("xae,xmi->amie", H.chol.a.vv, H.chol.a.oo, optimize=True)
            + np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov.copy()
    else:
        H.bb.voov = (
                np.einsum("xai,xme->amie", H.chol.b.vo, H.chol.b.ov, optimize=True)
                - np.einsum("xae,xmi->amie", H.chol.b.vv, H.chol.b.oo, optimize=True)
                + np.einsum("mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
        )
    H.ab.voov = (
            np.einsum("xai,xme->amie", H.chol.a.vo, H.chol.b.ov, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )
    H.ab.ovvo = (
            np.einsum("xbj,xme->mbej", H.chol.b.vo, H.chol.a.ov, optimize=True)
            + np.einsum("mnef,fbnj->mbej", H0.aa.oovv, T.ab, optimize=True)
            + np.einsum("mnef,fbnj->mbej", H0.ab.oovv, T.bb, optimize=True)
    )
    H.ab.ovov = (
            np.einsum("xmi,xbe->mbie", H.chol.a.oo, H.chol.b.vv, optimize=True)
            - np.einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab, optimize=True)
    )
    H.ab.vovo = (
            np.einsum("xae,xmj->amej", H.chol.a.vv, H.chol.b.oo, optimize=True)
            - np.einsum("nmef,afnj->amej", H0.ab.oovv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    H.aa.vvov = (
            np.einsum("xai,xbe->abie", x_a_vo, H.chol.a.vv, optimize=True)
            + 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
            - 0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
            #
            # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
            #
            - np.einsum("xbf,xne,afin->abie", H.chol.a.vv, H.chol.a.ov, T.aa, optimize=True)
    )
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov.copy()
    else:
        H.bb.vvov = (
                np.einsum("xai,xbe->abie", x_b_vo, H.chol.b.vv, optimize=True)
                + 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
                - 0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
                #
                # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
                #
                - np.einsum("xbf,xne,afin->abie", H.chol.b.vv, H.chol.b.ov, T.bb, optimize=True)
        )
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))
    H.ab.vvov = (
            np.einsum("xai,xbe->abie", H.chol.a.vo, H.chol.b.vv, optimize=True)
            + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
            - np.einsum("anfe,fbin->abie", H.ab.vovv, T.ab, optimize=True)
            + np.einsum("mnie,abmn->abie", H.ab.ooov, T.ab, optimize=True)
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
    )
    H.ab.vvvo = (
            np.einsum("xae,xbj->abej", H.chol.a.vv, H.chol.b.vo, optimize=True)
            + np.einsum("anef,fbnj->abej", H.aa.vovv, T.ab, optimize=True)
            + np.einsum("anef,fbnj->abej", H.ab.vovv, T.bb, optimize=True)
            - np.einsum("mbef,afmj->abej", H.ab.ovvv, T.ab, optimize=True)
            + np.einsum("mnej,abmn->abej", H.ab.oovo, T.ab, optimize=True)
            - np.einsum("me,abmj->abej", H.a.ov, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # # -------------------------------------------------------------------------
    # ### TYPE: VVVV
    # H.aa.vvvv = (
    #         np.einsum("xae,xbf->abef", H.chol.a.vv, H.chol.a.vv, optimize=True)
    #         - np.einsum("xaf,xbe->abef", H.chol.a.vv, H.chol.a.vv, optimize=True)
    #         + 0.5 * np.einsum("mnef,abmn->abef", H0.aa.oovv, T.aa, optimize=True)
    # )
    # H.ab.vvvv = (
    #         np.einsum("xae,xbf->abef", H.chol.a.vv, H.chol.b.vv, optimize=True)
    #         + np.einsum("mnef,abmn->abef", H0.ab.oovv, T.ab, optimize=True)
    # )
    # if RHF_symmetry:
    #     H.bb.vvvv = H.aa.vvvv.copy()
    # else:
    #     H.bb.vvvv = (
    #             np.einsum("xae,xbf->abef", H.chol.b.vv, H.chol.b.vv, optimize=True)
    #             - np.einsum("xaf,xbe->abef", H.chol.b.vv, H.chol.b.vv, optimize=True)
    #             + 0.5 * np.einsum("mnef,abmn->abef", H0.bb.oovv, T.bb, optimize=True)
    #     )
    # # -------------------------------------------------------------------------
    return H

def build_hbar_ccsd_chol_bak(T, H0, RHF_symmetry, *args):

    # Reference to HBar is copied from reference H (not duplicated!) 
    H = H0

    # Orbital dimensions
    nua, nub, noa, nob = T.ab.shape

    H.a.ov += (
                np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
                + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )

    H.a.oo += (
                np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
                + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.a.vv += (
                - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
                - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.ov += (
                np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
                + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    )

    H.b.oo += (
                np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
                + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
                + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.vv += (
                - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
                + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
                + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    # -------------------------------------------------------------------------
    # Make useful intermediates
    tau_aa = 0.5 * T.aa + np.einsum("ai,bj->abij", T.a, T.a, optimize=True)
    tau_aa -= np.transpose(tau_aa, (0, 1, 3, 2))
    if RHF_symmetry:
        tau_bb = tau_aa
    else:
        tau_bb = 0.5 * T.bb + np.einsum("ai,bj->abij", T.b, T.b, optimize=True)
        tau_bb -= np.transpose(tau_bb, (0, 1, 3, 2))
    tau_ab = T.ab + np.einsum("ai,bj->abij", T.a, T.b, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOO
    ### NEEDS: H0.ooov
    H.aa.oooo = (
            0.5 * H0.aa.oooo
            + np.einsum("nmje,ei->mnij", H0.aa.ooov, T.a, optimize=True)
            + 0.25 * np.einsum("mnef,efij->mnij", H0.aa.oovv, tau_aa, optimize=True)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo
    else:
        H.bb.oooo = (
                0.5 * H0.bb.oooo
                + np.einsum("nmje,ei->mnij", H0.bb.ooov, T.b, optimize=True)
                + 0.25 * np.einsum("mnef,efij->mnij", H0.bb.oovv, tau_bb, optimize=True)
        )
        H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))
    H.ab.oooo = (
            H0.ab.oooo
            + np.einsum("mnej,ei->mnij", H0.ab.oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", H0.ab.ooov, T.b, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, tau_ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: OOOV
    ### NEEDS: H0.ooov 
    H.aa.ooov += np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)
    if RHF_symmetry:
        H.bb.ooov = H.aa.ooov
    else:
        H.bb.ooov += np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    H.ab.ooov += np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    H.ab.oovo += np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOO
    ### NEEDS: H.ov, H.oooo, H.ooov, H0.voov, H0.vovv
    Q1 = (
            np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo = H0.aa.vooo + Q1 + (
            np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
    )
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo
    else:
        Q1 = (
                np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
                + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
        )
        Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
        Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
        Q1 += Q2
        Q1 -= np.transpose(Q1, (0, 1, 3, 2))
        H.bb.vooo = H0.bb.vooo + Q1 + (
                + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
                - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
                + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
        )
    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo = H0.ab.vooo + (
            np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
            - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
            + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
    )
    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    H.ab.ovoo = H0.ab.ovoo + (
            np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
            - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
            + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("mafe,feji->maji", H0.ab.ovvv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOOV
    ### NEDDS: H0.vovv, H.ooov
    H.aa.voov = (
            H0.aa.voov
            + np.einsum("amfe,fi->amie", H0.aa.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.aa.ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov
    else:
        H.bb.voov = (
                H0.bb.voov
                + np.einsum("amfe,fi->amie", H0.bb.vovv, T.b, optimize=True)
                - np.einsum("nmie,an->amie", H.bb.ooov, T.b, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
        )
    H.ab.voov = (
            H0.ab.voov
            + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.ab.ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )
    H.ab.ovvo = (
            H0.ab.ovvo
            + np.einsum("maef,fi->maei", H0.ab.ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", H.ab.oovo, T.b, optimize=True)
            + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
            + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )
    H.ab.ovov = (
            H0.ab.ovov
            + np.einsum("mafe,fi->maie", H0.ab.ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", H.ab.ooov, T.b, optimize=True)
            - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )
    H.ab.vovo = (
            H0.ab.vovo
            - np.einsum("nmei,an->amei", H.ab.oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", H0.ab.vovv, T.b, optimize=True)
            - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVOV
    ### NEDDS: H.ov, H.voov, H.ooov, H0.vvvv, H0.vovv
    x2a_voov = H.aa.voov + 0.5 * np.einsum("nmie,an->amie", H.aa.ooov, T.a, optimize=True) # defined to avoid double-counting from A(ab) on [8] and [15]
    H.aa.vvov = (
            0.5 * H0.aa.vvov # [1]
            - 0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True) # [4]+[12]+[13]
            - np.einsum("amie,bm->abie", x2a_voov, T.a, optimize=True) # [3]+[8']+[9]+[11]+[13]+[15']
            + 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True) # [6]+[10]
            # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
            + np.einsum("bnef,afin->abie", H0.aa.vovv, T.aa, optimize=True) # [5]
            + np.einsum("bnef,afin->abie", H0.ab.vovv, T.ab, optimize=True) # [7]
    )
    # Add on h0(vvvv) term using Cholesky
    for a in range(nua):
        for b in range(a + 1, nua):
            batch_ints = build_2index_batch_vvvv_aa(a, b, H0)
            H.aa.vvov[a, b, :, :] += np.einsum("fe,fi->ie", batch_ints, T.a, optimize=True)
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov
    else:
        x2c_voov = H.bb.voov + 0.5 * np.einsum("nmie,an->amie", H.bb.ooov, T.b, optimize=True)  # defined to avoid double-counting from A(ab) on [8] and [15]
        H.bb.vvov = (
                0.5 * H0.bb.vvov  # [1]
                - 0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)  # [4]+[12]+[13]
                - np.einsum("amie,bm->abie", x2c_voov, T.b, optimize=True)  # [3]+[8']+[9]+[11]+[13]+[15']
                + 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)  # [6]+[10]
                # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
                + np.einsum("bnef,afin->abie", H0.bb.vovv, T.bb, optimize=True)  # [5]
                + np.einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab, optimize=True)  # [7]
        )
        # Add on h0(vvvv) term using Cholesky
        for a in range(nub):
            for b in range(a + 1, nub):
                batch_ints = build_2index_batch_vvvv_bb(a, b, H0)
                H.bb.vvov[a, b, :, :] += np.einsum("fe,fi->ie", batch_ints, T.b, optimize=True)
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    # need to define x2b_voov and x2b_ovov such that [10] and [18] are not double counted
    x2b_voov = H.ab.voov + 0.5 * np.einsum("nmie,an->amie", H.ab.ooov, T.a, optimize=True) # nu2no3
    x2b_ovov = H.ab.ovov + 0.5 * np.einsum("nmie,bm->nbie", H.ab.ooov, T.b, optimize=True) # nu2no3
    H.ab.vvov = (
            H0.ab.vvov # [1]
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True) # [8] + [13] + [16]
            - np.einsum("amie,bm->abie", x2b_voov, T.b, optimize=True) # [4] + 1/2*[10] + [11] + [12] + [17] + 1/2*[18]
            - np.einsum("mbie,am->abie", x2b_ovov, T.a, optimize=True) # [2] + [9] + 1/2*[10] + [15] + 1/2*[18]
            + np.einsum("nmie,abnm->abie", H.ab.ooov, T.ab, optimize=True) # [7] + [14]
            # Terms we have to deal with directly that are nu^4: [2], [5], [6], and [19]
            + np.einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa, optimize=True) # [5]
            + np.einsum("bmef,afim->abie", H0.bb.vovv, T.ab, optimize=True) # [6]
            - np.einsum("amfe,fbim->abie", H0.ab.vovv, T.ab, optimize=True) # [19]
    )
    x2b_ovvo = H.ab.ovvo + 0.5 * np.einsum("nmei,am->naei", H.ab.oovo, T.b, optimize=True)
    x2b_vovo = H.ab.vovo + 0.5 * np.einsum("nmei,bn->bmei", H.ab.oovo, T.a, optimize=True)
    H.ab.vvvo = (
            H0.ab.vvvo # [1]
            - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True) # [8] + [13] + [16]
            - np.einsum("maei,bm->baei", x2b_ovvo, T.a, optimize=True) # [4] + 1/2*[10] + [11] + [12] + [14] + 1/2*[17]
            - np.einsum("bmei,am->baei", x2b_vovo, T.b, optimize=True) # [3] + [9] + 1/2*[10] + 1/2*[17] + [18]
            + np.einsum("nmei,banm->baei", H.ab.oovo, T.ab, optimize=True) # [6] + [15]
            # Terms we have to deal with directly that are nu^4: [2], [5], [7], and [19]
            + np.einsum("bmef,afim->baei", H0.ab.vovv, T.bb, optimize=True) # [5]
            + np.einsum("bmef,fami->baei", H0.aa.vovv, T.ab, optimize=True) # [7]
            - np.einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab, optimize=True) # [19]
    )
    # add on h0(vvvv) term using Cholesky
    for a in range(nua):
        batch_ints = build_3index_batch_vvvv_ab(a, H0)
        H.ab.vvov[a, :, :, :] += np.einsum("bfe,fi->bie", batch_ints, T.a, optimize=True)
        H.ab.vvvo[a, :, :, :] += np.einsum("bef,fi->bei", batch_ints, T.b, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVVV
    ### NEEDS: H0.vovv
    #H.aa.vvvv = (
    #        0.5 * H0.aa.vvvv
    #        + 0.25 * np.einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa, optimize=True)
    #        - np.einsum("amef,bm->abef", H0.aa.vovv, T.a, optimize=True)
    #)
    #H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    #if RHF_symmetry:
    #    H.bb.vvvv = H.aa.vvvv
    #else:
    #    H.bb.vvvv = (
    #            0.5 * H0.bb.vvvv
    #            + 0.25 * np.einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb, optimize=True)
    #            - np.einsum("amef,bm->abef", H0.bb.vovv, T.b, optimize=True)
    #    )
    #    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    #H.ab.vvvv = (
    #        H0.ab.vvvv
    #        - np.einsum("mbef,am->abef", H0.ab.ovvv, T.a, optimize=True)
    #        - np.einsum("amef,bm->abef", H0.ab.vovv, T.b, optimize=True)
    #        + np.einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab, optimize=True)
    #)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VOVV
    ### NEEDS: H0.vovv
    H.aa.vovv -= np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)
    if RHF_symmetry:
        H.bb.vovv = H.aa.vovv
    else:
        H.bb.vovv -= np.einsum("mnfe,an->amef", H0.bb.oovv, T.b, optimize=True)
    H.ab.vovv -= np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True) 
    H.ab.ovvv -= np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### T1-transformation of Cholesky vectors
    print("   Performing T1-transformation of Cholesky vectors")
    ### a ###
    H.chol.a.oo = H0.chol.a.oo.copy() + np.einsum("xme,ei->xmi", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vv = H0.chol.a.vv.copy() - np.einsum("xme,am->xae", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vo = (
            H.chol.a.vo.copy()
            - np.einsum("xmi,am->xai", H.chol.a.oo, T.a, optimize=True)
            + np.einsum("xae,ei->xai", H.chol.a.vv, T.a, optimize=True)
            + np.einsum("xme,ei,am->xai", H.chol.a.ov, T.a, T.a, optimize=True)
    )
    ### b ###
    H.chol.b.oo = H0.chol.b.oo.copy() + np.einsum("xme,ei->xmi", H.chol.b.ov, T.b, optimize=True)
    H.chol.b.vv = H0.chol.b.vv.copy() - np.einsum("xme,am->xae", H.chol.b.ov, T.b, optimize=True)
    H.chol.b.vo = (
            H.chol.b.vo.copy()
            - np.einsum("xmi,am->xai", H.chol.b.oo, T.b, optimize=True)
            + np.einsum("xae,ei->xai", H.chol.b.vv, T.b, optimize=True)
            + np.einsum("xme,ei,am->xai", H.chol.b.ov, T.b, T.b, optimize=True)
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
    tau_aa = 0.5 * T.aa + np.einsum("ai,bj->abij", T.a, T.a, optimize=True)
    tau_aa -= np.transpose(tau_aa, (0, 1, 3, 2))
    if RHF_symmetry:
        tau_bb = tau_aa
    else:
        tau_bb = 0.5 * T.bb + np.einsum("ai,bj->abij", T.b, T.b, optimize=True)
        tau_bb -= np.transpose(tau_bb, (0, 1, 3, 2))
    tau_ab = T.ab + np.einsum("ai,bj->abij", T.a, T.b, optimize=True)

    H.a.ov += (
                np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
                + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )

    H.a.oo += (
                np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
                + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.a.vv += (
                - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
                - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.ov += (
                np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
                + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    )

    H.b.oo += (
                np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
                + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
                + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.vv += (
                - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
                + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
                + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    )
    
    Q1 = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.bb.oovv, T.b, optimize=True)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    #x1 = time.time()
    # H.aa.vvvv = (
    #        0.5 * H0.aa.vvvv
    #        + 0.25 * np.einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa, optimize=True)
    #        - np.einsum("amef,bm->abef", H0.aa.vovv, T.a, optimize=True)
    # )
    # H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    # if RHF_symmetry:
    #    H.bb.vvvv = H.aa.vvvv
    # else:
    #    H.bb.vvvv = (
    #            0.5 * H0.bb.vvvv
    #            + 0.25 * np.einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb, optimize=True)
    #            - np.einsum("amef,bm->abef", H0.bb.vovv, T.b, optimize=True)
    #    )
    #    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    # H.ab.vvvv = (
    #        H0.ab.vvvv
    #        - np.einsum("mbef,am->abef", H0.ab.ovvv, T.a, optimize=True)
    #        - np.einsum("amef,bm->abef", H0.ab.vovv, T.b, optimize=True)
    #        + np.einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab, optimize=True)
    # )
    #x2 = time.time()
    #print("vvvv:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.oooo = (
            0.5 * H0.aa.oooo
            + np.einsum("nmje,ei->mnij", H0.aa.ooov, T.a, optimize=True)
            + 0.25 * np.einsum("mnef,efij->mnij", H0.aa.oovv, tau_aa, optimize=True)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo
    else:
        H.bb.oooo = (
                0.5 * H0.bb.oooo
                + np.einsum("nmje,ei->mnij", H0.bb.ooov, T.b, optimize=True)
                + 0.25 * np.einsum("mnef,efij->mnij", H0.bb.oovv, tau_bb, optimize=True)
        )
        H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))
    H.ab.oooo = (
            H0.ab.oooo
            + np.einsum("mnej,ei->mnij", H0.ab.oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", H0.ab.ooov, T.b, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, tau_ab, optimize=True)
    )
    #x2 = time.perf_counter()
    #print("oooo:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.voov = (
            H0.aa.voov
            + np.einsum("amfe,fi->amie", H0.aa.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.aa.ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov
    else:
        H.bb.voov = (
                H0.bb.voov
                + np.einsum("amfe,fi->amie", H0.bb.vovv, T.b, optimize=True)
                - np.einsum("nmie,an->amie", H.bb.ooov, T.b, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
        )
    H.ab.voov = (
            H0.ab.voov
            + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.ab.ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )
    H.ab.ovvo = (
            H0.ab.ovvo
            + np.einsum("maef,fi->maei", H0.ab.ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", H.ab.oovo, T.b, optimize=True)
            + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
            + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )
    H.ab.ovov = (
            H0.ab.ovov
            + np.einsum("mafe,fi->maie", H0.ab.ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", H.ab.ooov, T.b, optimize=True)
            - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )
    H.ab.vovo = (
            H0.ab.vovo
            - np.einsum("nmei,an->amei", H.ab.oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", H0.ab.vovv, T.b, optimize=True)
            - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )
    #x2 = time.perf_counter()
    #print("voov:", x2 - x1)

    #x1 = time.perf_counter()
    Q1 = (
            np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo = H0.aa.vooo + Q1 + (
            np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
    )
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo
    else:
        Q1 = (
                np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
                + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
        )
        Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
        Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
        Q1 += Q2
        Q1 -= np.transpose(Q1, (0, 1, 3, 2))
        H.bb.vooo = H0.bb.vooo + Q1 + (
                + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
                - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
                + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
        )
    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo = H0.ab.vooo + (
            np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
            - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
            + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
    )
    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    H.ab.ovoo = H0.ab.ovoo + (
            np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
            - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
            + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("mafe,feji->maji", H0.ab.ovvv, T.ab, optimize=True)
    )
    #x2 = time.perf_counter()
    #print("vooo:", x2 - x1)

    #x1 = time.perf_counter()
    x2a_voov = H.aa.voov + 0.5 * np.einsum("nmie,an->amie", H.aa.ooov, T.a, optimize=True) # defined to avoid double-counting from A(ab) on [8] and [15]
    H.aa.vvov = (
            0.5 * H0.aa.vvov # [1]
            - 0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True) # [4]+[12]+[13]
            - np.einsum("amie,bm->abie", x2a_voov, T.a, optimize=True) # [3]+[8']+[9]+[11]+[13]+[15']
            + 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True) # [6]+[10]
            # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
            #+ 0.5 * np.einsum("abfe,fi->abie", H0.aa.vvvv, T.a, optimize=True) # [2]
            + np.einsum("bnef,afin->abie", H0.aa.vovv, T.aa, optimize=True) # [5]
            + np.einsum("bnef,afin->abie", H0.ab.vovv, T.ab, optimize=True) # [7]
    )
    # Add on h0(vvvv) term using Cholesky
    for a in range(nua):
        for b in range(a + 1, nua):
            batch_ints = build_2index_batch_vvvv_aa(a, b, H0)
            H.aa.vvov[a, b, :, :] += np.einsum("fe,fi->ie", batch_ints, T.a, optimize=True)
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov
    else:
        x2c_voov = H.bb.voov + 0.5 * np.einsum("nmie,an->amie", H.bb.ooov, T.b, optimize=True)  # defined to avoid double-counting from A(ab) on [8] and [15]
        H.bb.vvov = (
                0.5 * H0.bb.vvov  # [1]
                - 0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)  # [4]+[12]+[13]
                - np.einsum("amie,bm->abie", x2c_voov, T.b, optimize=True)  # [3]+[8']+[9]+[11]+[13]+[15']
                + 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)  # [6]+[10]
                # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
                #+ 0.5 * np.einsum("abfe,fi->abie", H0.bb.vvvv, T.b, optimize=True)  # [2]
                + np.einsum("bnef,afin->abie", H0.bb.vovv, T.bb, optimize=True)  # [5]
                + np.einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab, optimize=True)  # [7]
        )
        # Add on h0(vvvv) term using Cholesky
        for a in range(nub):
            for b in range(a + 1, nub):
                batch_ints = build_2index_batch_vvvv_bb(a, b, H0)
                H.bb.vvov[a, b, :, :] += np.einsum("fe,fi->ie", batch_ints, T.b, optimize=True)
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    # need to define x2b_voov and x2b_ovov such that [10] and [18] are not double counted
    x2b_voov = H.ab.voov + 0.5 * np.einsum("nmie,an->amie", H.ab.ooov, T.a, optimize=True) # nu2no3
    x2b_ovov = H.ab.ovov + 0.5 * np.einsum("nmie,bm->nbie", H.ab.ooov, T.b, optimize=True) # nu2no3
    H.ab.vvov = (
            H0.ab.vvov # [1]
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True) # [8] + [13] + [16]
            - np.einsum("amie,bm->abie", x2b_voov, T.b, optimize=True) # [4] + 1/2*[10] + [11] + [12] + [17] + 1/2*[18]
            - np.einsum("mbie,am->abie", x2b_ovov, T.a, optimize=True) # [2] + [9] + 1/2*[10] + [15] + 1/2*[18]
            + np.einsum("nmie,abnm->abie", H.ab.ooov, T.ab, optimize=True) # [7] + [14]
            # Terms we have to deal with directly that are nu^4: [2], [5], [6], and [19]
            #+ np.einsum("abfe,fi->abie", H0.ab.vvvv, T.a, optimize=True) # [2]
            + np.einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa, optimize=True) # [5]
            + np.einsum("bmef,afim->abie", H0.bb.vovv, T.ab, optimize=True) # [6]
            - np.einsum("amfe,fbim->abie", H0.ab.vovv, T.ab, optimize=True) # [19]
    )
    x2b_ovvo = H.ab.ovvo + 0.5 * np.einsum("nmei,am->naei", H.ab.oovo, T.b, optimize=True)
    x2b_vovo = H.ab.vovo + 0.5 * np.einsum("nmei,bn->bmei", H.ab.oovo, T.a, optimize=True)
    H.ab.vvvo = (
            H0.ab.vvvo # [1]
            - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True) # [8] + [13] + [16]
            - np.einsum("maei,bm->baei", x2b_ovvo, T.a, optimize=True) # [4] + 1/2*[10] + [11] + [12] + [14] + 1/2*[17]
            - np.einsum("bmei,am->baei", x2b_vovo, T.b, optimize=True) # [3] + [9] + 1/2*[10] + 1/2*[17] + [18]
            + np.einsum("nmei,banm->baei", H.ab.oovo, T.ab, optimize=True) # [6] + [15]
            # Terms we have to deal with directly that are nu^4: [2], [5], [7], and [19]
            #+ np.einsum("baef,fi->baei", H0.ab.vvvv, T.b, optimize=True) # [2]
            + np.einsum("bmef,afim->baei", H0.ab.vovv, T.bb, optimize=True) # [5]
            + np.einsum("bmef,fami->baei", H0.aa.vovv, T.ab, optimize=True) # [7]
            - np.einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab, optimize=True) # [19]
    )
    #x2 = time.perf_counter()
    #print("vvov:", x2 - x1)
    # add on h0(vvvv) term using Cholesky
    for a in range(nua):
        batch_ints = build_3index_batch_vvvv_ab(a, H0)
        H.ab.vvov[a, :, :, :] += np.einsum("bfe,fi->bie", batch_ints, T.a, optimize=True)
        H.ab.vvvo[a, :, :, :] += np.einsum("bef,fi->bei", batch_ints, T.b, optimize=True)
    return H

