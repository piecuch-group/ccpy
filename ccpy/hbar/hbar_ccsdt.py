import numpy as np

def build_hbar_ccsdt(T, H0, RHF_symmetry, *args):

    # Reference to HBar is copied from reference H (not duplicated!) 
    H = H0

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
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
            + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
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
                + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
                + np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
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
            + np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.abb, optimize=True)
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
            + 0.5 * np.einsum("mnef,efajni->maji", H0.aa.oovv, T.aab, optimize=True)
            + np.einsum("mnef,efajni->maji", H0.ab.oovv, T.abb, optimize=True)
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
            + 0.5 * np.einsum("abfe,fi->abie", H0.aa.vvvv, T.a, optimize=True) # [2]
            + np.einsum("bnef,afin->abie", H0.aa.vovv, T.aa, optimize=True) # [5]
            + np.einsum("bnef,afin->abie", H0.ab.vovv, T.ab, optimize=True) # [7]
            - 0.25 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
            - 0.5 * np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    )
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
                + 0.5 * np.einsum("abfe,fi->abie", H0.bb.vvvv, T.b, optimize=True)  # [2]
                + np.einsum("bnef,afin->abie", H0.bb.vovv, T.bb, optimize=True)  # [5]
                + np.einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab, optimize=True)  # [7]
                - 0.25 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
                - 0.5 * np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
        )
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
            + np.einsum("abfe,fi->abie", H0.ab.vvvv, T.a, optimize=True) # [2]
            + np.einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa, optimize=True) # [5]
            + np.einsum("bmef,afim->abie", H0.bb.vovv, T.ab, optimize=True) # [6]
            - np.einsum("amfe,fbim->abie", H0.ab.vovv, T.ab, optimize=True) # [19]
            - np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
            - 0.5 * np.einsum("mnef,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
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
            + np.einsum("baef,fi->baei", H0.ab.vvvv, T.b, optimize=True) # [2]
            + np.einsum("bmef,afim->baei", H0.ab.vovv, T.bb, optimize=True) # [5]
            + np.einsum("bmef,fami->baei", H0.aa.vovv, T.ab, optimize=True) # [7]
            - np.einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab, optimize=True) # [19]
            - 0.5 * np.einsum("mnef,bfamni->baei", H0.aa.oovv, T.aab, optimize=True)
            - np.einsum("mnef,bfamni->baei", H0.ab.oovv, T.abb, optimize=True)
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### TYPE: VVVV
    ### NEEDS: H0.vovv
    H.aa.vvvv = (
            0.5 * H0.aa.vvvv
            + 0.25 * np.einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa, optimize=True)
            - np.einsum("amef,bm->abef", H0.aa.vovv, T.a, optimize=True)
    )
    H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvvv = H.aa.vvvv
    else:
        H.bb.vvvv = (
                0.5 * H0.bb.vvvv
                + 0.25 * np.einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb, optimize=True)
                - np.einsum("amef,bm->abef", H0.bb.vovv, T.b, optimize=True)
        )
        H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    H.ab.vvvv = (
            H0.ab.vvvv
            - np.einsum("mbef,am->abef", H0.ab.ovvv, T.a, optimize=True)
            - np.einsum("amef,bm->abef", H0.ab.vovv, T.b, optimize=True)
            + np.einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab, optimize=True)
    )
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
    return H

def build_hbar_ccsdt_debug(T, H0, RHF_symmetry, *args):
    """Calculate the CCSDT similarity-transformed Hamiltonian (H_N e^(T1+T2))_C.
    Copied as-is from original CCpy implementation."""
    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

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

    Q1 = -np.einsum("bmfe,am->abef", I2A_vovv, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.vvvv += (
            - np.einsum("mbef,am->abef", I2B_ovvv, T.a, optimize=True)
            - np.einsum("amef,bm->abef", I2B_vovv, T.b, optimize=True)
            + np.einsum("mnef,abmn->abef", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("bmfe,am->abef", I2C_vovv, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.bb.oovv, T.bb, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.oooo += (
            np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True) + Q1

    H.aa.voov += (
            np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.voov += (
            np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )

    H.ab.ovvo += (
            np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
            + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
            + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )

    H.ab.ovov += (
            np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
            - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.vovo += (
            - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
            - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )

    H.bb.voov += (
            np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
            + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
            + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    )

    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo += (
            np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
            - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
            + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
            + np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.abb, optimize=True)
    )

    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    H.ab.ovoo += (
            np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
            - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
            + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("mafe,feji->maji", H0.ab.ovvv, T.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efajni->maji", H0.aa.oovv, T.aab, optimize=True)
            + np.einsum("mnef,efajni->maji", H0.ab.oovv, T.abb, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
            + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    )
    Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
            - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
            + np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    )
    Q2 = H0.aa.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.aa.ooov, T.a, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
            + np.einsum("abfe,fi->abie", H.aa.vvvv, T.a, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.aa.ooov, T.aa, optimize=True)
            - 0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
            - np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    )

    Q1 = H0.ab.ovov - np.einsum("mnie,bn->mbie", H0.ab.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    H.ab.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
            + np.einsum("abfe,fi->abie", H.ab.vvvv, T.a, optimize=True)
            + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
            - np.einsum("amfe,fbim->abie", H.ab.vovv, T.ab, optimize=True)
            - np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True)
            + np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True)
            - np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
            - 0.5 * np.einsum("mnef,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    )

    Q1 = H0.ab.vovo - np.einsum("nmei,bn->bmei", H0.ab.oovo, T.a, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, T.b, optimize=True)
    H.ab.vvvo += Q1 + (
            - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True)
            + np.einsum("baef,fi->baei", H.ab.vvvv, T.b, optimize=True)
            + np.einsum("bnef,fani->baei", H.aa.vovv, T.ab, optimize=True)
            + np.einsum("bnef,fani->baei", H.ab.vovv, T.bb, optimize=True)
            - np.einsum("maef,bfmi->baei", H.ab.ovvv, T.ab, optimize=True)
            - np.einsum("naei,bn->baei", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("nmei,banm->baei", H0.ab.oovo, T.ab, optimize=True)
            - 0.5 * np.einsum("mnef,bfamni->baei", H0.aa.oovv, T.aab, optimize=True)
            - np.einsum("mnef,bfamni->baei", H0.ab.oovv, T.abb, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
            + np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    )
    Q2 = H.bb.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.bb.ooov, T.b, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
            + np.einsum("abfe,fi->abie", H.bb.vvvv, T.b, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.bb.ooov, T.bb, optimize=True)
            - 0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
            - np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    )

    # For RHF symmetry, copy a parts to b and aa parts to bb
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
        H.b.oo = H.a.oo.copy()
        H.b.vv = H.a.vv.copy()
        H.bb.oooo = H.aa.oooo.copy()
        H.bb.ooov = H.aa.ooov.copy()
        H.bb.vooo = H.aa.vooo.copy()
        H.bb.oovv = H.aa.oovv.copy()
        H.bb.voov = H.aa.voov.copy()
        H.bb.vovv = H.aa.vovv.copy()
        H.bb.vvov = H.aa.vvov.copy()
        H.bb.vvvv = H.aa.vvvv.copy()

    return H

def add_VT3_intermediates(T, H):

    H.aa.vooo += (
            0.5 * np.einsum("mnef,aefijn->amij", H.aa.oovv, T.aaa, optimize=True)
            + np.einsum("mnef,aefijn->amij", H.ab.oovv, T.aab, optimize=True)
    )

    H.ab.vooo += (
            np.einsum("nmfe,afeinj->amij", H.ab.oovv, T.aab, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H.bb.oovv, T.abb, optimize=True)
    )

    H.ab.ovoo += (
            0.5 * np.einsum("mnef,efajni->maji", H.aa.oovv, T.aab, optimize=True)
            + np.einsum("mnef,efajni->maji", H.ab.oovv, T.abb, optimize=True)
    )

    H.bb.vooo += (
            0.5 * np.einsum("mnef,aefijn->amij", H.bb.oovv, T.bbb, optimize=True)
            + np.einsum("nmfe,faenij->amij", H.ab.oovv, T.abb, optimize=True)
    )

    H.aa.vvov += (
            - 0.5 * np.einsum("mnef,abfimn->abie", H.aa.oovv, T.aaa, optimize=True)
            - np.einsum("mnef,abfimn->abie", H.ab.oovv, T.aab, optimize=True)
    )

    H.ab.vvov += (
            - np.einsum("nmfe,afbinm->abie", H.ab.oovv, T.aab, optimize=True)
            - 0.5 * np.einsum("mnef,afbinm->abie", H.bb.oovv, T.abb, optimize=True)
    )

    H.ab.vvvo += (
            - 0.5 * np.einsum("mnef,bfamni->baei", H.aa.oovv, T.aab, optimize=True)
            - np.einsum("mnef,bfamni->baei", H.ab.oovv, T.abb, optimize=True)
    )

    H.bb.vvov += (
            - 0.5 * np.einsum("mnef,abfimn->abie", H.bb.oovv, T.bbb, optimize=True)
            - np.einsum("nmfe,fabnim->abie", H.ab.oovv, T.abb, optimize=True)
    )

    return H


