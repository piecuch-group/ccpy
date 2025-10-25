import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def build_hbar_ccsdt1(T, H0, RHF_symmetry, system, *args):
    """Calculate the CCSDt similarity-transformed Hamiltonian (H_N e^(T1+T2))_C."""
    from copy import deepcopy
    from ccpy.utilities.active_space import get_active_slices

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

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

    Q1 = -ccpy_einsum("bmfe,am->abef", I2A_vovv, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, T.aa) + Q1

    H.ab.vvvv += (
            - ccpy_einsum("mbef,am->abef", I2B_ovvv, T.a)
            - ccpy_einsum("amef,bm->abef", I2B_vovv, T.b)
            + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, T.ab)
    )

    Q1 = -ccpy_einsum("bmfe,am->abef", I2C_vovv, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, T.bb) + Q1

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2A_ooov, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa) + Q1

    H.ab.oooo += (
            ccpy_einsum("mnej,ei->mnij", I2B_oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", I2B_ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    )

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2C_ooov, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb) + Q1

    H.aa.voov += (
            ccpy_einsum("amfe,fi->amie", I2A_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2A_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )

    H.ab.voov += (
            ccpy_einsum("amfe,fi->amie", I2B_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2B_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )

    H.ab.ovvo += (
            ccpy_einsum("maef,fi->maei", I2B_ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", I2B_oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )

    H.ab.ovov += (
            ccpy_einsum("mafe,fi->maie", I2B_ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", I2B_ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )

    H.ab.vovo += (
            - ccpy_einsum("nmei,an->amei", I2B_oovo, T.a)
            + ccpy_einsum("amef,fi->amei", I2B_vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )

    H.bb.voov += (
            ccpy_einsum("amfe,fi->amie", I2C_vovv, T.b)
            - ccpy_einsum("nmie,an->amie", I2C_ooov, T.b)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
            + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )

    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo += (
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
    H.ab.ovoo += (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
            + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
    )
    Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
            - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.aa.vovv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.ab.vovv, T.ab)
    )
    Q2 = H0.aa.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.aa.ooov, T.a)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
            + ccpy_einsum("abfe,fi->abie", H.aa.vvvv, T.a)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.aa.ooov, T.aa)
    )

    Q1 = H0.ab.ovov - ccpy_einsum("mnie,bn->mbie", H0.ab.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    H.ab.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab)
            + ccpy_einsum("abfe,fi->abie", H.ab.vvvv, T.a)
            + ccpy_einsum("nbfe,afin->abie", H.ab.ovvv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.ab)
            - ccpy_einsum("amfe,fbim->abie", H.ab.vovv, T.ab)
            - ccpy_einsum("amie,bm->abie", H0.ab.voov, T.b)
            + ccpy_einsum("nmie,abnm->abie", H0.ab.ooov, T.ab)
    )

    Q1 = H0.ab.vovo - ccpy_einsum("nmei,bn->bmei", H0.ab.oovo, T.a)
    Q1 = -ccpy_einsum("bmei,am->baei", Q1, T.b)
    H.ab.vvvo += Q1 + (
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab)
            + ccpy_einsum("baef,fi->baei", H.ab.vvvv, T.b)
            + ccpy_einsum("bnef,fani->baei", H.aa.vovv, T.ab)
            + ccpy_einsum("bnef,fani->baei", H.ab.vovv, T.bb)
            - ccpy_einsum("maef,bfmi->baei", H.ab.ovvv, T.ab)
            - ccpy_einsum("naei,bn->baei", H0.ab.ovvo, T.a)
            + ccpy_einsum("nmei,banm->baei", H0.ab.oovo, T.ab)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.bb)
            + ccpy_einsum("nbfe,fani->abie", H.ab.ovvv, T.ab)
    )
    Q2 = H.bb.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.bb.ooov, T.b)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
            + ccpy_einsum("abfe,fi->abie", H.bb.vvvv, T.b)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.bb.ooov, T.bb)
    )

    #########################################################################
    ############################ aa intermdiates ############################
    #########################################################################
    
    ##### H.aa.vooo #####
    # AmIJ
    H.aa.vooo[Va, :, Oa, Oa] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,AfenIJ->AmIJ', H.aa.oovv[:, oa, va, va], T.aaa.VvvoOO)
            - 0.5 * ccpy_einsum('mNef,AfeIJN->AmIJ', H.aa.oovv[:, Oa, va, va], T.aaa.VvvOOO)
            + 1.0 * ccpy_einsum('mneF,FAenIJ->AmIJ', H.aa.oovv[:, oa, va, Va], T.aaa.VVvoOO)
            + 1.0 * ccpy_einsum('mNeF,FAeIJN->AmIJ', H.aa.oovv[:, Oa, va, Va], T.aaa.VVvOOO)
            - 0.5 * ccpy_einsum('mnEF,FEAnIJ->AmIJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVVoOO)
            - 0.5 * ccpy_einsum('mNEF,FEAIJN->AmIJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVOOO)
    )
    H.aa.vooo[Va, :, Oa, Oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnef,AefIJn->AmIJ', H.ab.oovv[:, ob, va, vb], T.aab.VvvOOo)
            + 1.0 * ccpy_einsum('mNef,AefIJN->AmIJ', H.ab.oovv[:, Ob, va, vb], T.aab.VvvOOO)
            - 1.0 * ccpy_einsum('mnEf,EAfIJn->AmIJ', H.ab.oovv[:, ob, Va, vb], T.aab.VVvOOo)
            - 1.0 * ccpy_einsum('mNEf,EAfIJN->AmIJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvOOO)
            + 1.0 * ccpy_einsum('mneF,AeFIJn->AmIJ', H.ab.oovv[:, ob, va, Vb], T.aab.VvVOOo)
            + 1.0 * ccpy_einsum('mNeF,AeFIJN->AmIJ', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVOOO)
            - 1.0 * ccpy_einsum('mnEF,EAFIJn->AmIJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VVVOOo)
            - 1.0 * ccpy_einsum('mNEF,EAFIJN->AmIJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVOOO)
    )
    # AmiJ
    H.aa.vooo[Va, :, oa, Oa] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,AfeinJ->AmiJ', H.aa.oovv[:, oa, va, va], T.aaa.VvvooO)
            - 0.5 * ccpy_einsum('mNef,AfeiJN->AmiJ', H.aa.oovv[:, Oa, va, va], T.aaa.VvvoOO)
            + 1.0 * ccpy_einsum('mnEf,EAfinJ->AmiJ', H.aa.oovv[:, oa, Va, va], T.aaa.VVvooO)
            - 1.0 * ccpy_einsum('mNEf,EAfiJN->AmiJ', H.aa.oovv[:, Oa, Va, va], T.aaa.VVvoOO)
            + 0.5 * ccpy_einsum('mnEF,FEAinJ->AmiJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVVooO)
            - 0.5 * ccpy_einsum('mNEF,FEAiJN->AmiJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVoOO)
    )
    H.aa.vooo[Va, :, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnef,AefiJn->AmiJ', H.ab.oovv[:, ob, va, vb], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('mNef,AefiJN->AmiJ', H.ab.oovv[:, Ob, va, vb], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('mneF,AeFiJn->AmiJ', H.ab.oovv[:, ob, va, Vb], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('mNeF,AeFiJN->AmiJ', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVoOO)
            - 1.0 * ccpy_einsum('mnEf,EAfiJn->AmiJ', H.ab.oovv[:, ob, Va, vb], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('mNEf,EAfiJN->AmiJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvoOO)
            - 1.0 * ccpy_einsum('mnEF,EAFiJn->AmiJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VVVoOo)
            - 1.0 * ccpy_einsum('mNEF,EAFiJN->AmiJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVoOO)
    )
    H.aa.vooo[Va, :, Oa, oa] = -1.0 * np.transpose(H.aa.vooo[Va, :, oa, Oa], (0, 1, 3, 2))
    # Amij
    H.aa.vooo[Va, :, oa, oa] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mNef,AfeijN->Amij', H.aa.oovv[:, Oa, va, va], T.aaa.VvvooO)
            - 1.0 * ccpy_einsum('mNEf,EAfijN->Amij', H.aa.oovv[:, Oa, Va, va], T.aaa.VVvooO)
            - 0.5 * ccpy_einsum('mNEF,FEAijN->Amij', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVooO)
    )
    H.aa.vooo[Va, :, oa, oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNef,AefijN->Amij', H.ab.oovv[:, Ob, va, vb], T.aab.VvvooO)
            + 1.0 * ccpy_einsum('mNeF,AeFijN->Amij', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVooO)
            - 1.0 * ccpy_einsum('mNEf,EAfijN->Amij', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvooO)
            - 1.0 * ccpy_einsum('mNEF,EAFijN->Amij', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVooO)
    )
    # amIJ
    H.aa.vooo[va, :, Oa, Oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnEf,EfanIJ->amIJ', H.aa.oovv[:, oa, Va, va], T.aaa.VvvoOO)
            - 0.5 * ccpy_einsum('mnEF,FEanIJ->amIJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVvoOO)
            + 1.0 * ccpy_einsum('mNEf,EfaIJN->amIJ', H.aa.oovv[:, Oa, Va, va], T.aaa.VvvOOO)
            - 0.5 * ccpy_einsum('mNEF,FEaIJN->amIJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvOOO)
    )
    H.aa.vooo[va, :, Oa, Oa] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,eaFIJn->amIJ', H.ab.oovv[:, ob, va, Vb], T.aab.vvVOOo)
            - 1.0 * ccpy_einsum('mnEf,EafIJn->amIJ', H.ab.oovv[:, ob, Va, vb], T.aab.VvvOOo)
            - 1.0 * ccpy_einsum('mnEF,EaFIJn->amIJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VvVOOo)
            - 1.0 * ccpy_einsum('mNeF,eaFIJN->amIJ', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVOOO)
            - 1.0 * ccpy_einsum('mNEf,EafIJN->amIJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvOOO)
            - 1.0 * ccpy_einsum('mNEF,EaFIJN->amIJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVOOO)
    )
    # amIj
    H.aa.vooo[va, :, Oa, oa] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,FeajnI->amIj', H.aa.oovv[:, oa, va, Va], T.aaa.VvvooO)
            - 0.5 * ccpy_einsum('mnEF,FEajnI->amIj', H.aa.oovv[:, oa, Va, Va], T.aaa.VVvooO)
            + 1.0 * ccpy_einsum('mNeF,FeajIN->amIj', H.aa.oovv[:, Oa, va, Va], T.aaa.VvvoOO)
            + 0.5 * ccpy_einsum('mNEF,FEajIN->amIj', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvoOO)
    )
    H.aa.vooo[va, :, Oa, oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnEf,EafjIn->amIj', H.ab.oovv[:, ob, Va, vb], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('mneF,eaFjIn->amIj', H.ab.oovv[:, ob, va, Vb], T.aab.vvVoOo)
            + 1.0 * ccpy_einsum('mnEF,EaFjIn->amIj', H.ab.oovv[:, ob, Va, Vb], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('mNEf,EafjIN->amIj', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('mNeF,eaFjIN->amIj', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVoOO)
            + 1.0 * ccpy_einsum('mNEF,EaFjIN->amIj', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVoOO)
    )
    H.aa.vooo[va, :, oa, Oa] = -1.0 * np.transpose(H.aa.vooo[va, :, Oa, oa], (0, 1, 3, 2))
    # amij
    H.aa.vooo[va, :, oa, oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNEf,EfaijN->amij', H.aa.oovv[:, Oa, Va, va], T.aaa.VvvooO)
            - 0.5 * ccpy_einsum('mNEF,FEaijN->amij', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvooO)
    )
    H.aa.vooo[va, :, oa, oa] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mNeF,eaFijN->amij', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVooO)
            - 1.0 * ccpy_einsum('mNEf,EafijN->amij', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvooO)
            - 1.0 * ccpy_einsum('mNEF,EaFijN->amij', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVooO)
    )
    
    ##### H.aa.vvov #####
    # ABIe
    H.aa.vvov[Va, Va, Oa, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,BAfmnI->ABIe', H.aa.oovv[oa, oa, :, va], T.aaa.VVvooO)
            + 0.5 * ccpy_einsum('mneF,FBAmnI->ABIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VVVooO)
            - 1.0 * ccpy_einsum('mNef,BAfmIN->ABIe', H.aa.oovv[oa, Oa, :, va], T.aaa.VVvoOO)
            - 1.0 * ccpy_einsum('mNeF,FBAmIN->ABIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVVoOO)
            + 0.5 * ccpy_einsum('MNef,BAfIMN->ABIe', H.aa.oovv[Oa, Oa, :, va], T.aaa.VVvOOO)
            + 0.5 * ccpy_einsum('MNeF,FBAIMN->ABIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVVOOO)
    )
    H.aa.vvov[Va, Va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnef,BAfmIn->ABIe', H.ab.oovv[oa, ob, :, vb], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('mneF,BAFmIn->ABIe', H.ab.oovv[oa, ob, :, Vb], T.aab.VVVoOo)
            + 1.0 * ccpy_einsum('Mnef,BAfIMn->ABIe', H.ab.oovv[Oa, ob, :, vb], T.aab.VVvOOo)
            + 1.0 * ccpy_einsum('MneF,BAFIMn->ABIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.VVVOOo)
            - 1.0 * ccpy_einsum('mNef,BAfmIN->ABIe', H.ab.oovv[oa, Ob, :, vb], T.aab.VVvoOO)
            - 1.0 * ccpy_einsum('mNeF,BAFmIN->ABIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.VVVoOO)
            + 1.0 * ccpy_einsum('MNef,BAfIMN->ABIe', H.ab.oovv[Oa, Ob, :, vb], T.aab.VVvOOO)
            + 1.0 * ccpy_einsum('MNeF,BAFIMN->ABIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VVVOOO)
    )
    # aBIe
    H.aa.vvov[va, Va, Oa, :] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,BfamnI->aBIe', H.aa.oovv[oa, oa, :, va], T.aaa.VvvooO)
            + 0.5 * ccpy_einsum('mneF,FBamnI->aBIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VVvooO)
            + 1.0 * ccpy_einsum('mNef,BfamIN->aBIe', H.aa.oovv[oa, Oa, :, va], T.aaa.VvvoOO)
            - 1.0 * ccpy_einsum('mNeF,FBamIN->aBIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVvoOO)
            - 0.5 * ccpy_einsum('MNef,BfaIMN->aBIe', H.aa.oovv[Oa, Oa, :, va], T.aaa.VvvOOO)
            + 0.5 * ccpy_einsum('MNeF,FBaIMN->aBIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVvOOO)
    )
    H.aa.vvov[va, Va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnef,BafmIn->aBIe', H.ab.oovv[oa, ob, :, vb], T.aab.VvvoOo)
            - 1.0 * ccpy_einsum('mneF,BaFmIn->aBIe', H.ab.oovv[oa, ob, :, Vb], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('Mnef,BafIMn->aBIe', H.ab.oovv[Oa, ob, :, vb], T.aab.VvvOOo)
            + 1.0 * ccpy_einsum('MneF,BaFIMn->aBIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.VvVOOo)
            - 1.0 * ccpy_einsum('mNef,BafmIN->aBIe', H.ab.oovv[oa, Ob, :, vb], T.aab.VvvoOO)
            - 1.0 * ccpy_einsum('mNeF,BaFmIN->aBIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.VvVoOO)
            + 1.0 * ccpy_einsum('MNef,BafIMN->aBIe', H.ab.oovv[Oa, Ob, :, vb], T.aab.VvvOOO)
            + 1.0 * ccpy_einsum('MNeF,BaFIMN->aBIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VvVOOO)
    )
    H.aa.vvov[Va, va, Oa, :] = -1.0 * np.transpose(H.aa.vvov[va, Va, Oa, :], (1, 0, 2, 3))
    # abIe
    H.aa.vvov[va, va, Oa, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mneF,FbamnI->abIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VvvooO)
            - 1.0 * ccpy_einsum('mNeF,FbamIN->abIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VvvoOO)
            + 0.5 * ccpy_einsum('MNeF,FbaIMN->abIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VvvOOO)
    )
    H.aa.vvov[va, va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,baFmIn->abIe', H.ab.oovv[oa, ob, :, Vb], T.aab.vvVoOo)
            + 1.0 * ccpy_einsum('MneF,baFIMn->abIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.vvVOOo)
            - 1.0 * ccpy_einsum('mNeF,baFmIN->abIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.vvVoOO)
            + 1.0 * ccpy_einsum('MNeF,baFIMN->abIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.vvVOOO)
    )
    # ABie
    H.aa.vvov[Va, Va, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNef,BAfimN->ABie', H.aa.oovv[oa, Oa, :, va], T.aaa.VVvooO)
            + 1.0 * ccpy_einsum('mNeF,FBAimN->ABie', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVVooO)
            + 0.5 * ccpy_einsum('MNef,BAfiMN->ABie', H.aa.oovv[Oa, Oa, :, va], T.aaa.VVvoOO)
            + 0.5 * ccpy_einsum('MNeF,FBAiMN->ABie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVVoOO)
    )
    H.aa.vvov[Va, Va, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Mnef,BAfiMn->ABie', H.ab.oovv[Oa, ob, :, vb], T.aab.VVvoOo)
            + 1.0 * ccpy_einsum('MneF,BAFiMn->ABie', H.ab.oovv[Oa, ob, :, Vb], T.aab.VVVoOo)
            + 1.0 * ccpy_einsum('mNef,BAfimN->ABie', H.ab.oovv[oa, Ob, :, vb], T.aab.VVvooO)
            + 1.0 * ccpy_einsum('mNeF,BAFimN->ABie', H.ab.oovv[oa, Ob, :, Vb], T.aab.VVVooO)
            + 1.0 * ccpy_einsum('MNef,BAfiMN->ABie', H.ab.oovv[Oa, Ob, :, vb], T.aab.VVvoOO)
            + 1.0 * ccpy_einsum('MNeF,BAFiMN->ABie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VVVoOO)
    )
    # Abie
    H.aa.vvov[Va, va, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNef,AfbimN->Abie', H.aa.oovv[oa, Oa, :, va], T.aaa.VvvooO)
            + 0.5 * ccpy_einsum('MNef,AfbiMN->Abie', H.aa.oovv[Oa, Oa, :, va], T.aaa.VvvoOO)
            - 1.0 * ccpy_einsum('mNeF,FAbimN->Abie', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVvooO)
            - 0.5 * ccpy_einsum('MNeF,FAbiMN->Abie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVvoOO)
    )
    H.aa.vvov[Va, va, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Mnef,AbfiMn->Abie', H.ab.oovv[Oa, ob, :, vb], T.aab.VvvoOo)
            - 1.0 * ccpy_einsum('mNef,AbfimN->Abie', H.ab.oovv[oa, Ob, :, vb], T.aab.VvvooO)
            - 1.0 * ccpy_einsum('MNef,AbfiMN->Abie', H.ab.oovv[Oa, Ob, :, vb], T.aab.VvvoOO)
            - 1.0 * ccpy_einsum('MneF,AbFiMn->Abie', H.ab.oovv[Oa, ob, :, Vb], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('mNeF,AbFimN->Abie', H.ab.oovv[oa, Ob, :, Vb], T.aab.VvVooO)
            - 1.0 * ccpy_einsum('MNeF,AbFiMN->Abie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VvVoOO)
    )
    H.aa.vvov[va, Va, oa, :] = -1.0 * np.transpose(H.aa.vvov[Va, va, oa, :], (1, 0, 2, 3))
    # abie
    H.aa.vvov[va, va, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('MneF,FbainM->abie', H.aa.oovv[Oa, oa, :, Va], T.aaa.VvvooO)
            + 0.5 * ccpy_einsum('MNeF,FbaiMN->abie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VvvoOO)
    )
    H.aa.vvov[va, va, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNeF,baFimN->abie', H.ab.oovv[oa, Ob, :, Vb], T.aab.vvVooO)
            + 1.0 * ccpy_einsum('MneF,baFiMn->abie', H.ab.oovv[Oa, ob, :, Vb], T.aab.vvVoOo)
            + 1.0 * ccpy_einsum('MNeF,baFiMN->abie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.vvVoOO)
    )
    
    #########################################################################
    ############################ ab intermdiates ############################
    #########################################################################
    
    ##### H.ab.ovoo #####
    # mBIJ
    H.ab.ovoo[:, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,feBnIJ->mBIJ', H.aa.oovv[:, oa, va, va], T.aab.vvVoOO)
            - 1.0 * ccpy_einsum('mnEf,EfBnIJ->mBIJ', H.aa.oovv[:, oa, Va, va], T.aab.VvVoOO)
            + 0.5 * ccpy_einsum('mnEF,FEBnIJ->mBIJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVVoOO)
            - 0.5 * ccpy_einsum('mNef,feBINJ->mBIJ', H.aa.oovv[:, Oa, va, va], T.aab.vvVOOO)
            + 1.0 * ccpy_einsum('mNEf,EfBINJ->mBIJ', H.aa.oovv[:, Oa, Va, va], T.aab.VvVOOO)
            - 0.5 * ccpy_einsum('mNEF,FEBINJ->mBIJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVOOO)
    )
    H.ab.ovoo[:, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnef,eBfInJ->mBIJ', H.ab.oovv[:, ob, va, vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('mneF,eBFInJ->mBIJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVVOoO)
            - 1.0 * ccpy_einsum('mnEf,EBfInJ->mBIJ', H.ab.oovv[:, ob, Va, vb], T.abb.VVvOoO)
            - 1.0 * ccpy_einsum('mnEF,EBFInJ->mBIJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVOoO)
            - 1.0 * ccpy_einsum('mNef,eBfINJ->mBIJ', H.ab.oovv[:, Ob, va, vb], T.abb.vVvOOO)
            - 1.0 * ccpy_einsum('mNeF,eBFINJ->mBIJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVOOO)
            - 1.0 * ccpy_einsum('mNEf,EBfINJ->mBIJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvOOO)
            - 1.0 * ccpy_einsum('mNEF,EBFINJ->mBIJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVOOO)
    )
    # mBiJ
    H.ab.ovoo[:, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,feBinJ->mBiJ', H.aa.oovv[:, oa, va, va], T.aab.vvVooO)
            - 0.5 * ccpy_einsum('mNef,feBiNJ->mBiJ', H.aa.oovv[:, Oa, va, va], T.aab.vvVoOO)
            - 1.0 * ccpy_einsum('mneF,FeBinJ->mBiJ', H.aa.oovv[:, oa, va, Va], T.aab.VvVooO)
            - 0.5 * ccpy_einsum('mnEF,FEBinJ->mBiJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVVooO)
            - 1.0 * ccpy_einsum('mNeF,FeBiNJ->mBiJ', H.aa.oovv[:, Oa, va, Va], T.aab.VvVoOO)
            - 0.5 * ccpy_einsum('mNEF,FEBiNJ->mBiJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVoOO)
    )
    H.ab.ovoo[:, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnef,eBfinJ->mBiJ', H.ab.oovv[:, ob, va, vb], T.abb.vVvooO)
            - 1.0 * ccpy_einsum('mnEf,EBfinJ->mBiJ', H.ab.oovv[:, ob, Va, vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('mNef,eBfiNJ->mBiJ', H.ab.oovv[:, Ob, va, vb], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('mNEf,EBfiNJ->mBiJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvoOO)
            - 1.0 * ccpy_einsum('mneF,eBFinJ->mBiJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVVooO)
            - 1.0 * ccpy_einsum('mnEF,EBFinJ->mBiJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVooO)
            - 1.0 * ccpy_einsum('mNeF,eBFiNJ->mBiJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVoOO)
            - 1.0 * ccpy_einsum('mNEF,EBFiNJ->mBiJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVoOO)
    )
    # mBIj
    H.ab.ovoo[:, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,feBnIj->mBIj', H.aa.oovv[:, oa, va, va], T.aab.vvVoOo)
            - 0.5 * ccpy_einsum('mNef,feBINj->mBIj', H.aa.oovv[:, Oa, va, va], T.aab.vvVOOo)
            - 1.0 * ccpy_einsum('mnEf,EfBnIj->mBIj', H.aa.oovv[:, oa, Va, va], T.aab.VvVoOo)
            + 0.5 * ccpy_einsum('mnEF,FEBnIj->mBIj', H.aa.oovv[:, oa, Va, Va], T.aab.VVVoOo)
            + 1.0 * ccpy_einsum('mNEf,EfBINj->mBIj', H.aa.oovv[:, Oa, Va, va], T.aab.VvVOOo)
            - 0.5 * ccpy_einsum('mNEF,FEBINj->mBIj', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVOOo)
    )
    H.ab.ovoo[:, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnef,eBfInj->mBIj', H.ab.oovv[:, ob, va, vb], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('mneF,eBFInj->mBIj', H.ab.oovv[:, ob, va, Vb], T.abb.vVVOoo)
            + 1.0 * ccpy_einsum('mNef,eBfIjN->mBIj', H.ab.oovv[:, Ob, va, vb], T.abb.vVvOoO)
            + 1.0 * ccpy_einsum('mNeF,eBFIjN->mBIj', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVOoO)
            - 1.0 * ccpy_einsum('mnEf,EBfInj->mBIj', H.ab.oovv[:, ob, Va, vb], T.abb.VVvOoo)
            - 1.0 * ccpy_einsum('mnEF,EBFInj->mBIj', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVOoo)
            + 1.0 * ccpy_einsum('mNEf,EBfIjN->mBIj', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvOoO)
            + 1.0 * ccpy_einsum('mNEF,EBFIjN->mBIj', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVOoO)
    )
    # mBij
    H.ab.ovoo[:, Vb, oa, ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mNef,feBiNj->mBij', H.aa.oovv[:, Oa, va, va], T.aab.vvVoOo)
            - 1.0 * ccpy_einsum('mNeF,FeBiNj->mBij', H.aa.oovv[:, Oa, va, Va], T.aab.VvVoOo)
            - 0.5 * ccpy_einsum('mNEF,FEBiNj->mBij', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVoOo)
    )
    H.ab.ovoo[:, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNef,eBfijN->mBij', H.ab.oovv[:, Ob, va, vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('mNEf,EBfijN->mBij', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvooO)
            + 1.0 * ccpy_einsum('mNeF,eBFijN->mBij', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVooO)
            + 1.0 * ccpy_einsum('mNEF,EBFijN->mBij', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVooO)
    )
    # mbIJ
    H.ab.ovoo[:, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnEf,EfbnIJ->mbIJ', H.aa.oovv[:, oa, Va, va], T.aab.VvvoOO)
            + 0.5 * ccpy_einsum('mnEF,FEbnIJ->mbIJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVvoOO)
            + 1.0 * ccpy_einsum('mNEf,EfbINJ->mbIJ', H.aa.oovv[:, Oa, Va, va], T.aab.VvvOOO)
            - 0.5 * ccpy_einsum('mNEF,FEbINJ->mbIJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvOOO)
    )
    H.ab.ovoo[:, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mneF,eFbInJ->mbIJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('mnEf,EbfInJ->mbIJ', H.ab.oovv[:, ob, Va, vb], T.abb.VvvOoO)
            + 1.0 * ccpy_einsum('mnEF,EFbInJ->mbIJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvOoO)
            + 1.0 * ccpy_einsum('mNeF,eFbINJ->mbIJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvOOO)
            - 1.0 * ccpy_einsum('mNEf,EbfINJ->mbIJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvOOO)
            + 1.0 * ccpy_einsum('mNEF,EFbINJ->mbIJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvOOO)
    )
    # mbiJ
    H.ab.ovoo[:, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,FebinJ->mbiJ', H.aa.oovv[:, oa, va, Va], T.aab.VvvooO)
            - 1.0 * ccpy_einsum('mNeF,FebiNJ->mbiJ', H.aa.oovv[:, Oa, va, Va], T.aab.VvvoOO)
            - 0.5 * ccpy_einsum('mnEF,FEbinJ->mbiJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVvooO)
            - 0.5 * ccpy_einsum('mNEF,FEbiNJ->mbiJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvoOO)
    )
    H.ab.ovoo[:, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnEf,EbfinJ->mbiJ', H.ab.oovv[:, ob, Va, vb], T.abb.VvvooO)
            - 1.0 * ccpy_einsum('mNEf,EbfiNJ->mbiJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvoOO)
            + 1.0 * ccpy_einsum('mneF,eFbinJ->mbiJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('mNeF,eFbiNJ->mbiJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvoOO)
            + 1.0 * ccpy_einsum('mnEF,EFbinJ->mbiJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvooO)
            + 1.0 * ccpy_einsum('mNEF,EFbiNJ->mbiJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvoOO)
    )
    # mbIj
    H.ab.ovoo[:, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnEf,EfbnIj->mbIj', H.aa.oovv[:, oa, Va, va], T.aab.VvvoOo)
            + 0.5 * ccpy_einsum('mnEF,FEbnIj->mbIj', H.aa.oovv[:, oa, Va, Va], T.aab.VVvoOo)
            + 1.0 * ccpy_einsum('mNEf,EfbINj->mbIj', H.aa.oovv[:, Oa, Va, va], T.aab.VvvOOo)
            - 0.5 * ccpy_einsum('mNEF,FEbINj->mbIj', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvOOo)
    )
    H.ab.ovoo[:, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mneF,eFbInj->mbIj', H.ab.oovv[:, ob, va, Vb], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('mNeF,eFbIjN->mbIj', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('mnEf,EbfInj->mbIj', H.ab.oovv[:, ob, Va, vb], T.abb.VvvOoo)
            + 1.0 * ccpy_einsum('mnEF,EFbInj->mbIj', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvOoo)
            + 1.0 * ccpy_einsum('mNEf,EbfIjN->mbIj', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvOoO)
            - 1.0 * ccpy_einsum('mNEF,EFbIjN->mbIj', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvOoO)
    )
    # mbij
    H.ab.ovoo[:, vb, oa, oa] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNEf,EfbiNj->mbij', H.aa.oovv[:, Oa, Va, va], T.aab.VvvoOo)
            - 0.5 * ccpy_einsum('mNEF,FEbiNj->mbij', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvoOo)
    )
    H.ab.ovoo[:, vb, oa, oa] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mNeF,eFbijN->mbij', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('mNEf,EbfijN->mbij', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvooO)
            - 1.0 * ccpy_einsum('mNEF,EFbijN->mbij', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvooO)
    )
    
    ##### H.ab.vvvo #####
    # ABeJ
    H.ab.vvvo[Va, Vb, :, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,AfBmnJ->ABeJ', H.aa.oovv[oa, oa, :, va], T.aab.VvVooO)
            + 0.5 * ccpy_einsum('mneF,FABmnJ->ABeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VVVooO)
            + 1.0 * ccpy_einsum('Mnef,AfBnMJ->ABeJ', H.aa.oovv[Oa, oa, :, va], T.aab.VvVoOO)
            - 1.0 * ccpy_einsum('MneF,FABnMJ->ABeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VVVoOO)
            - 0.5 * ccpy_einsum('MNef,AfBMNJ->ABeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.VvVOOO)
            + 0.5 * ccpy_einsum('MNeF,FABMNJ->ABeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVVOOO)
    )
    H.ab.vvvo[Va, Vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnef,ABfmnJ->ABeJ', H.ab.oovv[oa, ob, :, vb], T.abb.VVvooO)
            + 1.0 * ccpy_einsum('mneF,ABFmnJ->ABeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.VVVooO)
            + 1.0 * ccpy_einsum('mNef,ABfmNJ->ABeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.VVvoOO)
            + 1.0 * ccpy_einsum('mNeF,ABFmNJ->ABeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVVoOO)
            + 1.0 * ccpy_einsum('Mnef,ABfMnJ->ABeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.VVvOoO)
            + 1.0 * ccpy_einsum('MneF,ABFMnJ->ABeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVVOoO)
            + 1.0 * ccpy_einsum('MNef,ABfMNJ->ABeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.VVvOOO)
            + 1.0 * ccpy_einsum('MNeF,ABFMNJ->ABeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVVOOO)
    )
    # AbeJ
    H.ab.vvvo[Va, vb, :, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,AfbmnJ->AbeJ', H.aa.oovv[oa, oa, :, va], T.aab.VvvooO)
            + 0.5 * ccpy_einsum('mneF,FAbmnJ->AbeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VVvooO)
            + 1.0 * ccpy_einsum('Mnef,AfbnMJ->AbeJ', H.aa.oovv[Oa, oa, :, va], T.aab.VvvoOO)
            - 0.5 * ccpy_einsum('MNef,AfbMNJ->AbeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.VvvOOO)
            - 1.0 * ccpy_einsum('MneF,FAbnMJ->AbeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VVvoOO)
            + 0.5 * ccpy_einsum('MNeF,FAbMNJ->AbeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVvOOO)
    )
    H.ab.vvvo[Va, vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnef,AbfmnJ->AbeJ', H.ab.oovv[oa, ob, :, vb], T.abb.VvvooO)
            + 1.0 * ccpy_einsum('mNef,AbfmNJ->AbeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.VvvoOO)
            - 1.0 * ccpy_einsum('mneF,AFbmnJ->AbeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('mNeF,AFbmNJ->AbeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVvoOO)
            + 1.0 * ccpy_einsum('Mnef,AbfMnJ->AbeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.VvvOoO)
            + 1.0 * ccpy_einsum('MNef,AbfMNJ->AbeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.VvvOOO)
            - 1.0 * ccpy_einsum('MneF,AFbMnJ->AbeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVvOoO)
            - 1.0 * ccpy_einsum('MNeF,AFbMNJ->AbeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVvOOO)
    )
    # aBeJ
    H.ab.vvvo[va, Vb, :, Ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,faBmnJ->aBeJ', H.aa.oovv[oa, oa, :, va], T.aab.vvVooO)
            - 1.0 * ccpy_einsum('Mnef,faBnMJ->aBeJ', H.aa.oovv[Oa, oa, :, va], T.aab.vvVoOO)
            + 0.5 * ccpy_einsum('MNef,faBMNJ->aBeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.vvVOOO)
            + 0.5 * ccpy_einsum('mneF,FaBmnJ->aBeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VvVooO)
            - 1.0 * ccpy_einsum('MneF,FaBnMJ->aBeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VvVoOO)
            + 0.5 * ccpy_einsum('MNeF,FaBMNJ->aBeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvVOOO)
    )
    H.ab.vvvo[va, Vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mnef,aBfmnJ->aBeJ', H.ab.oovv[oa, ob, :, vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('mNef,aBfmNJ->aBeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.vVvoOO)
            + 1.0 * ccpy_einsum('Mnef,aBfMnJ->aBeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.vVvOoO)
            + 1.0 * ccpy_einsum('MNef,aBfMNJ->aBeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.vVvOOO)
            + 1.0 * ccpy_einsum('mneF,aBFmnJ->aBeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.vVVooO)
            + 1.0 * ccpy_einsum('mNeF,aBFmNJ->aBeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVVoOO)
            + 1.0 * ccpy_einsum('MneF,aBFMnJ->aBeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVVOoO)
            + 1.0 * ccpy_einsum('MNeF,aBFMNJ->aBeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVVOOO)
    )
    # abeJ
    H.ab.vvvo[va, vb, :, Ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mneF,FabmnJ->abeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VvvooO)
            - 1.0 * ccpy_einsum('MneF,FabnMJ->abeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VvvoOO)
            + 0.5 * ccpy_einsum('MNeF,FabMNJ->abeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvvOOO)
    )
    H.ab.vvvo[va, vb, :, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,aFbmnJ->abeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.vVvooO)
            - 1.0 * ccpy_einsum('mNeF,aFbmNJ->abeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('MneF,aFbMnJ->abeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('MNeF,aFbMNJ->abeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVvOOO)
    )
    # ABej
    H.ab.vvvo[Va, Vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Mnef,AfBnMj->ABej', H.aa.oovv[Oa, oa, :, va], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('MneF,FABnMj->ABej', H.aa.oovv[Oa, oa, :, Va], T.aab.VVVoOo)
            - 0.5 * ccpy_einsum('MNef,AfBMNj->ABej', H.aa.oovv[Oa, Oa, :, va], T.aab.VvVOOo)
            + 0.5 * ccpy_einsum('MNeF,FABMNj->ABej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVVOOo)
    )
    H.ab.vvvo[Va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mNef,ABfmjN->ABej', H.ab.oovv[oa, Ob, :, vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('mNeF,ABFmjN->ABej', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVVooO)
            + 1.0 * ccpy_einsum('Mnef,ABfMnj->ABej', H.ab.oovv[Oa, ob, :, vb], T.abb.VVvOoo)
            + 1.0 * ccpy_einsum('MneF,ABFMnj->ABej', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVVOoo)
            - 1.0 * ccpy_einsum('MNef,ABfMjN->ABej', H.ab.oovv[Oa, Ob, :, vb], T.abb.VVvOoO)
            - 1.0 * ccpy_einsum('MNeF,ABFMjN->ABej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVVOoO)
    )
    # Abej
    H.ab.vvvo[Va, vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Mnef,AfbnMj->Abej', H.aa.oovv[Oa, oa, :, va], T.aab.VvvoOo)
            - 0.5 * ccpy_einsum('MNef,AfbMNj->Abej', H.aa.oovv[Oa, Oa, :, va], T.aab.VvvOOo)
            - 1.0 * ccpy_einsum('MneF,FAbnMj->Abej', H.aa.oovv[Oa, oa, :, Va], T.aab.VVvoOo)
            + 0.5 * ccpy_einsum('MNeF,FAbMNj->Abej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVvOOo)
    )
    H.ab.vvvo[Va, vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mNef,AbfmjN->Abej', H.ab.oovv[oa, Ob, :, vb], T.abb.VvvooO)
            + 1.0 * ccpy_einsum('Mnef,AbfMnj->Abej', H.ab.oovv[Oa, ob, :, vb], T.abb.VvvOoo)
            - 1.0 * ccpy_einsum('MNef,AbfMjN->Abej', H.ab.oovv[Oa, Ob, :, vb], T.abb.VvvOoO)
            + 1.0 * ccpy_einsum('mNeF,AFbmjN->Abej', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('MneF,AFbMnj->Abej', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVvOoo)
            + 1.0 * ccpy_einsum('MNeF,AFbMjN->Abej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVvOoO)
    )
    # aBej
    H.ab.vvvo[va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Mnef,faBnMj->aBej', H.aa.oovv[Oa, oa, :, va], T.aab.vvVoOo)
            + 0.5 * ccpy_einsum('MNef,faBMNj->aBej', H.aa.oovv[Oa, Oa, :, va], T.aab.vvVOOo)
            - 1.0 * ccpy_einsum('MneF,FaBnMj->aBej', H.aa.oovv[Oa, oa, :, Va], T.aab.VvVoOo)
            + 0.5 * ccpy_einsum('MNeF,FaBMNj->aBej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvVOOo)
    )
    H.ab.vvvo[va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mNef,aBfmjN->aBej', H.ab.oovv[oa, Ob, :, vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('Mnef,aBfMnj->aBej', H.ab.oovv[Oa, ob, :, vb], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('MNef,aBfMjN->aBej', H.ab.oovv[Oa, Ob, :, vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('mNeF,aBFmjN->aBej', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVVooO)
            + 1.0 * ccpy_einsum('MneF,aBFMnj->aBej', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVVOoo)
            - 1.0 * ccpy_einsum('MNeF,aBFMjN->aBej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVVOoO)
    )
    # abej
    H.ab.vvvo[va, vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNeF,FabmNj->abej', H.aa.oovv[oa, Oa, :, Va], T.aab.VvvoOo)
            + 0.5 * ccpy_einsum('MNeF,FabMNj->abej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvvOOo)
    )
    H.ab.vvvo[va, vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('MneF,aFbMnj->abej', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVvOoo)
            + 1.0 * ccpy_einsum('mNeF,aFbmjN->abej', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('MNeF,aFbMjN->abej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVvOoO)
    )
    
    ##### H.ab.vooo #####
    # AmIJ
    H.ab.vooo[Va, :, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfe,AfenIJ->AmIJ', H.ab.oovv[oa, :, va, vb], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('nmFe,FAenIJ->AmIJ', H.ab.oovv[oa, :, Va, vb], T.aab.VVvoOO)
            - 1.0 * ccpy_einsum('nmfE,AfEnIJ->AmIJ', H.ab.oovv[oa, :, va, Vb], T.aab.VvVoOO)
            + 1.0 * ccpy_einsum('nmFE,FAEnIJ->AmIJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVoOO)
            + 1.0 * ccpy_einsum('Nmfe,AfeINJ->AmIJ', H.ab.oovv[Oa, :, va, vb], T.aab.VvvOOO)
            - 1.0 * ccpy_einsum('NmFe,FAeINJ->AmIJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvOOO)
            + 1.0 * ccpy_einsum('NmfE,AfEINJ->AmIJ', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVOOO)
            - 1.0 * ccpy_einsum('NmFE,FAEINJ->AmIJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVOOO)
    )
    H.ab.vooo[Va, :, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('nmfe,AefInJ->AmIJ', H.bb.oovv[ob, :, vb, vb], T.abb.VvvOoO)
            - 1.0 * ccpy_einsum('nmfE,AEfInJ->AmIJ', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvOoO)
            - 0.5 * ccpy_einsum('nmFE,AEFInJ->AmIJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVOoO)
            - 0.5 * ccpy_einsum('Nmfe,AefINJ->AmIJ', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvOOO)
            - 1.0 * ccpy_einsum('NmfE,AEfINJ->AmIJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvOOO)
            - 0.5 * ccpy_einsum('NmFE,AEFINJ->AmIJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVOOO)
    )
    # AmiJ
    H.ab.vooo[Va, :, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,AfeinJ->AmiJ', H.ab.oovv[oa, :, va, vb], T.aab.VvvooO)
            + 1.0 * ccpy_einsum('Nmfe,AfeiNJ->AmiJ', H.ab.oovv[Oa, :, va, vb], T.aab.VvvoOO)
            - 1.0 * ccpy_einsum('nmFe,FAeinJ->AmiJ', H.ab.oovv[oa, :, Va, vb], T.aab.VVvooO)
            - 1.0 * ccpy_einsum('NmFe,FAeiNJ->AmiJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvoOO)
            + 1.0 * ccpy_einsum('nmfE,AfEinJ->AmiJ', H.ab.oovv[oa, :, va, Vb], T.aab.VvVooO)
            + 1.0 * ccpy_einsum('NmfE,AfEiNJ->AmiJ', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVoOO)
            - 1.0 * ccpy_einsum('nmFE,FAEinJ->AmiJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVooO)
            - 1.0 * ccpy_einsum('NmFE,FAEiNJ->AmiJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVoOO)
    )
    H.ab.vooo[Va, :, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('nmfe,AefinJ->AmiJ', H.bb.oovv[ob, :, vb, vb], T.abb.VvvooO)
            - 0.5 * ccpy_einsum('Nmfe,AefiNJ->AmiJ', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvoOO)
            - 1.0 * ccpy_einsum('nmfE,AEfinJ->AmiJ', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('NmfE,AEfiNJ->AmiJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvoOO)
            - 0.5 * ccpy_einsum('nmFE,AEFinJ->AmiJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVooO)
            - 0.5 * ccpy_einsum('NmFE,AEFiNJ->AmiJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVoOO)
    )
    # AmIj
    H.ab.vooo[Va, :, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfe,AfenIj->AmIj', H.ab.oovv[oa, :, va, vb], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('nmFe,FAenIj->AmIj', H.ab.oovv[oa, :, Va, vb], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('nmfE,AfEnIj->AmIj', H.ab.oovv[oa, :, va, Vb], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('nmFE,FAEnIj->AmIj', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVoOo)
            + 1.0 * ccpy_einsum('Nmfe,AfeINj->AmIj', H.ab.oovv[Oa, :, va, vb], T.aab.VvvOOo)
            - 1.0 * ccpy_einsum('NmFe,FAeINj->AmIj', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvOOo)
            + 1.0 * ccpy_einsum('NmfE,AfEINj->AmIj', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVOOo)
            - 1.0 * ccpy_einsum('NmFE,FAEINj->AmIj', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVOOo)
    )
    H.ab.vooo[Va, :, Oa, ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('nmfe,AefInj->AmIj', H.bb.oovv[ob, :, vb, vb], T.abb.VvvOoo)
            - 1.0 * ccpy_einsum('nmfE,AEfInj->AmIj', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvOoo)
            - 0.5 * ccpy_einsum('nmFE,AEFInj->AmIj', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVOoo)
            + 0.5 * ccpy_einsum('Nmfe,AefIjN->AmIj', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvOoO)
            + 1.0 * ccpy_einsum('NmfE,AEfIjN->AmIj', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvOoO)
            + 0.5 * ccpy_einsum('NmFE,AEFIjN->AmIj', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVOoO)
    )
    # Amij
    H.ab.vooo[Va, :, oa, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Nmfe,AfeiNj->Amij', H.ab.oovv[Oa, :, va, vb], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('NmfE,AfEiNj->Amij', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('NmFe,FAeiNj->Amij', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('NmFE,FAEiNj->Amij', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVoOo)
    )
    H.ab.vooo[Va, :, oa, ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('Nmfe,AefijN->Amij', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvooO)
            - 1.0 * ccpy_einsum('NmFe,AFeijN->Amij', H.bb.oovv[Ob, :, Vb, vb], T.abb.VVvooO)
            + 0.5 * ccpy_einsum('NmFE,AEFijN->Amij', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVooO)
    )
    # amIJ
    H.ab.vooo[va, :, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmFe,FaenIJ->amIJ', H.ab.oovv[oa, :, Va, vb], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('nmfE,faEnIJ->amIJ', H.ab.oovv[oa, :, va, Vb], T.aab.vvVoOO)
            + 1.0 * ccpy_einsum('nmFE,FaEnIJ->amIJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVoOO)
            - 1.0 * ccpy_einsum('NmFe,FaeINJ->amIJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvOOO)
            - 1.0 * ccpy_einsum('NmfE,faEINJ->amIJ', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVOOO)
            - 1.0 * ccpy_einsum('NmFE,FaEINJ->amIJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVOOO)
    )
    H.ab.vooo[va, :, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfE,aEfInJ->amIJ', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvOoO)
            - 0.5 * ccpy_einsum('nmFE,aEFInJ->amIJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVOoO)
            - 1.0 * ccpy_einsum('NmfE,aEfINJ->amIJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvOOO)
            - 0.5 * ccpy_einsum('NmFE,aEFINJ->amIJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVOOO)
    )
    # amIj
    H.ab.vooo[va, :, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmFe,FaenIj->amIj', H.ab.oovv[oa, :, Va, vb], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('nmfE,faEnIj->amIj', H.ab.oovv[oa, :, va, Vb], T.aab.vvVoOo)
            + 1.0 * ccpy_einsum('nmFE,FaEnIj->amIj', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('NmFe,FaeINj->amIj', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvOOo)
            - 1.0 * ccpy_einsum('NmfE,faEINj->amIj', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVOOo)
            - 1.0 * ccpy_einsum('NmFE,FaEINj->amIj', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVOOo)
    )
    H.ab.vooo[va, :, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfE,aEfInj->amIj', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvOoo)
            - 0.5 * ccpy_einsum('nmFE,aEFInj->amIj', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVOoo)
            + 1.0 * ccpy_einsum('NmfE,aEfIjN->amIj', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvOoO)
            + 0.5 * ccpy_einsum('NmFE,aEFIjN->amIj', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVOoO)
    )
    # amiJ
    H.ab.vooo[va, :, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmFe,FaeinJ->amiJ', H.ab.oovv[oa, :, Va, vb], T.aab.VvvooO)
            - 1.0 * ccpy_einsum('nmfE,faEinJ->amiJ', H.ab.oovv[oa, :, va, Vb], T.aab.vvVooO)
            - 1.0 * ccpy_einsum('nmFE,FaEinJ->amiJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVooO)
            - 1.0 * ccpy_einsum('NmFe,FaeiNJ->amiJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvoOO)
            - 1.0 * ccpy_einsum('NmfE,faEiNJ->amiJ', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVoOO)
            - 1.0 * ccpy_einsum('NmFE,FaEiNJ->amiJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVoOO)
    )
    H.ab.vooo[va, :, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfE,aEfinJ->amiJ', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvooO)
            - 0.5 * ccpy_einsum('nmFE,aEFinJ->amiJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVooO)
            - 1.0 * ccpy_einsum('NmfE,aEfiNJ->amiJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvoOO)
            - 0.5 * ccpy_einsum('NmFE,aEFiNJ->amiJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVoOO)
    )
    # amij
    H.ab.vooo[va, :, oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('NmfE,faEiNj->amij', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVoOo)
            - 1.0 * ccpy_einsum('NmFe,FaeiNj->amij', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvoOo)
            - 1.0 * ccpy_einsum('NmFE,FaEiNj->amij', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVoOo)
    )
    H.ab.vooo[va, :, oa, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('NmFe,aFeijN->amij', H.bb.oovv[Ob, :, Vb, vb], T.abb.vVvooO)
            + 0.5 * ccpy_einsum('NmFE,aEFijN->amij', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVooO)
    )
    
    ##### H.ab.vvov #####
    # ABIe
    H.ab.vvov[Va, Vb, Oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,AfBnIm->ABIe', H.ab.oovv[oa, ob, va, :], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('nMfe,AfBnIM->ABIe', H.ab.oovv[oa, Ob, va, :], T.aab.VvVoOO)
            - 1.0 * ccpy_einsum('Nmfe,AfBINm->ABIe', H.ab.oovv[Oa, ob, va, :], T.aab.VvVOOo)
            - 1.0 * ccpy_einsum('NMfe,AfBINM->ABIe', H.ab.oovv[Oa, Ob, va, :], T.aab.VvVOOO)
            - 1.0 * ccpy_einsum('nmFe,FABnIm->ABIe', H.ab.oovv[oa, ob, Va, :], T.aab.VVVoOo)
            - 1.0 * ccpy_einsum('nMFe,FABnIM->ABIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VVVoOO)
            + 1.0 * ccpy_einsum('NmFe,FABINm->ABIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VVVOOo)
            + 1.0 * ccpy_einsum('NMFe,FABINM->ABIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVVOOO)
    )
    H.ab.vvov[Va, Vb, Oa, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('nmfe,ABfInm->ABIe', H.bb.oovv[ob, ob, vb, :], T.abb.VVvOoo)
            - 1.0 * ccpy_einsum('Nmfe,ABfImN->ABIe', H.bb.oovv[Ob, ob, vb, :], T.abb.VVvOoO)
            + 0.5 * ccpy_einsum('NMfe,ABfINM->ABIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.VVvOOO)
            + 0.5 * ccpy_einsum('nmFe,ABFInm->ABIe', H.bb.oovv[ob, ob, Vb, :], T.abb.VVVOoo)
            - 1.0 * ccpy_einsum('NmFe,ABFImN->ABIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVVOoO)
            + 0.5 * ccpy_einsum('NMFe,ABFINM->ABIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVVOOO)
    )
    # AbIe
    H.ab.vvov[Va, vb, Oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,AfbnIm->AbIe', H.ab.oovv[oa, ob, va, :], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('nMfe,AfbnIM->AbIe', H.ab.oovv[oa, Ob, va, :], T.aab.VvvoOO)
            - 1.0 * ccpy_einsum('nmFe,FAbnIm->AbIe', H.ab.oovv[oa, ob, Va, :], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('nMFe,FAbnIM->AbIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VVvoOO)
            - 1.0 * ccpy_einsum('Nmfe,AfbINm->AbIe', H.ab.oovv[Oa, ob, va, :], T.aab.VvvOOo)
            - 1.0 * ccpy_einsum('NMfe,AfbINM->AbIe', H.ab.oovv[Oa, Ob, va, :], T.aab.VvvOOO)
            + 1.0 * ccpy_einsum('NmFe,FAbINm->AbIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VVvOOo)
            + 1.0 * ccpy_einsum('NMFe,FAbINM->AbIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVvOOO)
    )
    H.ab.vvov[Va, vb, Oa, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('nmfe,AbfInm->AbIe', H.bb.oovv[ob, ob, vb, :], T.abb.VvvOoo)
            - 0.5 * ccpy_einsum('nmFe,AFbInm->AbIe', H.bb.oovv[ob, ob, Vb, :], T.abb.VVvOoo)
            - 1.0 * ccpy_einsum('Nmfe,AbfImN->AbIe', H.bb.oovv[Ob, ob, vb, :], T.abb.VvvOoO)
            + 0.5 * ccpy_einsum('NMfe,AbfINM->AbIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.VvvOOO)
            + 1.0 * ccpy_einsum('NmFe,AFbImN->AbIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVvOoO)
            - 0.5 * ccpy_einsum('NMFe,AFbINM->AbIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVvOOO)
    )
    # aBIe
    H.ab.vvov[va, Vb, Oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfe,faBnIm->aBIe', H.ab.oovv[oa, ob, va, :], T.aab.vvVoOo)
            - 1.0 * ccpy_einsum('nMfe,faBnIM->aBIe', H.ab.oovv[oa, Ob, va, :], T.aab.vvVoOO)
            + 1.0 * ccpy_einsum('Nmfe,faBINm->aBIe', H.ab.oovv[Oa, ob, va, :], T.aab.vvVOOo)
            + 1.0 * ccpy_einsum('NMfe,faBINM->aBIe', H.ab.oovv[Oa, Ob, va, :], T.aab.vvVOOO)
            - 1.0 * ccpy_einsum('nmFe,FaBnIm->aBIe', H.ab.oovv[oa, ob, Va, :], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('nMFe,FaBnIM->aBIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VvVoOO)
            + 1.0 * ccpy_einsum('NmFe,FaBINm->aBIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VvVOOo)
            + 1.0 * ccpy_einsum('NMFe,FaBINM->aBIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvVOOO)
    )
    H.ab.vvov[va, Vb, Oa, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('nmfe,aBfInm->aBIe', H.bb.oovv[ob, ob, vb, :], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('Nmfe,aBfImN->aBIe', H.bb.oovv[Ob, ob, vb, :], T.abb.vVvOoO)
            + 0.5 * ccpy_einsum('NMfe,aBfINM->aBIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.vVvOOO)
            + 0.5 * ccpy_einsum('nmFe,aBFInm->aBIe', H.bb.oovv[ob, ob, Vb, :], T.abb.vVVOoo)
            - 1.0 * ccpy_einsum('NmFe,aBFImN->aBIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVVOoO)
            + 0.5 * ccpy_einsum('NMFe,aBFINM->aBIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVVOOO)
    )
    # abIe
    H.ab.vvov[va, vb, Oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmFe,FabnIm->abIe', H.ab.oovv[oa, ob, Va, :], T.aab.VvvoOo)
            - 1.0 * ccpy_einsum('nMFe,FabnIM->abIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('NmFe,FabINm->abIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VvvOOo)
            + 1.0 * ccpy_einsum('NMFe,FabINM->abIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvvOOO)
    )
    H.ab.vvov[va, vb, Oa, :] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('nmFe,aFbInm->abIe', H.bb.oovv[ob, ob, Vb, :], T.abb.vVvOoo)
            + 1.0 * ccpy_einsum('NmFe,aFbImN->abIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVvOoO)
            - 0.5 * ccpy_einsum('NMFe,aFbINM->abIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVvOOO)
    )
    # ABie
    H.ab.vvov[Va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nMfe,AfBinM->ABie', H.ab.oovv[oa, Ob, va, :], T.aab.VvVooO)
            + 1.0 * ccpy_einsum('nMFe,FABinM->ABie', H.ab.oovv[oa, Ob, Va, :], T.aab.VVVooO)
            - 1.0 * ccpy_einsum('Nmfe,AfBiNm->ABie', H.ab.oovv[Oa, ob, va, :], T.aab.VvVoOo)
            - 1.0 * ccpy_einsum('NMfe,AfBiNM->ABie', H.ab.oovv[Oa, Ob, va, :], T.aab.VvVoOO)
            + 1.0 * ccpy_einsum('NmFe,FABiNm->ABie', H.ab.oovv[Oa, ob, Va, :], T.aab.VVVoOo)
            + 1.0 * ccpy_einsum('NMFe,FABiNM->ABie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVVoOO)
    )
    H.ab.vvov[Va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Nmfe,ABfimN->ABie', H.bb.oovv[Ob, ob, vb, :], T.abb.VVvooO)
            + 0.5 * ccpy_einsum('NMfe,ABfiNM->ABie', H.bb.oovv[Ob, Ob, vb, :], T.abb.VVvoOO)
            - 1.0 * ccpy_einsum('NmFe,ABFimN->ABie', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVVooO)
            + 0.5 * ccpy_einsum('NMFe,ABFiNM->ABie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVVoOO)
    )
    # Abie
    H.ab.vvov[Va, vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nMfe,AfbinM->Abie', H.ab.oovv[oa, Ob, va, :], T.aab.VvvooO)
            + 1.0 * ccpy_einsum('nMFe,FAbinM->Abie', H.ab.oovv[oa, Ob, Va, :], T.aab.VVvooO)
            - 1.0 * ccpy_einsum('Nmfe,AfbiNm->Abie', H.ab.oovv[Oa, ob, va, :], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('NmFe,FAbiNm->Abie', H.ab.oovv[Oa, ob, Va, :], T.aab.VVvoOo)
            - 1.0 * ccpy_einsum('NMfe,AfbiNM->Abie', H.ab.oovv[Oa, Ob, va, :], T.aab.VvvoOO)
            + 1.0 * ccpy_einsum('NMFe,FAbiNM->Abie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVvoOO)
    )
    H.ab.vvov[Va, vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Nmfe,AbfimN->Abie', H.bb.oovv[Ob, ob, vb, :], T.abb.VvvooO)
            + 1.0 * ccpy_einsum('NmFe,AFbimN->Abie', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVvooO)
            + 0.5 * ccpy_einsum('NMfe,AbfiNM->Abie', H.bb.oovv[Ob, Ob, vb, :], T.abb.VvvoOO)
            - 0.5 * ccpy_einsum('NMFe,AFbiNM->Abie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVvoOO)
    )
    # aBie
    H.ab.vvov[va, Vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nMfe,faBinM->aBie', H.ab.oovv[oa, Ob, va, :], T.aab.vvVooO)
            + 1.0 * ccpy_einsum('Nmfe,faBiNm->aBie', H.ab.oovv[Oa, ob, va, :], T.aab.vvVoOo)
            + 1.0 * ccpy_einsum('NMfe,faBiNM->aBie', H.ab.oovv[Oa, Ob, va, :], T.aab.vvVoOO)
            + 1.0 * ccpy_einsum('nMFe,FaBinM->aBie', H.ab.oovv[oa, Ob, Va, :], T.aab.VvVooO)
            + 1.0 * ccpy_einsum('NmFe,FaBiNm->aBie', H.ab.oovv[Oa, ob, Va, :], T.aab.VvVoOo)
            + 1.0 * ccpy_einsum('NMFe,FaBiNM->aBie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvVoOO)
    )
    H.ab.vvov[va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Nmfe,aBfimN->aBie', H.bb.oovv[Ob, ob, vb, :], T.abb.vVvooO)
            + 0.5 * ccpy_einsum('NMfe,aBfiNM->aBie', H.bb.oovv[Ob, Ob, vb, :], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('NmFe,aBFimN->aBie', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVVooO)
            + 0.5 * ccpy_einsum('NMFe,aBFiNM->aBie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVVoOO)
    )
    # abie
    H.ab.vvov[va, vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nMFe,FabinM->abie', H.ab.oovv[oa, Ob, Va, :], T.aab.VvvooO)
            + 1.0 * ccpy_einsum('NmFe,FabiNm->abie', H.ab.oovv[Oa, ob, Va, :], T.aab.VvvoOo)
            + 1.0 * ccpy_einsum('NMFe,FabiNM->abie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvvoOO)
    )
    H.ab.vvov[va, vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('NmFe,aFbimN->abie', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVvooO)
            - 0.5 * ccpy_einsum('NMFe,aFbiNM->abie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVvoOO)
    )
    
    #########################################################################
    ############################ bb intermdiates ############################
    #########################################################################
    
    ##### H.bb.vooo #####
    # AmIJ
    H.bb.vooo[Vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mnef,AfenIJ->AmIJ', H.bb.oovv[:, ob, vb, vb], T.bbb.VvvoOO)
            - 0.5 * ccpy_einsum('mNef,AfeIJN->AmIJ', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvOOO)
            + 1.0 * ccpy_einsum('mneF,FAenIJ->AmIJ', H.bb.oovv[:, ob, vb, Vb], T.bbb.VVvoOO)
            - 0.5 * ccpy_einsum('mnEF,FEAnIJ->AmIJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVVoOO)
            + 1.0 * ccpy_einsum('mNeF,FAeIJN->AmIJ', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VVvOOO)
            - 0.5 * ccpy_einsum('mNEF,FEAIJN->AmIJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVOOO)
    )
    H.bb.vooo[Vb, :, Ob, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,fAenIJ->AmIJ', H.ab.oovv[oa, :, va, vb], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('nmfE,fEAnIJ->AmIJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVVoOO)
            + 1.0 * ccpy_einsum('Nmfe,fAeNIJ->AmIJ', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOOO)
            - 1.0 * ccpy_einsum('NmfE,fEANIJ->AmIJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOOO)
            + 1.0 * ccpy_einsum('nmFe,FAenIJ->AmIJ', H.ab.oovv[oa, :, Va, vb], T.abb.VVvoOO)
            - 1.0 * ccpy_einsum('nmFE,FEAnIJ->AmIJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVVoOO)
            + 1.0 * ccpy_einsum('NmFe,FAeNIJ->AmIJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOOO)
            - 1.0 * ccpy_einsum('NmFE,FEANIJ->AmIJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOOO)
    )
    # AmiJ
    H.bb.vooo[Vb, :, ob, Ob] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,AfeinJ->AmiJ', H.bb.oovv[:, ob, vb, vb], T.bbb.VvvooO)
            + 1.0 * ccpy_einsum('mnEf,EAfinJ->AmiJ', H.bb.oovv[:, ob, Vb, vb], T.bbb.VVvooO)
            + 0.5 * ccpy_einsum('mnEF,FEAinJ->AmiJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVVooO)
            - 0.5 * ccpy_einsum('mNef,AfeiJN->AmiJ', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvoOO)
            - 1.0 * ccpy_einsum('mNEf,EAfiJN->AmiJ', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VVvoOO)
            - 0.5 * ccpy_einsum('mNEF,FEAiJN->AmiJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVoOO)
    )
    H.bb.vooo[Vb, :, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,fAeniJ->AmiJ', H.ab.oovv[oa, :, va, vb], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('nmFe,FAeniJ->AmiJ', H.ab.oovv[oa, :, Va, vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('nmfE,fEAniJ->AmiJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVVooO)
            - 1.0 * ccpy_einsum('nmFE,FEAniJ->AmiJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVVooO)
            + 1.0 * ccpy_einsum('Nmfe,fAeNiJ->AmiJ', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOoO)
            + 1.0 * ccpy_einsum('NmFe,FAeNiJ->AmiJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOoO)
            - 1.0 * ccpy_einsum('NmfE,fEANiJ->AmiJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOoO)
            - 1.0 * ccpy_einsum('NmFE,FEANiJ->AmiJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOoO)
    )
    H.bb.vooo[Vb, :, Ob, ob] = -1.0 * np.transpose(H.bb.vooo[Vb, :, ob, Ob], (0, 1, 3, 2))
    # Amij
    H.bb.vooo[Vb, :, ob, ob] += (1.0 / 1.0) * (
            -0.5 * ccpy_einsum('mNef,AfeijN->Amij', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvooO)
            + 1.0 * ccpy_einsum('mNeF,FAeijN->Amij', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VVvooO)
            - 0.5 * ccpy_einsum('mNEF,FEAijN->Amij', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVooO)
    )
    H.bb.vooo[Vb, :, ob, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Nmfe,fAeNij->Amij', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('NmfE,fEANij->Amij', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOoo)
            + 1.0 * ccpy_einsum('NmFe,FAeNij->Amij', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOoo)
            - 1.0 * ccpy_einsum('NmFE,FEANij->Amij', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOoo)
    )
    # amIJ
    H.bb.vooo[vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mneF,FeanIJ->amIJ', H.bb.oovv[:, ob, vb, Vb], T.bbb.VvvoOO)
            - 0.5 * ccpy_einsum('mnEF,FEanIJ->amIJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVvoOO)
            - 1.0 * ccpy_einsum('mNeF,FeaIJN->amIJ', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VvvOOO)
            - 0.5 * ccpy_einsum('mNEF,FEaIJN->amIJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvOOO)
    )
    H.bb.vooo[vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfE,fEanIJ->amIJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('nmFe,FeanIJ->amIJ', H.ab.oovv[oa, :, Va, vb], T.abb.VvvoOO)
            - 1.0 * ccpy_einsum('nmFE,FEanIJ->amIJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVvoOO)
            - 1.0 * ccpy_einsum('NmfE,fEaNIJ->amIJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOOO)
            - 1.0 * ccpy_einsum('NmFe,FeaNIJ->amIJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOOO)
            - 1.0 * ccpy_einsum('NmFE,FEaNIJ->amIJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOOO)
    )
    # amiJ
    H.bb.vooo[vb, :, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('mnEf,EfainJ->amiJ', H.bb.oovv[:, ob, Vb, vb], T.bbb.VvvooO)
            + 0.5 * ccpy_einsum('mnEF,FEainJ->amiJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVvooO)
            + 1.0 * ccpy_einsum('mNEf,EfaiJN->amiJ', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VvvoOO)
            - 0.5 * ccpy_einsum('mNEF,FEaiJN->amiJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvoOO)
    )
    H.bb.vooo[vb, :, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmFe,FeaniJ->amiJ', H.ab.oovv[oa, :, Va, vb], T.abb.VvvooO)
            - 1.0 * ccpy_einsum('NmFe,FeaNiJ->amiJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOoO)
            - 1.0 * ccpy_einsum('nmfE,fEaniJ->amiJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVvooO)
            - 1.0 * ccpy_einsum('nmFE,FEaniJ->amiJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('NmfE,fEaNiJ->amiJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('NmFE,FEaNiJ->amiJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOoO)
    )
    H.bb.vooo[vb, :, Ob, ob] = -1.0 * np.transpose(H.bb.vooo[vb, :, ob, Ob], (0, 1, 3, 2))
    # amij
    H.bb.vooo[vb, :, ob, ob] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNEf,EfaijN->amij', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VvvooO)
            - 0.5 * ccpy_einsum('mNEF,FEaijN->amij', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvooO)
    )
    H.bb.vooo[vb, :, ob, ob] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('NmFe,FeaNij->amij', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOoo)
            - 1.0 * ccpy_einsum('NmfE,fEaNij->amij', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('NmFE,FEaNij->amij', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOoo)
    )
    ##### H.bb.vvov #####
    # ABIe
    H.bb.vvov[Vb, Vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,BAfmnI->ABIe', H.bb.oovv[ob, ob, :, vb], T.bbb.VVvooO)
            + 0.5 * ccpy_einsum('mneF,FBAmnI->ABIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VVVooO)
            + 1.0 * ccpy_einsum('Mnef,BAfnIM->ABIe', H.bb.oovv[Ob, ob, :, vb], T.bbb.VVvoOO)
            + 0.5 * ccpy_einsum('MNef,BAfIMN->ABIe', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VVvOOO)
            + 1.0 * ccpy_einsum('MneF,FBAnIM->ABIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVVoOO)
            + 0.5 * ccpy_einsum('MNeF,FBAIMN->ABIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVVOOO)
    )
    H.bb.vvov[Vb, Vb, Ob, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmfe,fBAnmI->ABIe', H.ab.oovv[oa, ob, va, :], T.abb.vVVooO)
            - 1.0 * ccpy_einsum('nmFe,FBAnmI->ABIe', H.ab.oovv[oa, ob, Va, :], T.abb.VVVooO)
            - 1.0 * ccpy_einsum('Nmfe,fBANmI->ABIe', H.ab.oovv[Oa, ob, va, :], T.abb.vVVOoO)
            - 1.0 * ccpy_einsum('NmFe,FBANmI->ABIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VVVOoO)
            + 1.0 * ccpy_einsum('nMfe,fBAnIM->ABIe', H.ab.oovv[oa, Ob, va, :], T.abb.vVVoOO)
            + 1.0 * ccpy_einsum('nMFe,FBAnIM->ABIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VVVoOO)
            + 1.0 * ccpy_einsum('NMfe,fBANIM->ABIe', H.ab.oovv[Oa, Ob, va, :], T.abb.vVVOOO)
            + 1.0 * ccpy_einsum('NMFe,FBANIM->ABIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVVOOO)
    )
    # AbIe
    H.bb.vvov[Vb, vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mnef,AfbmnI->AbIe', H.bb.oovv[ob, ob, :, vb], T.bbb.VvvooO)
            - 0.5 * ccpy_einsum('mneF,FAbmnI->AbIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VVvooO)
            + 1.0 * ccpy_einsum('Mnef,AfbnIM->AbIe', H.bb.oovv[Ob, ob, :, vb], T.bbb.VvvoOO)
            - 1.0 * ccpy_einsum('MneF,FAbnIM->AbIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVvoOO)
            + 0.5 * ccpy_einsum('MNef,AfbIMN->AbIe', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VvvOOO)
            - 0.5 * ccpy_einsum('MNeF,FAbIMN->AbIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVvOOO)
    )
    H.bb.vvov[Vb, vb, Ob, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('nmfe,fAbnmI->AbIe', H.ab.oovv[oa, ob, va, :], T.abb.vVvooO)
            + 1.0 * ccpy_einsum('nmFe,FAbnmI->AbIe', H.ab.oovv[oa, ob, Va, :], T.abb.VVvooO)
            + 1.0 * ccpy_einsum('Nmfe,fAbNmI->AbIe', H.ab.oovv[Oa, ob, va, :], T.abb.vVvOoO)
            + 1.0 * ccpy_einsum('NmFe,FAbNmI->AbIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VVvOoO)
            - 1.0 * ccpy_einsum('nMfe,fAbnIM->AbIe', H.ab.oovv[oa, Ob, va, :], T.abb.vVvoOO)
            - 1.0 * ccpy_einsum('nMFe,FAbnIM->AbIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VVvoOO)
            - 1.0 * ccpy_einsum('NMfe,fAbNIM->AbIe', H.ab.oovv[Oa, Ob, va, :], T.abb.vVvOOO)
            - 1.0 * ccpy_einsum('NMFe,FAbNIM->AbIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVvOOO)
    )
    H.bb.vvov[vb, Vb, Ob, :] = -1.0 * np.transpose(H.bb.vvov[Vb, vb, Ob, :], (1, 0, 2, 3))
    # abIe
    H.bb.vvov[vb, vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * ccpy_einsum('mneF,FbamnI->abIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VvvooO)
            + 1.0 * ccpy_einsum('MneF,FbanIM->abIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VvvoOO)
            + 0.5 * ccpy_einsum('MNeF,FbaIMN->abIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VvvOOO)
    )
    H.bb.vvov[vb, vb, Ob, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nmFe,FbanmI->abIe', H.ab.oovv[oa, ob, Va, :], T.abb.VvvooO)
            - 1.0 * ccpy_einsum('NmFe,FbaNmI->abIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VvvOoO)
            + 1.0 * ccpy_einsum('nMFe,FbanIM->abIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VvvoOO)
            + 1.0 * ccpy_einsum('NMFe,FbaNIM->abIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VvvOOO)
    )
    # ABie
    H.bb.vvov[Vb, Vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('Mnef,BAfinM->ABie', H.bb.oovv[Ob, ob, :, vb], T.bbb.VVvooO)
            + 0.5 * ccpy_einsum('MNef,BAfiMN->ABie', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VVvoOO)
            - 1.0 * ccpy_einsum('MneF,FBAinM->ABie', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVVooO)
            + 0.5 * ccpy_einsum('MNeF,FBAiMN->ABie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVVoOO)
    )
    H.bb.vvov[Vb, Vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('Nmfe,fBANim->ABie', H.ab.oovv[Oa, ob, va, :], T.abb.vVVOoo)
            + 1.0 * ccpy_einsum('NmFe,FBANim->ABie', H.ab.oovv[Oa, ob, Va, :], T.abb.VVVOoo)
            + 1.0 * ccpy_einsum('nMfe,fBAniM->ABie', H.ab.oovv[oa, Ob, va, :], T.abb.vVVooO)
            + 1.0 * ccpy_einsum('NMfe,fBANiM->ABie', H.ab.oovv[Oa, Ob, va, :], T.abb.vVVOoO)
            + 1.0 * ccpy_einsum('nMFe,FBAniM->ABie', H.ab.oovv[oa, Ob, Va, :], T.abb.VVVooO)
            + 1.0 * ccpy_einsum('NMFe,FBANiM->ABie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVVOoO)
    )
    # Abie
    H.bb.vvov[Vb, vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('mNef,AfbimN->Abie', H.bb.oovv[ob, Ob, :, vb], T.bbb.VvvooO)
            + 0.5 * ccpy_einsum('MNef,AfbiMN->Abie', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VvvoOO)
            - 1.0 * ccpy_einsum('mNeF,FAbimN->Abie', H.bb.oovv[ob, Ob, :, Vb], T.bbb.VVvooO)
            - 0.5 * ccpy_einsum('MNeF,FAbiMN->Abie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVvoOO)
    )
    H.bb.vvov[Vb, vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('nMfe,fAbniM->Abie', H.ab.oovv[oa, Ob, va, :], T.abb.vVvooO)
            - 1.0 * ccpy_einsum('nMFe,FAbniM->Abie', H.ab.oovv[oa, Ob, Va, :], T.abb.VVvooO)
            - 1.0 * ccpy_einsum('Nmfe,fAbNim->Abie', H.ab.oovv[Oa, ob, va, :], T.abb.vVvOoo)
            - 1.0 * ccpy_einsum('NMfe,fAbNiM->Abie', H.ab.oovv[Oa, Ob, va, :], T.abb.vVvOoO)
            - 1.0 * ccpy_einsum('NmFe,FAbNim->Abie', H.ab.oovv[Oa, ob, Va, :], T.abb.VVvOoo)
            - 1.0 * ccpy_einsum('NMFe,FAbNiM->Abie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVvOoO)
    )
    H.bb.vvov[vb, Vb, ob, :] = -1.0 * np.transpose(H.bb.vvov[Vb, vb, ob, :], (1, 0, 2, 3))
    # abie
    H.bb.vvov[vb, vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * ccpy_einsum('MneF,FbainM->abie', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VvvooO)
            + 0.5 * ccpy_einsum('MNeF,FbaiMN->abie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VvvoOO)
    )
    H.bb.vvov[vb, vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * ccpy_einsum('NmFe,FbaNim->abie', H.ab.oovv[Oa, ob, Va, :], T.abb.VvvOoo)
            + 1.0 * ccpy_einsum('nMFe,FbaniM->abie', H.ab.oovv[oa, Ob, Va, :], T.abb.VvvooO)
            + 1.0 * ccpy_einsum('NMFe,FbaNiM->abie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VvvOoO)
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
