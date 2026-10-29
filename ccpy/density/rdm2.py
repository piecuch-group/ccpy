import numpy as np
from ccpy.models.integrals import Integral

def calc_rdm2_normal_order(T, L, system, flag_RHF=False):

    assert flag_RHF

    rdm2 = Integral.from_empty(system, 2, use_none=True)
    tau2_asy = Integral.from_empty(system, 2, use_none=True)
    tau2_sy = Integral.from_empty(system, 2, use_none=True)

    #Preparing effective t2s
    tau2_asy.aa = ( np.einsum('ai,bj->abij', T.a, T.a)
                -np.einsum('bi,aj->abij', T.a, T.a)
    )
    tau2_asy.aa += T.aa.copy()
    tau2_asy.ab = T.ab.copy() + np.einsum('ai,bj->abij', T.a, T.b)
    if flag_RHF:
        tau2_asy.bb = tau2_asy.aa
    else:
        tau2_asy.bb = ( np.einsum('ai,bj->abij', T.b, T.b)
                -np.einsum('bi,aj->abij', T.b, T.b)
        )
        tau2_asy.bb += T.bb.copy()

    tau2_sy.aa = 2*np.einsum('ai,bj->abij', T.a, T.a) + T.aa.copy()
    tau2_sy.ab = 2*np.einsum('ai,bj->abij', T.a, T.b) + T.ab.copy()
    if flag_RHF:
        tau2_sy.bb = tau2_sy.aa
    else:
        tau2_sy.bb = 2*np.einsum('ai,bj->abij', T.b, T.b) + T.bb.copy()

    #L2
    rdm2.aa.oovv = np.transpose(L.aa, (2,3,0,1))
    rdm2.ab.oovv = np.transpose(L.ab, (2,3,0,1))

    #tau2
    rdm2.aa.vvoo = tau2_asy.aa.copy()
    rdm2.ab.vvoo = tau2_asy.ab.copy()

    #L1T1
    rdm2.aa.ovov = -np.einsum('bi,aj->iajb', L.a, T.a, optimize=True)
    rdm2.ab.ovov = 0
    rdm2.ab.vovo = 0
    rdm2.ab.ovvo = np.einsum('bi,aj->iabj', L.a, T.b, optimize=True)
    rdm2.ab.voov = np.einsum('bi,aj->aijb', L.b, T.a, optimize=True)

    #L1tau2
    rdm2.aa.ovoo = -np.einsum('ei,eajk->iajk', L.a, tau2_asy.aa, optimize=True)
    rdm2.ab.ovoo = -np.einsum('ei,eajk->iajk', L.a, tau2_asy.ab, optimize=True)
    rdm2.ab.vooo = -np.einsum('ei,aekj->aikj', L.b, tau2_asy.ab, optimize=True)

    rdm2.aa.vvov = np.einsum('cm,abim->abic', L.a, tau2_asy.aa, optimize=True)
    rdm2.ab.vvov = np.einsum('cm,abim->abic', L.b, tau2_asy.ab, optimize=True)
    rdm2.ab.vvvo = np.einsum('cm,bami->baci', L.a, tau2_asy.ab, optimize=True)

    #L1tau2t1
    rdm2.aa.vvoo += -np.einsum('em,aeij,bm->abij', L.a, tau2_asy.aa, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('em,beij,am->abij', L.a, tau2_asy.aa, T.a, optimize=True)
    rdm2.ab.vvoo += -np.einsum('em,aeij,bm->abij', L.b, tau2_asy.ab, T.b, optimize=True)
    rdm2.ab.vvoo += -np.einsum('em,eaji,bm->baji', L.a, tau2_asy.ab, T.a, optimize=True)

    rdm2.aa.vvoo += -np.einsum('em,abim,ej->abij', L.a, T.aa, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('em,abjm,ei->abij', L.a, T.aa, T.a, optimize=True)
    rdm2.ab.vvoo += -np.einsum('em,abim,ej->abij', L.b, T.ab, T.b, optimize=True)
    rdm2.ab.vvoo += -np.einsum('em,bami,ej->baji', L.b, T.ab, T.b, optimize=True)

    rdm2.aa.vvoo += np.einsum('em,bj,aeim->abij', L.a, T.a, T.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('em,bj,aeim->abij', L.b, T.a, T.ab, optimize=True)
    rdm2.aa.vvoo += -np.einsum('em,bi,aejm->abij', L.a, T.a, T.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('em,bi,aejm->abij', L.b, T.a, T.ab, optimize=True)
    rdm2.aa.vvoo += -np.einsum('em,aj,beim->abij', L.a, T.a, T.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('em,aj,beim->abij', L.b, T.a, T.ab, optimize=True)
    rdm2.aa.vvoo += np.einsum('em,ai,bejm->abij', L.a, T.a, T.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('em,ai,bejm->abij', L.b, T.a, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('em,bj,aeim->abij', L.a, T.b, T.aa, optimize=True)
    rdm2.ab.vvoo += np.einsum('em,bj,aeim->baji', L.b, T.b, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('em,ai,bejm->abij', L.a, T.b , T.aa, optimize=True)
    rdm2.ab.vvoo += np.einsum('em,ai,bejm->baji', L.b, T.b, T.ab, optimize=True)

    #L2T1
    rdm2.aa.ooov = -np.einsum('eaij,ek->ijka', L.aa, T.a, optimize=True)
    rdm2.ab.ooov = -np.einsum('eaij,ek->ijka', L.ab, T.a, optimize=True)
    rdm2.ab.oovo = -np.einsum('aeji,ek->jiak', L.ab, T.b, optimize=True)

    rdm2.aa.ovvv = np.einsum('bcim,am->iabc', L.aa, T.a, optimize=True)
    rdm2.ab.ovvv = np.einsum('bcim,am->iabc', L.ab, T.b, optimize=True)
    rdm2.ab.vovv = np.einsum('cbmi,am->aicb', L.ab, T.a, optimize=True)

    #L2tau2
    rdm2.aa.oooo = 0.5*np.einsum('efij,efkl->ijkl', L.aa, tau2_sy.aa, optimize=True)
    rdm2.ab.oooo = np.einsum('efij,efkl->ijkl', L.ab, tau2_asy.ab, optimize=True)

    # The vvvv blocks are not stored (no quantity with four virtual indices is formed);
    # calc_rdm2_vvvv_slice builds them three virtual indices at a time. Full expressions:
    # rdm2.aa.vvvv = 0.5*np.einsum('cdmn,abmn->abcd', L.aa, tau2_sy.aa, optimize=True)
    # rdm2.ab.vvvv = np.einsum('cdmn,abmn->abcd', L.ab, tau2_asy.ab, optimize=True)

    rdm2.aa.ovov += -np.einsum('ebim,eajm->iajb', L.aa, T.aa, optimize=True)
    rdm2.aa.ovov += -np.einsum('beim,aejm->iajb', L.ab, T.ab, optimize=True)
    rdm2.ab.ovov += -np.einsum('ebim,eajm->iajb', L.ab, T.ab, optimize=True)
    rdm2.ab.vovo += -np.einsum('bemi,aemj->aibj', L.ab, T.ab, optimize=True)
    rdm2.ab.ovvo += np.einsum('beim,eamj->iabj', L.aa, T.ab, optimize=True)
    rdm2.ab.ovvo += np.einsum('beim,eamj->iabj', L.ab, T.bb, optimize=True)
    rdm2.ab.voov += np.einsum('beim,aejm->aijb', L.bb, T.ab, optimize=True)
    rdm2.ab.voov += np.einsum('ebmi,eamj->aijb', L.ab, T.aa, optimize=True)

    rdm2.aa.ovov += -np.einsum('ebim,ej,am->iajb', L.aa,T.a, T.a, optimize=True)
    rdm2.ab.ovov += -np.einsum('ebim,ej,am->iajb', L.ab,T.a, T.b, optimize=True)
    rdm2.ab.vovo += -np.einsum('bemi,ej,am->aibj', L.ab,T.b, T.a, optimize=True)
    rdm2.ab.ovvo += -np.einsum('beim,ej,am->iabj', L.ab, T.b, T.b, optimize=True)
    rdm2.ab.voov += -np.einsum('ebmi,ej,am->aijb', L.ab, T.a, T.a, optimize=True)

    #L2tau2t1
    rdm2.aa.ovoo += -np.einsum('efim,afkm,ej->iajk', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.ovoo += -np.einsum('efim,afkm,ej->iajk', L.ab, T.ab, T.a, optimize=True)
    rdm2.aa.ovoo += np.einsum('efim,afjm,ek->iajk', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.ovoo += np.einsum('efim,afjm,ek->iajk', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.ovoo += -np.einsum('efim,famk,ej->iajk', L.aa, T.ab, T.a, optimize=True)
    rdm2.ab.ovoo += -np.einsum('efim,famk,ej->iajk', L.ab, T.bb, T.a, optimize=True)
    rdm2.ab.vooo += -np.einsum('femi,famk,ej->aikj', L.ab, T.aa, T.b, optimize=True)
    rdm2.ab.vooo += -np.einsum('femi,afkm,ej->aikj', L.bb, T.ab, T.b, optimize=True)
    rdm2.ab.ovoo += np.einsum('feim,fajm,ek->iajk', L.ab, T.ab, T.b, optimize=True)
    rdm2.ab.vooo += np.einsum('efmi,afmj,ek->aikj', L.ab, T.ab, T.a, optimize=True)

    rdm2.aa.ovoo += 0.5*np.einsum('efim,efjk,am->iajk', L.aa, tau2_sy.aa, T.a, optimize=True)
    rdm2.ab.ovoo += np.einsum('efim,efjk,am->iajk', L.ab, tau2_asy.ab, T.b, optimize=True)
    rdm2.ab.vooo += np.einsum('femi,fekj,am->aikj', L.ab, tau2_asy.ab, T.a, optimize=True)

    rdm2.aa.ovoo += -0.5*np.einsum('efmi,efmj,ak->iajk', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.ovoo += -np.einsum('feim,fejm,ak->iajk', L.ab, T.ab, T.a, optimize=True)
    rdm2.aa.ovoo += 0.5*np.einsum('efmi,efmk,aj->iajk', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.ovoo += np.einsum('feim,fekm,aj->iajk', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.ovoo += -0.5*np.einsum('feim,fejm,ak->iajk', L.aa, T.aa, T.b, optimize=True)
    rdm2.ab.ovoo += -np.einsum('feim,fejm,ak->iajk', L.ab, T.ab, T.b, optimize=True)
    rdm2.ab.vooo += -np.einsum('efmi,efmj,ak->aikj', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.vooo += -0.5*np.einsum('efmi,efmj,ak->aikj', L.bb, T.bb, T.a, optimize=True)

    rdm2.aa.vvov += np.einsum('ecmn,eami,bn->abic', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.vvov += np.einsum('cenm,aeim,bn->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.aa.vvov += -np.einsum('ecmn,ebmi,an->abic', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.vvov += -np.einsum('cenm,beim,an->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.vvov += np.einsum('ecmn,eami,bn->abic', L.ab, T.aa, T.b, optimize=True)
    rdm2.ab.vvov += np.einsum('ecmn,aeim,bn->abic', L.bb, T.ab, T.b, optimize=True)
    rdm2.ab.vvvo += np.einsum('cenm,eami,bn->baci', L.aa, T.ab, T.a, optimize=True)
    rdm2.ab.vvvo += np.einsum('cenm,eami,bn->baci', L.ab, T.bb, T.a, optimize=True)
    rdm2.ab.vvov += -np.einsum('ecnm,ebim,an->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.vvvo += -np.einsum('cemn,bemi,an->baci', L.ab, T.ab, T.b, optimize=True)

    rdm2.aa.vvov += -0.5*np.einsum('ecmn,abmn,ei->abic', L.aa, tau2_sy.aa,T.a, optimize=True)
    rdm2.ab.vvov += -np.einsum('ecmn,abmn,ei->abic', L.ab, tau2_asy.ab, T.a, optimize=True)
    rdm2.ab.vvvo += -np.einsum('cenm,banm,ei->baci', L.ab, tau2_asy.ab, T.b, optimize=True)

    rdm2.aa.vvov += 0.5*np.einsum('cemn,bemn,ai->abic', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.vvov += np.einsum('cemn,bemn,ai->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.aa.vvov += -0.5*np.einsum('cemn,aemn,bi->abic', L.aa, T.aa, T.a, optimize=True)
    rdm2.aa.vvov += -np.einsum('cemn,aemn,bi->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.vvov += np.einsum('ecnm,ebnm,ai->abic', L.ab, T.ab, T.a, optimize=True)
    rdm2.ab.vvov += 0.5*np.einsum('ecnm,ebnm,ai->abic', L.bb, T.bb, T.a, optimize=True)
    rdm2.ab.vvvo += 0.5*np.einsum('cemn,bemn,ai->baci', L.aa, T.aa, T.b, optimize=True)
    rdm2.ab.vvvo += np.einsum('cemn,bemn,ai->baci', L.ab, T.ab, T.b, optimize=True)

    #L2tau2**2
    rdm2.aa.vvoo += 0.5*np.einsum('efmn,efmi,abjn->abij', L.aa, T.aa, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('fenm,feim,abjn->abij', L.ab, T.ab, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += -0.5*np.einsum('efmn,efmj,abin->abij', L.aa, T.aa, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('fenm,fejm,abin->abij', L.ab, T.ab, tau2_asy.aa, optimize=True)
    rdm2.ab.vvoo += -np.einsum('efmn,efmj,abin->abij', L.ab, T.ab, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -0.5*np.einsum('efmn,efmj,abin->abij', L.bb, T.bb, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -0.5*np.einsum('efmn,efmj,bani->baji', L.aa, T.aa, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -np.einsum('fenm,fejm,bani->baji', L.ab, T.ab, tau2_asy.ab, optimize=True)

    rdm2.aa.vvoo += 0.5*np.einsum('efmn,eamn,bfij->abij', L.aa, T.aa, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('fenm,aenm,bfij->abij', L.ab, T.ab, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += -0.5*np.einsum('efmn,ebmn,afij->abij', L.aa, T.aa, tau2_asy.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('fenm,benm,afij->abij', L.ab, T.ab, tau2_asy.aa, optimize=True)
    rdm2.ab.vvoo += -np.einsum('efmn,ebmn,afij->abij', L.ab, T.ab, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -0.5*np.einsum('efmn,ebmn,afij->abij', L.bb, T.bb, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -0.5*np.einsum('efmn,ebmn,faji->baji', L.aa, T.aa, tau2_asy.ab, optimize=True)
    rdm2.ab.vvoo += -np.einsum('fenm,benm,faji->baji', L.ab, T.ab, tau2_asy.ab, optimize=True)

    rdm2.aa.vvoo += 0.25*np.einsum('efmn,efij,abmn->abij', L.aa, tau2_sy.aa, tau2_sy.aa, optimize=True)
    rdm2.ab.vvoo += np.einsum('efmn,efij,abmn->abij', L.ab, tau2_asy.ab, tau2_asy.ab, optimize=True)

    rdm2.aa.vvoo += np.einsum('efmn,eami,bfjn->abij', L.aa, T.aa, T.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('fenm,aeim,bfjn->abij', L.ab, T.ab, T.aa, optimize=True)
    rdm2.aa.vvoo += np.einsum('efmn,eami,bfjn->abij', L.ab, T.aa, T.ab, optimize=True)
    rdm2.aa.vvoo += np.einsum('efmn,aeim,bfjn->abij', L.bb, T.ab, T.ab, optimize=True)
    rdm2.aa.vvoo += -np.einsum('efmn,eamj,bfin->abij', L.aa, T.aa, T.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('fenm,aejm,bfin->abij', L.ab, T.ab, T.aa, optimize=True)
    rdm2.aa.vvoo += -np.einsum('efmn,eamj,bfin->abij', L.ab, T.aa, T.ab, optimize=True)
    rdm2.aa.vvoo += -np.einsum('efmn,aejm,bfin->abij', L.bb, T.ab, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('efmn,eami,fbnj->abij', L.aa, T.aa, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('fenm,aeim,fbnj->abij', L.ab, T.ab, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('efmn,eami,fbnj->abij', L.ab, T.aa, T.bb, optimize=True)
    rdm2.ab.vvoo += np.einsum('efmn,aeim,fbnj->abij', L.bb, T.ab, T.bb, optimize=True)

    rdm2.aa.vvoo += -np.einsum('efmn,eami,bn,fj->abij', L.aa, T.aa, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += -np.einsum('fenm,aeim,bn,fj->abij', L.ab, T.ab, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('efmn,ebmi,an,fj->abij', L.aa, T.aa, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('fenm,beim,an,fj->abij', L.ab, T.ab, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('efmn,eamj,bn,fi->abij', L.aa, T.aa, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += np.einsum('fenm,aejm,bn,fi->abij', L.ab, T.ab, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += -np.einsum('efmn,ebmj,an,fi->abij', L.aa, T.aa, T.a, T.a, optimize=True)
    rdm2.aa.vvoo += -np.einsum('fenm,bejm,an,fi->abij', L.ab, T.ab, T.a, T.a, optimize=True)
    rdm2.ab.vvoo += -np.einsum('efmn,eami,bn,fj->abij', L.ab, T.aa, T.b, T.b, optimize=True)
    rdm2.ab.vvoo += -np.einsum('efmn,aeim,bn,fj->abij', L.bb, T.ab, T.b, T.b, optimize=True)
    rdm2.ab.vvoo += -np.einsum('efmn,eami,bn,fj->baji', L.aa, T.ab, T.a, T.a, optimize=True)
    rdm2.ab.vvoo += -np.einsum('fenm,eami,bn,fj->baji', L.ab, T.bb, T.a, T.a, optimize=True)
    # exchange-type terms coupling the alpha (i, a) and beta (j, b) pairs through L.ab
    rdm2.ab.vvoo += 0.5*np.einsum('efnm,ebim,afnj->abij', L.ab, T.ab, T.ab, optimize=True)
    rdm2.ab.vvoo += 0.5*np.einsum('femn,aemj,fbin->abij', L.ab, T.ab, T.ab, optimize=True)
    rdm2.ab.vvoo += np.einsum('efnm,ebim,fj,an->abij', L.ab, T.ab, T.b, T.a, optimize=True)
    rdm2.ab.vvoo += np.einsum('femn,aemj,fi,bn->abij', L.ab, T.ab, T.a, T.b, optimize=True)
    return rdm2

def calc_rdm2_vvvv_slice(T, L, a):
    """The vvvv blocks of calc_rdm2_normal_order (RHF) for one value of the first index,
    rdm2.aa.vvvv[a, :, :, :] and rdm2.ab.vvvv[a, :, :, :], so that no quantity with four
    virtual indices is formed."""
    tau2_sy_aa_a = 2*np.einsum('i,bj->bij', T.a[a, :], T.a) + T.aa[a]
    tau2_asy_ab_a = T.ab[a] + np.einsum('i,bj->bij', T.a[a, :], T.b)
    vvvv_aa = 0.5*np.einsum('cdmn,bmn->bcd', L.aa, tau2_sy_aa_a, optimize=True)
    vvvv_ab = np.einsum('cdmn,bmn->bcd', L.ab, tau2_asy_ab_a, optimize=True)
    return vvvv_aa, vvvv_ab
