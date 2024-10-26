import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

import time as time

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.vvVoOo = (2.0 / 2.0) * (
            +1.0 * np.einsum('bCek,aeiJ->abCiJk', H.ab.vvvo[va, Vb, :, ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,abim->abCiJk', H.ab.ovoo[:, Vb, Oa, ob], T.aa[va, va, oa, :], optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,abJm->abCiJk', H.ab.ovoo[:, Vb, oa, ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJk->abCiJk', H.ab.vvov[va, Vb, oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beik->abCiJk', H.ab.vvov[va, Vb, Oa, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amik,bCJm->abCiJk', H.ab.vooo[va, :, oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJk,bCim->abCiJk', H.ab.vooo[va, :, Oa, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJk->abCiJk', H.aa.vvov[va, va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCik->abCiJk', H.aa.vvov[va, va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmk->abCiJk', H.aa.vooo[va, :, oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJk->abCiJk', H.a.oo[oa, oa], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJk->abCiJk', H.a.oo[Oa, oa], T.aab.vvVOOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,baCiMk->abCiJk', H.a.oo[Oa, Oa], T.aab.vvVoOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,baCiJm->abCiJk', H.b.oo[ob, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,baCiJM->abCiJk', H.b.oo[Ob, ob], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJk->abCiJk', H.a.vv[va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJk->abCiJk', H.a.vv[va, Va], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJk->abCiJk', H.b.vv[Vb, Vb], T.aab.vvVoOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MniJ,baCnMk->abCiJk', H.aa.oooo[Oa, oa, oa, Oa], T.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNk->abCiJk', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.vvVOOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJk,baCimN->abCiJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MnJk,baCiMn->abCiJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MNJk,baCiMN->abCiJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,baCmJn->abCiJk', H.ab.oooo[oa, ob, oa, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNik,baCmJN->abCiJk', H.ab.oooo[oa, Ob, oa, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mnik,baCJMn->abCiJk', H.ab.oooo[Oa, ob, oa, ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MNik,baCJMN->abCiJk', H.ab.oooo[Oa, Ob, oa, ob], T.aab.vvVOOO, optimize=True)
    )
    #t1 = time.time()
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJk->abCiJk', H.aa.vvvv[va, va, va, va], T.aab.vvVoOo, optimize=True) ###
            - 1.0 * np.einsum('abeF,FeCiJk->abCiJk', H.aa.vvvv[va, va, va, Va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJk->abCiJk', H.aa.vvvv[va, va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCEf,EafiJk->abCiJk', H.ab.vvvv[va, Vb, Va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFiJk->abCiJk', H.ab.vvvv[va, Vb, va, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJk->abCiJk', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    #print("Time for t3b vvVoOo = ", time.time() - t1)
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJk->abCiJk', H.aa.voov[va, oa, oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJk->abCiJk', H.aa.voov[va, oa, oa, Va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJk->abCiJk', H.aa.voov[va, Oa, oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJk->abCiJk', H.aa.voov[va, Oa, oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,beCiMk->abCiJk', H.aa.voov[va, Oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMk->abCiJk', H.aa.voov[va, Oa, Oa, Va], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmk->abCiJk', H.ab.voov[va, ob, oa, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmk->abCiJk', H.ab.voov[va, ob, oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMie,bCeJkM->abCiJk', H.ab.voov[va, Ob, oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMiE,bCEJkM->abCiJk', H.ab.voov[va, Ob, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,bCeikM->abCiJk', H.ab.voov[va, Ob, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMJE,bCEikM->abCiJk', H.ab.voov[va, Ob, Oa, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEk,EbaimJ->abCiJk', H.ab.ovvo[oa, Vb, Va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EbaiJM->abCiJk', H.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VvvoOO, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmkE,baEiJm->abCiJk', H.bb.voov[Vb, ob, ob, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,baEiJM->abCiJk', H.bb.voov[Vb, Ob, ob, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amek,beCiJm->abCiJk', H.ab.vovo[va, ob, va, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('amEk,EbCiJm->abCiJk', H.ab.vovo[va, ob, Va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('aMek,beCiJM->abCiJk', H.ab.vovo[va, Ob, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMEk,EbCiJM->abCiJk', H.ab.vovo[va, Ob, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJk->abCiJk', H.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJk->abCiJk', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVOOo, optimize=True)
    )
    dT.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJE,baEiMk->abCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVoOo, optimize=True)
    )

    dT.aab.vvVoOo -= np.transpose(dT.aab.vvVoOo, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.vvVoOo, dT.aab.vvVoOo = cc_active_loops.update_t3b_001010(
        T.aab.vvVoOo,
        dT.aab.vvVoOo,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT