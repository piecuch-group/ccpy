import numpy as np

from ccpy.utilities.active_space import get_active_slices

def build_VT3_intermediates(T, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    #########################################################################
    ############################ aa intermdiates ############################
    #########################################################################

    ##### I2A_vooo #####
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    # AmIJ
    I2A_vooo[Va, :, Oa, Oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,AfenIJ->AmIJ', H.aa.oovv[:, oa, va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNef,AfeIJN->AmIJ', H.aa.oovv[:, Oa, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mneF,FAenIJ->AmIJ', H.aa.oovv[:, oa, va, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,FAeIJN->AmIJ', H.aa.oovv[:, Oa, va, Va], T.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEAnIJ->AmIJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAIJN->AmIJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    I2A_vooo[Va, :, Oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnef,AefIJn->AmIJ', H.ab.oovv[:, ob, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mNef,AefIJN->AmIJ', H.ab.oovv[:, Ob, va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mnEf,EAfIJn->AmIJ', H.ab.oovv[:, ob, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfIJN->AmIJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mneF,AeFIJn->AmIJ', H.ab.oovv[:, ob, va, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mNeF,AeFIJN->AmIJ', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('mnEF,EAFIJn->AmIJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFIJN->AmIJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    # AmiJ
    I2A_vooo[Va, :, oa, Oa] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,AfeinJ->AmiJ', H.aa.oovv[:, oa, va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('mNef,AfeiJN->AmiJ', H.aa.oovv[:, Oa, va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mnEf,EAfinJ->AmiJ', H.aa.oovv[:, oa, Va, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfiJN->AmiJ', H.aa.oovv[:, Oa, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEAinJ->AmiJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVVooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAiJN->AmiJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    I2A_vooo[Va, :, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnef,AefiJn->AmiJ', H.ab.oovv[:, ob, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNef,AefiJN->AmiJ', H.ab.oovv[:, Ob, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mneF,AeFiJn->AmiJ', H.ab.oovv[:, ob, va, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNeF,AeFiJN->AmiJ', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mnEf,EAfiJn->AmiJ', H.ab.oovv[:, ob, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfiJN->AmiJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mnEF,EAFiJn->AmiJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFiJN->AmiJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    I2A_vooo[Va, :, Oa, oa] = -1.0 * np.transpose(I2A_vooo[Va, :, oa, Oa], (0, 1, 3, 2))
    # Amij
    I2A_vooo[Va, :, oa, oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,AfeijN->Amij', H.aa.oovv[:, Oa, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfijN->Amij', H.aa.oovv[:, Oa, Va, va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAijN->Amij', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    I2A_vooo[Va, :, oa, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,AefijN->Amij', H.ab.oovv[:, Ob, va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,AeFijN->Amij', H.ab.oovv[:, Ob, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfijN->Amij', H.ab.oovv[:, Ob, Va, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFijN->Amij', H.ab.oovv[:, Ob, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    # amIJ
    I2A_vooo[va, :, Oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnEf,EfanIJ->amIJ', H.aa.oovv[:, oa, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEanIJ->amIJ', H.aa.oovv[:, oa, Va, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EfaIJN->amIJ', H.aa.oovv[:, Oa, Va, va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaIJN->amIJ', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    I2A_vooo[va, :, Oa, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,eaFIJn->amIJ', H.ab.oovv[:, ob, va, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mnEf,EafIJn->amIJ', H.ab.oovv[:, ob, Va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mnEF,EaFIJn->amIJ', H.ab.oovv[:, ob, Va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNeF,eaFIJN->amIJ', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EafIJN->amIJ', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EaFIJN->amIJ', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    # amIj
    I2A_vooo[va, :, Oa, oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,FeajnI->amIj', H.aa.oovv[:, oa, va, Va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEajnI->amIj', H.aa.oovv[:, oa, Va, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,FeajIN->amIj', H.aa.oovv[:, Oa, va, Va], T.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('mNEF,FEajIN->amIj', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    I2A_vooo[va, :, Oa, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnEf,EafjIn->amIj', H.ab.oovv[:, ob, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mneF,eaFjIn->amIj', H.ab.oovv[:, ob, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mnEF,EaFjIn->amIj', H.ab.oovv[:, ob, Va, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EafjIN->amIj', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,eaFjIN->amIj', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EaFjIN->amIj', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    I2A_vooo[va, :, oa, Oa] = -1.0 * np.transpose(I2A_vooo[va, :, Oa, oa], (0, 1, 3, 2))
    # amij
    I2A_vooo[va, :, oa, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaijN->amij', H.aa.oovv[:, Oa, Va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaijN->amij', H.aa.oovv[:, Oa, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    I2A_vooo[va, :, oa, oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNeF,eaFijN->amij', H.ab.oovv[:, Ob, va, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EafijN->amij', H.ab.oovv[:, Ob, Va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EaFijN->amij', H.ab.oovv[:, Ob, Va, Vb], T.aab.VvVooO, optimize=True)
    )

    ##### I2A_vvov #####
    I2A_vvov = H.aa.vvov.copy()
    # ABIe
    I2A_vvov[Va, Va, Oa, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,BAfmnI->ABIe', H.aa.oovv[oa, oa, :, va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('mneF,FBAmnI->ABIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNef,BAfmIN->ABIe', H.aa.oovv[oa, Oa, :, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FBAmIN->ABIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNef,BAfIMN->ABIe', H.aa.oovv[Oa, Oa, :, va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FBAIMN->ABIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVVOOO, optimize=True)
    )
    I2A_vvov[Va, Va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,BAfmIn->ABIe', H.ab.oovv[oa, ob, :, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mneF,BAFmIn->ABIe', H.ab.oovv[oa, ob, :, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mnef,BAfIMn->ABIe', H.ab.oovv[Oa, ob, :, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MneF,BAFIMn->ABIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNef,BAfmIN->ABIe', H.ab.oovv[oa, Ob, :, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,BAFmIN->ABIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNef,BAfIMN->ABIe', H.ab.oovv[Oa, Ob, :, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,BAFIMN->ABIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VVVOOO, optimize=True)
    )
    # aBIe
    I2A_vvov[va, Va, Oa, :] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,BfamnI->aBIe', H.aa.oovv[oa, oa, :, va], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('mneF,FBamnI->aBIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNef,BfamIN->aBIe', H.aa.oovv[oa, Oa, :, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FBamIN->aBIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNef,BfaIMN->aBIe', H.aa.oovv[Oa, Oa, :, va], T.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FBaIMN->aBIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVvOOO, optimize=True)
    )
    I2A_vvov[va, Va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,BafmIn->aBIe', H.ab.oovv[oa, ob, :, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mneF,BaFmIn->aBIe', H.ab.oovv[oa, ob, :, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Mnef,BafIMn->aBIe', H.ab.oovv[Oa, ob, :, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MneF,BaFIMn->aBIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNef,BafmIN->aBIe', H.ab.oovv[oa, Ob, :, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,BaFmIN->aBIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNef,BafIMN->aBIe', H.ab.oovv[Oa, Ob, :, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,BaFIMN->aBIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VvVOOO, optimize=True)
    )
    I2A_vvov[Va, va, Oa, :] = -1.0 * np.transpose(I2A_vvov[va, Va, Oa, :], (1, 0, 2, 3))
    # abIe
    I2A_vvov[va, va, Oa, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mneF,FbamnI->abIe', H.aa.oovv[oa, oa, :, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNeF,FbamIN->abIe', H.aa.oovv[oa, Oa, :, Va], T.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FbaIMN->abIe', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VvvOOO, optimize=True)
    )
    I2A_vvov[va, va, Oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,baFmIn->abIe', H.ab.oovv[oa, ob, :, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MneF,baFIMn->abIe', H.ab.oovv[Oa, ob, :, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNeF,baFmIN->abIe', H.ab.oovv[oa, Ob, :, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNeF,baFIMN->abIe', H.ab.oovv[Oa, Ob, :, Vb], T.aab.vvVOOO, optimize=True)
    )
    # ABie
    I2A_vvov[Va, Va, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,BAfimN->ABie', H.aa.oovv[oa, Oa, :, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,FBAimN->ABie', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNef,BAfiMN->ABie', H.aa.oovv[Oa, Oa, :, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FBAiMN->ABie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVVoOO, optimize=True)
    )
    I2A_vvov[Va, Va, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnef,BAfiMn->ABie', H.ab.oovv[Oa, ob, :, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MneF,BAFiMn->ABie', H.ab.oovv[Oa, ob, :, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNef,BAfimN->ABie', H.ab.oovv[oa, Ob, :, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,BAFimN->ABie', H.ab.oovv[oa, Ob, :, Vb], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MNef,BAfiMN->ABie', H.ab.oovv[Oa, Ob, :, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNeF,BAFiMN->ABie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VVVoOO, optimize=True)
    )
    # Abie
    I2A_vvov[Va, va, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,AfbimN->Abie', H.aa.oovv[oa, Oa, :, va], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNef,AfbiMN->Abie', H.aa.oovv[Oa, Oa, :, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FAbimN->Abie', H.aa.oovv[oa, Oa, :, Va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNeF,FAbiMN->Abie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VVvoOO, optimize=True)
    )
    I2A_vvov[Va, va, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Mnef,AbfiMn->Abie', H.ab.oovv[Oa, ob, :, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNef,AbfimN->Abie', H.ab.oovv[oa, Ob, :, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MNef,AbfiMN->Abie', H.ab.oovv[Oa, Ob, :, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,AbFiMn->Abie', H.ab.oovv[Oa, ob, :, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNeF,AbFimN->Abie', H.ab.oovv[oa, Ob, :, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MNeF,AbFiMN->Abie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.VvVoOO, optimize=True)
    )
    I2A_vvov[va, Va, oa, :] = -1.0 * np.transpose(I2A_vvov[Va, va, oa, :], (1, 0, 2, 3))
    # abie
    I2A_vvov[va, va, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MneF,FbainM->abie', H.aa.oovv[Oa, oa, :, Va], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FbaiMN->abie', H.aa.oovv[Oa, Oa, :, Va], T.aaa.VvvoOO, optimize=True)
    )
    I2A_vvov[va, va, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNeF,baFimN->abie', H.ab.oovv[oa, Ob, :, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MneF,baFiMn->abie', H.ab.oovv[Oa, ob, :, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MNeF,baFiMN->abie', H.ab.oovv[Oa, Ob, :, Vb], T.aab.vvVoOO, optimize=True)
    )

    #########################################################################
    ############################ ab intermdiates ############################
    #########################################################################

    ##### I2B_ovoo #####
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    # mBIJ
    I2B_ovoo[:, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,feBnIJ->mBIJ', H.aa.oovv[:, oa, va, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mnEf,EfBnIJ->mBIJ', H.aa.oovv[:, oa, Va, va], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEBnIJ->mBIJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('mNef,feBINJ->mBIJ', H.aa.oovv[:, Oa, va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EfBINJ->mBIJ', H.aa.oovv[:, Oa, Va, va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEBINJ->mBIJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    I2B_ovoo[:, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,eBfInJ->mBIJ', H.ab.oovv[:, ob, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mneF,eBFInJ->mBIJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mnEf,EBfInJ->mBIJ', H.ab.oovv[:, ob, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mnEF,EBFInJ->mBIJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNef,eBfINJ->mBIJ', H.ab.oovv[:, Ob, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mNeF,eBFINJ->mBIJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EBfINJ->mBIJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EBFINJ->mBIJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    # mBiJ
    I2B_ovoo[:, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,feBinJ->mBiJ', H.aa.oovv[:, oa, va, va], T.aab.vvVooO, optimize=True)
            - 0.5 * np.einsum('mNef,feBiNJ->mBiJ', H.aa.oovv[:, Oa, va, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mneF,FeBinJ->mBiJ', H.aa.oovv[:, oa, va, Va], T.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEBinJ->mBiJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNeF,FeBiNJ->mBiJ', H.aa.oovv[:, Oa, va, Va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEBiNJ->mBiJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    I2B_ovoo[:, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,eBfinJ->mBiJ', H.ab.oovv[:, ob, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mnEf,EBfinJ->mBiJ', H.ab.oovv[:, ob, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNef,eBfiNJ->mBiJ', H.ab.oovv[:, Ob, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EBfiNJ->mBiJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mneF,eBFinJ->mBiJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mnEF,EBFinJ->mBiJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNeF,eBFiNJ->mBiJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EBFiNJ->mBiJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    # mBIj
    I2B_ovoo[:, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,feBnIj->mBIj', H.aa.oovv[:, oa, va, va], T.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('mNef,feBINj->mBIj', H.aa.oovv[:, Oa, va, va], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mnEf,EfBnIj->mBIj', H.aa.oovv[:, oa, Va, va], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('mnEF,FEBnIj->mBIj', H.aa.oovv[:, oa, Va, Va], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EfBINj->mBIj', H.aa.oovv[:, Oa, Va, va], T.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEBINj->mBIj', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    I2B_ovoo[:, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,eBfInj->mBIj', H.ab.oovv[:, ob, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mneF,eBFInj->mBIj', H.ab.oovv[:, ob, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('mNef,eBfIjN->mBIj', H.ab.oovv[:, Ob, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNeF,eBFIjN->mBIj', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mnEf,EBfInj->mBIj', H.ab.oovv[:, ob, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mnEF,EBFInj->mBIj', H.ab.oovv[:, ob, Va, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('mNEf,EBfIjN->mBIj', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNEF,EBFIjN->mBIj', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    # mBij
    I2B_ovoo[:, Vb, oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,feBiNj->mBij', H.aa.oovv[:, Oa, va, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNeF,FeBiNj->mBij', H.aa.oovv[:, Oa, va, Va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEBiNj->mBij', H.aa.oovv[:, Oa, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    I2B_ovoo[:, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,eBfijN->mBij', H.ab.oovv[:, Ob, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNEf,EBfijN->mBij', H.ab.oovv[:, Ob, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,eBFijN->mBij', H.ab.oovv[:, Ob, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mNEF,EBFijN->mBij', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    # mbIJ
    I2B_ovoo[:, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnEf,EfbnIJ->mbIJ', H.aa.oovv[:, oa, Va, va], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEbnIJ->mbIJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EfbINJ->mbIJ', H.aa.oovv[:, Oa, Va, va], T.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEbINJ->mbIJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvOOO, optimize=True)
    )
    I2B_ovoo[:, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mneF,eFbInJ->mbIJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mnEf,EbfInJ->mbIJ', H.ab.oovv[:, ob, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('mnEF,EFbInJ->mbIJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNeF,eFbINJ->mbIJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EbfINJ->mbIJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EFbINJ->mbIJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    # mbiJ
    I2B_ovoo[:, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,FebinJ->mbiJ', H.aa.oovv[:, oa, va, Va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNeF,FebiNJ->mbiJ', H.aa.oovv[:, Oa, va, Va], T.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEbinJ->mbiJ', H.aa.oovv[:, oa, Va, Va], T.aab.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEbiNJ->mbiJ', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvoOO, optimize=True)
    )
    I2B_ovoo[:, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnEf,EbfinJ->mbiJ', H.ab.oovv[:, ob, Va, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EbfiNJ->mbiJ', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mneF,eFbinJ->mbiJ', H.ab.oovv[:, ob, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,eFbiNJ->mbiJ', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mnEF,EFbinJ->mbiJ', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNEF,EFbiNJ->mbiJ', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvoOO, optimize=True)
    )
    # mbIj
    I2B_ovoo[:, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnEf,EfbnIj->mbIj', H.aa.oovv[:, oa, Va, va], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('mnEF,FEbnIj->mbIj', H.aa.oovv[:, oa, Va, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EfbINj->mbIj', H.aa.oovv[:, Oa, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEbINj->mbIj', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    I2B_ovoo[:, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mneF,eFbInj->mbIj', H.ab.oovv[:, ob, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mNeF,eFbIjN->mbIj', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mnEf,EbfInj->mbIj', H.ab.oovv[:, ob, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('mnEF,EFbInj->mbIj', H.ab.oovv[:, ob, Va, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNEf,EbfIjN->mbIj', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mNEF,EFbIjN->mbIj', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    # mbij
    I2B_ovoo[:, vb, oa, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfbiNj->mbij', H.aa.oovv[:, Oa, Va, va], T.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEbiNj->mbij', H.aa.oovv[:, Oa, Va, Va], T.aab.VVvoOo, optimize=True)
    )
    I2B_ovoo[:, vb, oa, oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNeF,eFbijN->mbij', H.ab.oovv[:, Ob, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNEf,EbfijN->mbij', H.ab.oovv[:, Ob, Va, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EFbijN->mbij', H.ab.oovv[:, Ob, Va, Vb], T.abb.VVvooO, optimize=True)
    )

    ##### I2B_vvvo #####
    I2B_vvvo = H.ab.vvvo.copy()
    # ABeJ
    I2B_vvvo[Va, Vb, :, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,AfBmnJ->ABeJ', H.aa.oovv[oa, oa, :, va], T.aab.VvVooO, optimize=True)
            + 0.5 * np.einsum('mneF,FABmnJ->ABeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnef,AfBnMJ->ABeJ', H.aa.oovv[Oa, oa, :, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MneF,FABnMJ->ABeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNef,AfBMNJ->ABeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.VvVOOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FABMNJ->ABeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVVOOO, optimize=True)
    )
    I2B_vvvo[Va, Vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnef,ABfmnJ->ABeJ', H.ab.oovv[oa, ob, :, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mneF,ABFmnJ->ABeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('mNef,ABfmNJ->ABeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,ABFmNJ->ABeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mnef,ABfMnJ->ABeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MneF,ABFMnJ->ABeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('MNef,ABfMNJ->ABeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,ABFMNJ->ABeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVVOOO, optimize=True)
    )
    # AbeJ
    I2B_vvvo[Va, vb, :, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,AfbmnJ->AbeJ', H.aa.oovv[oa, oa, :, va], T.aab.VvvooO, optimize=True)
            + 0.5 * np.einsum('mneF,FAbmnJ->AbeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnef,AfbnMJ->AbeJ', H.aa.oovv[Oa, oa, :, va], T.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNef,AfbMNJ->AbeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MneF,FAbnMJ->AbeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FAbMNJ->AbeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVvOOO, optimize=True)
    )
    I2B_vvvo[Va, vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnef,AbfmnJ->AbeJ', H.ab.oovv[oa, ob, :, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNef,AbfmNJ->AbeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mneF,AFbmnJ->AbeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNeF,AFbmNJ->AbeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnef,AbfMnJ->AbeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MNef,AbfMNJ->AbeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MneF,AFbMnJ->AbeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MNeF,AFbMNJ->AbeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVvOOO, optimize=True)
    )
    # aBeJ
    I2B_vvvo[va, Vb, :, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,faBmnJ->aBeJ', H.aa.oovv[oa, oa, :, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('Mnef,faBnMJ->aBeJ', H.aa.oovv[Oa, oa, :, va], T.aab.vvVoOO, optimize=True)
            + 0.5 * np.einsum('MNef,faBMNJ->aBeJ', H.aa.oovv[Oa, Oa, :, va], T.aab.vvVOOO, optimize=True)
            + 0.5 * np.einsum('mneF,FaBmnJ->aBeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MneF,FaBnMJ->aBeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FaBMNJ->aBeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvVOOO, optimize=True)
    )
    I2B_vvvo[va, Vb, :, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnef,aBfmnJ->aBeJ', H.ab.oovv[oa, ob, :, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNef,aBfmNJ->aBeJ', H.ab.oovv[oa, Ob, :, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnef,aBfMnJ->aBeJ', H.ab.oovv[Oa, ob, :, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNef,aBfMNJ->aBeJ', H.ab.oovv[Oa, Ob, :, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mneF,aBFmnJ->aBeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mNeF,aBFmNJ->aBeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MneF,aBFMnJ->aBeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MNeF,aBFMNJ->aBeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVVOOO, optimize=True)
    )
    # abeJ
    I2B_vvvo[va, vb, :, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mneF,FabmnJ->abeJ', H.aa.oovv[oa, oa, :, Va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MneF,FabnMJ->abeJ', H.aa.oovv[Oa, oa, :, Va], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FabMNJ->abeJ', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvvOOO, optimize=True)
    )
    I2B_vvvo[va, vb, :, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,aFbmnJ->abeJ', H.ab.oovv[oa, ob, :, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mNeF,aFbmNJ->abeJ', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,aFbMnJ->abeJ', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MNeF,aFbMNJ->abeJ', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVvOOO, optimize=True)
    )
    # ABej
    I2B_vvvo[Va, Vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnef,AfBnMj->ABej', H.aa.oovv[Oa, oa, :, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MneF,FABnMj->ABej', H.aa.oovv[Oa, oa, :, Va], T.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNef,AfBMNj->ABej', H.aa.oovv[Oa, Oa, :, va], T.aab.VvVOOo, optimize=True)
            + 0.5 * np.einsum('MNeF,FABMNj->ABej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVVOOo, optimize=True)
    )
    I2B_vvvo[Va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNef,ABfmjN->ABej', H.ab.oovv[oa, Ob, :, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNeF,ABFmjN->ABej', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnef,ABfMnj->ABej', H.ab.oovv[Oa, ob, :, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MneF,ABFMnj->ABej', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('MNef,ABfMjN->ABej', H.ab.oovv[Oa, Ob, :, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MNeF,ABFMjN->ABej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVVOoO, optimize=True)
    )
    # Abej
    I2B_vvvo[Va, vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnef,AfbnMj->Abej', H.aa.oovv[Oa, oa, :, va], T.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('MNef,AfbMNj->Abej', H.aa.oovv[Oa, Oa, :, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MneF,FAbnMj->Abej', H.aa.oovv[Oa, oa, :, Va], T.aab.VVvoOo, optimize=True)
            + 0.5 * np.einsum('MNeF,FAbMNj->Abej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VVvOOo, optimize=True)
    )
    I2B_vvvo[Va, vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNef,AbfmjN->Abej', H.ab.oovv[oa, Ob, :, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnef,AbfMnj->Abej', H.ab.oovv[Oa, ob, :, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNef,AbfMjN->Abej', H.ab.oovv[Oa, Ob, :, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('mNeF,AFbmjN->Abej', H.ab.oovv[oa, Ob, :, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MneF,AFbMnj->Abej', H.ab.oovv[Oa, ob, :, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MNeF,AFbMjN->Abej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.VVvOoO, optimize=True)
    )
    # aBej
    I2B_vvvo[va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Mnef,faBnMj->aBej', H.aa.oovv[Oa, oa, :, va], T.aab.vvVoOo, optimize=True)
            + 0.5 * np.einsum('MNef,faBMNj->aBej', H.aa.oovv[Oa, Oa, :, va], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MneF,FaBnMj->aBej', H.aa.oovv[Oa, oa, :, Va], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNeF,FaBMNj->aBej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvVOOo, optimize=True)
    )
    I2B_vvvo[va, Vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNef,aBfmjN->aBej', H.ab.oovv[oa, Ob, :, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('Mnef,aBfMnj->aBej', H.ab.oovv[Oa, ob, :, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MNef,aBfMjN->aBej', H.ab.oovv[Oa, Ob, :, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mNeF,aBFmjN->aBej', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MneF,aBFMnj->aBej', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNeF,aBFMjN->aBej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVVOoO, optimize=True)
    )
    # abej
    I2B_vvvo[va, vb, :, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNeF,FabmNj->abej', H.aa.oovv[oa, Oa, :, Va], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNeF,FabMNj->abej', H.aa.oovv[Oa, Oa, :, Va], T.aab.VvvOOo, optimize=True)
    )
    I2B_vvvo[va, vb, :, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MneF,aFbMnj->abej', H.ab.oovv[Oa, ob, :, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNeF,aFbmjN->abej', H.ab.oovv[oa, Ob, :, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MNeF,aFbMjN->abej', H.ab.oovv[Oa, Ob, :, Vb], T.abb.vVvOoO, optimize=True)
    )

    ##### I2B_vooo #####
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    # AmIJ
    I2B_vooo[Va, :, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfe,AfenIJ->AmIJ', H.ab.oovv[oa, :, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('nmFe,FAenIJ->AmIJ', H.ab.oovv[oa, :, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('nmfE,AfEnIJ->AmIJ', H.ab.oovv[oa, :, va, Vb], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('nmFE,FAEnIJ->AmIJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Nmfe,AfeINJ->AmIJ', H.ab.oovv[Oa, :, va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('NmFe,FAeINJ->AmIJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('NmfE,AfEINJ->AmIJ', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('NmFE,FAEINJ->AmIJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    I2B_vooo[Va, :, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('nmfe,AefInJ->AmIJ', H.bb.oovv[ob, :, vb, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('nmfE,AEfInJ->AmIJ', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('nmFE,AEFInJ->AmIJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('Nmfe,AefINJ->AmIJ', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('NmfE,AEfINJ->AmIJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('NmFE,AEFINJ->AmIJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    # AmiJ
    I2B_vooo[Va, :, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,AfeinJ->AmiJ', H.ab.oovv[oa, :, va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Nmfe,AfeiNJ->AmiJ', H.ab.oovv[Oa, :, va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nmFe,FAeinJ->AmiJ', H.ab.oovv[oa, :, Va, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmFe,FAeiNJ->AmiJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('nmfE,AfEinJ->AmiJ', H.ab.oovv[oa, :, va, Vb], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('NmfE,AfEiNJ->AmiJ', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('nmFE,FAEinJ->AmiJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('NmFE,FAEiNJ->AmiJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    I2B_vooo[Va, :, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('nmfe,AefinJ->AmiJ', H.bb.oovv[ob, :, vb, vb], T.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('Nmfe,AefiNJ->AmiJ', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nmfE,AEfinJ->AmiJ', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmfE,AEfiNJ->AmiJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('nmFE,AEFinJ->AmiJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('NmFE,AEFiNJ->AmiJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    # AmIj
    I2B_vooo[Va, :, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfe,AfenIj->AmIj', H.ab.oovv[oa, :, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nmFe,FAenIj->AmIj', H.ab.oovv[oa, :, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('nmfE,AfEnIj->AmIj', H.ab.oovv[oa, :, va, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('nmFE,FAEnIj->AmIj', H.ab.oovv[oa, :, Va, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Nmfe,AfeINj->AmIj', H.ab.oovv[Oa, :, va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NmFe,FAeINj->AmIj', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('NmfE,AfEINj->AmIj', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('NmFE,FAEINj->AmIj', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    I2B_vooo[Va, :, Oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('nmfe,AefInj->AmIj', H.bb.oovv[ob, :, vb, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('nmfE,AEfInj->AmIj', H.bb.oovv[ob, :, vb, Vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('nmFE,AEFInj->AmIj', H.bb.oovv[ob, :, Vb, Vb], T.abb.VVVOoo, optimize=True)
            + 0.5 * np.einsum('Nmfe,AefIjN->AmIj', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('NmfE,AEfIjN->AmIj', H.bb.oovv[Ob, :, vb, Vb], T.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('NmFE,AEFIjN->AmIj', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    # Amij
    I2B_vooo[Va, :, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Nmfe,AfeiNj->Amij', H.ab.oovv[Oa, :, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NmfE,AfEiNj->Amij', H.ab.oovv[Oa, :, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NmFe,FAeiNj->Amij', H.ab.oovv[Oa, :, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('NmFE,FAEiNj->Amij', H.ab.oovv[Oa, :, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    I2B_vooo[Va, :, oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('Nmfe,AefijN->Amij', H.bb.oovv[Ob, :, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NmFe,AFeijN->Amij', H.bb.oovv[Ob, :, Vb, vb], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('NmFE,AEFijN->Amij', H.bb.oovv[Ob, :, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    # amIJ
    I2B_vooo[va, :, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmFe,FaenIJ->amIJ', H.ab.oovv[oa, :, Va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('nmfE,faEnIJ->amIJ', H.ab.oovv[oa, :, va, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('nmFE,FaEnIJ->amIJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('NmFe,FaeINJ->amIJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('NmfE,faEINJ->amIJ', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('NmFE,FaEINJ->amIJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    I2B_vooo[va, :, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfE,aEfInJ->amIJ', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('nmFE,aEFInJ->amIJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('NmfE,aEfINJ->amIJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('NmFE,aEFINJ->amIJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    # amIj
    I2B_vooo[va, :, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmFe,FaenIj->amIj', H.ab.oovv[oa, :, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nmfE,faEnIj->amIj', H.ab.oovv[oa, :, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('nmFE,FaEnIj->amIj', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NmFe,FaeINj->amIj', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NmfE,faEINj->amIj', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('NmFE,FaEINj->amIj', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    I2B_vooo[va, :, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfE,aEfInj->amIj', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('nmFE,aEFInj->amIj', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('NmfE,aEfIjN->amIj', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('NmFE,aEFIjN->amIj', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    # amiJ
    I2B_vooo[va, :, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFe,FaeinJ->amiJ', H.ab.oovv[oa, :, Va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('nmfE,faEinJ->amiJ', H.ab.oovv[oa, :, va, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('nmFE,FaEinJ->amiJ', H.ab.oovv[oa, :, Va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('NmFe,FaeiNJ->amiJ', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NmfE,faEiNJ->amiJ', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('NmFE,FaEiNJ->amiJ', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    I2B_vooo[va, :, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfE,aEfinJ->amiJ', H.bb.oovv[ob, :, vb, Vb], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('nmFE,aEFinJ->amiJ', H.bb.oovv[ob, :, Vb, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('NmfE,aEfiNJ->amiJ', H.bb.oovv[Ob, :, vb, Vb], T.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('NmFE,aEFiNJ->amiJ', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVoOO, optimize=True)
    )
    # amij
    I2B_vooo[va, :, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('NmfE,faEiNj->amij', H.ab.oovv[Oa, :, va, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('NmFe,FaeiNj->amij', H.ab.oovv[Oa, :, Va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NmFE,FaEiNj->amij', H.ab.oovv[Oa, :, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    I2B_vooo[va, :, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('NmFe,aFeijN->amij', H.bb.oovv[Ob, :, Vb, vb], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('NmFE,aEFijN->amij', H.bb.oovv[Ob, :, Vb, Vb], T.abb.vVVooO, optimize=True)
    )

    ##### I2B_vvov #####
    I2B_vvov = H.ab.vvov.copy()
    # ABIe
    I2B_vvov[Va, Vb, Oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,AfBnIm->ABIe', H.ab.oovv[oa, ob, va, :], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('nMfe,AfBnIM->ABIe', H.ab.oovv[oa, Ob, va, :], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('Nmfe,AfBINm->ABIe', H.ab.oovv[Oa, ob, va, :], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('NMfe,AfBINM->ABIe', H.ab.oovv[Oa, Ob, va, :], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('nmFe,FABnIm->ABIe', H.ab.oovv[oa, ob, Va, :], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('nMFe,FABnIM->ABIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FABINm->ABIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FABINM->ABIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVVOOO, optimize=True)
    )
    I2B_vvov[Va, Vb, Oa, :]  += (1.0 / 1.0) * (
            +0.5 * np.einsum('nmfe,ABfInm->ABIe', H.bb.oovv[ob, ob, vb, :], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('Nmfe,ABfImN->ABIe', H.bb.oovv[Ob, ob, vb, :], T.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('NMfe,ABfINM->ABIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('nmFe,ABFInm->ABIe', H.bb.oovv[ob, ob, Vb, :], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('NmFe,ABFImN->ABIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVVOoO, optimize=True)
            + 0.5 * np.einsum('NMFe,ABFINM->ABIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVVOOO, optimize=True)
    )
    # AbIe
    I2B_vvov[Va, vb, Oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,AfbnIm->AbIe', H.ab.oovv[oa, ob, va, :], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nMfe,AfbnIM->AbIe', H.ab.oovv[oa, Ob, va, :], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nmFe,FAbnIm->AbIe', H.ab.oovv[oa, ob, Va, :], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('nMFe,FAbnIM->AbIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Nmfe,AfbINm->AbIe', H.ab.oovv[Oa, ob, va, :], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NMfe,AfbINM->AbIe', H.ab.oovv[Oa, Ob, va, :], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FAbINm->AbIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FAbINM->AbIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVvOOO, optimize=True)
    )
    I2B_vvov[Va, vb, Oa, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('nmfe,AbfInm->AbIe', H.bb.oovv[ob, ob, vb, :], T.abb.VvvOoo, optimize=True)
            - 0.5 * np.einsum('nmFe,AFbInm->AbIe', H.bb.oovv[ob, ob, Vb, :], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('Nmfe,AbfImN->AbIe', H.bb.oovv[Ob, ob, vb, :], T.abb.VvvOoO, optimize=True)
            + 0.5 * np.einsum('NMfe,AbfINM->AbIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('NmFe,AFbImN->AbIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('NMFe,AFbINM->AbIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVvOOO, optimize=True)
    )
    # aBIe
    I2B_vvov[va, Vb, Oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfe,faBnIm->aBIe', H.ab.oovv[oa, ob, va, :], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('nMfe,faBnIM->aBIe', H.ab.oovv[oa, Ob, va, :], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Nmfe,faBINm->aBIe', H.ab.oovv[Oa, ob, va, :], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('NMfe,faBINM->aBIe', H.ab.oovv[Oa, Ob, va, :], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('nmFe,FaBnIm->aBIe', H.ab.oovv[oa, ob, Va, :], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('nMFe,FaBnIM->aBIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FaBINm->aBIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FaBINM->aBIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvVOOO, optimize=True)
    )
    I2B_vvov[va, Vb, Oa, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('nmfe,aBfInm->aBIe', H.bb.oovv[ob, ob, vb, :], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('Nmfe,aBfImN->aBIe', H.bb.oovv[Ob, ob, vb, :], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('NMfe,aBfINM->aBIe', H.bb.oovv[Ob, Ob, vb, :], T.abb.vVvOOO, optimize=True)
            + 0.5 * np.einsum('nmFe,aBFInm->aBIe', H.bb.oovv[ob, ob, Vb, :], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NmFe,aBFImN->aBIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVVOoO, optimize=True)
            + 0.5 * np.einsum('NMFe,aBFINM->aBIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVVOOO, optimize=True)
    )
    # abIe
    I2B_vvov[va, vb, Oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFe,FabnIm->abIe', H.ab.oovv[oa, ob, Va, :], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('nMFe,FabnIM->abIe', H.ab.oovv[oa, Ob, Va, :], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FabINm->abIe', H.ab.oovv[Oa, ob, Va, :], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FabINM->abIe', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvvOOO, optimize=True)
    )
    I2B_vvov[va, vb, Oa, :] += (1.0 / 1.0) * (
            -0.5 * np.einsum('nmFe,aFbInm->abIe', H.bb.oovv[ob, ob, Vb, :], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NmFe,aFbImN->abIe', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('NMFe,aFbINM->abIe', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVvOOO, optimize=True)
    )
    # ABie
    I2B_vvov[Va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfe,AfBinM->ABie', H.ab.oovv[oa, Ob, va, :], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('nMFe,FABinM->ABie', H.ab.oovv[oa, Ob, Va, :], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Nmfe,AfBiNm->ABie', H.ab.oovv[Oa, ob, va, :], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NMfe,AfBiNM->ABie', H.ab.oovv[Oa, Ob, va, :], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FABiNm->ABie', H.ab.oovv[Oa, ob, Va, :], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FABiNM->ABie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVVoOO, optimize=True)
    )
    I2B_vvov[Va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Nmfe,ABfimN->ABie', H.bb.oovv[Ob, ob, vb, :], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('NMfe,ABfiNM->ABie', H.bb.oovv[Ob, Ob, vb, :], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NmFe,ABFimN->ABie', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVVooO, optimize=True)
            + 0.5 * np.einsum('NMFe,ABFiNM->ABie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVVoOO, optimize=True)
    )
    # Abie
    I2B_vvov[Va, vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfe,AfbinM->Abie', H.ab.oovv[oa, Ob, va, :], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('nMFe,FAbinM->Abie', H.ab.oovv[oa, Ob, Va, :], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('Nmfe,AfbiNm->Abie', H.ab.oovv[Oa, ob, va, :], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NmFe,FAbiNm->Abie', H.ab.oovv[Oa, ob, Va, :], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('NMfe,AfbiNM->Abie', H.ab.oovv[Oa, Ob, va, :], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFe,FAbiNM->Abie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VVvoOO, optimize=True)
    )
    I2B_vvov[Va, vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Nmfe,AbfimN->Abie', H.bb.oovv[Ob, ob, vb, :], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('NmFe,AFbimN->Abie', H.bb.oovv[Ob, ob, Vb, :], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('NMfe,AbfiNM->Abie', H.bb.oovv[Ob, Ob, vb, :], T.abb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('NMFe,AFbiNM->Abie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.VVvoOO, optimize=True)
    )
    # aBie
    I2B_vvov[va, Vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nMfe,faBinM->aBie', H.ab.oovv[oa, Ob, va, :], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('Nmfe,faBiNm->aBie', H.ab.oovv[Oa, ob, va, :], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('NMfe,faBiNM->aBie', H.ab.oovv[Oa, Ob, va, :], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('nMFe,FaBinM->aBie', H.ab.oovv[oa, Ob, Va, :], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('NmFe,FaBiNm->aBie', H.ab.oovv[Oa, ob, Va, :], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FaBiNM->aBie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvVoOO, optimize=True)
    )
    I2B_vvov[va, Vb, oa, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Nmfe,aBfimN->aBie', H.bb.oovv[Ob, ob, vb, :], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('NMfe,aBfiNM->aBie', H.bb.oovv[Ob, Ob, vb, :], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('NmFe,aBFimN->aBie', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVVooO, optimize=True)
            + 0.5 * np.einsum('NMFe,aBFiNM->aBie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVVoOO, optimize=True)
    )
    # abie
    I2B_vvov[va, vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nMFe,FabinM->abie', H.ab.oovv[oa, Ob, Va, :], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('NmFe,FabiNm->abie', H.ab.oovv[Oa, ob, Va, :], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NMFe,FabiNM->abie', H.ab.oovv[Oa, Ob, Va, :], T.aab.VvvoOO, optimize=True)
    )
    I2B_vvov[va, vb, oa, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NmFe,aFbimN->abie', H.bb.oovv[Ob, ob, Vb, :], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('NMFe,aFbiNM->abie', H.bb.oovv[Ob, Ob, Vb, :], T.abb.vVvoOO, optimize=True)
    )

    #########################################################################
    ############################ bb intermdiates ############################
    #########################################################################

    ##### I2C_vooo #####
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    # AmIJ
    I2C_vooo[Vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnef,AfenIJ->AmIJ', H.bb.oovv[:, ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNef,AfeIJN->AmIJ', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mneF,FAenIJ->AmIJ', H.bb.oovv[:, ob, vb, Vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEAnIJ->AmIJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,FAeIJN->AmIJ', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAIJN->AmIJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    I2C_vooo[Vb, :, Ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,fAenIJ->AmIJ', H.ab.oovv[oa, :, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nmfE,fEAnIJ->AmIJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('Nmfe,fAeNIJ->AmIJ', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NmfE,fEANIJ->AmIJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('nmFe,FAenIJ->AmIJ', H.ab.oovv[oa, :, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('nmFE,FEAnIJ->AmIJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NmFe,FAeNIJ->AmIJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('NmFE,FEANIJ->AmIJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    # AmiJ
    I2C_vooo[Vb, :, ob, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,AfeinJ->AmiJ', H.bb.oovv[:, ob, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mnEf,EAfinJ->AmiJ', H.bb.oovv[:, ob, Vb, vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEAinJ->AmiJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
            - 0.5 * np.einsum('mNef,AfeiJN->AmiJ', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfiJN->AmiJ', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAiJN->AmiJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    I2C_vooo[Vb, :, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,fAeniJ->AmiJ', H.ab.oovv[oa, :, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nmFe,FAeniJ->AmiJ', H.ab.oovv[oa, :, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('nmfE,fEAniJ->AmiJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nmFE,FEAniJ->AmiJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nmfe,fAeNiJ->AmiJ', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NmFe,FAeNiJ->AmiJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('NmfE,fEANiJ->AmiJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('NmFE,FEANiJ->AmiJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    I2C_vooo[Vb, :, Ob, ob] = -1.0*np.transpose(I2C_vooo[Vb, :, ob, Ob], (0, 1, 3, 2))
    # Amij
    I2C_vooo[Vb, :, ob, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,AfeijN->Amij', H.bb.oovv[:, Ob, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,FAeijN->Amij', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAijN->Amij', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    I2C_vooo[Vb, :, ob, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Nmfe,fAeNij->Amij', H.ab.oovv[Oa, :, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NmfE,fEANij->Amij', H.ab.oovv[Oa, :, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('NmFe,FAeNij->Amij', H.ab.oovv[Oa, :, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('NmFE,FEANij->Amij', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    # amIJ
    I2C_vooo[vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mneF,FeanIJ->amIJ', H.bb.oovv[:, ob, vb, Vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mnEF,FEanIJ->amIJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FeaIJN->amIJ', H.bb.oovv[:, Ob, vb, Vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaIJN->amIJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    I2C_vooo[vb, :, Ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfE,fEanIJ->amIJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nmFe,FeanIJ->amIJ', H.ab.oovv[oa, :, Va, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nmFE,FEanIJ->amIJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NmfE,fEaNIJ->amIJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NmFe,FeaNIJ->amIJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('NmFE,FEaNIJ->amIJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    # amiJ
    I2C_vooo[vb, :, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnEf,EfainJ->amiJ', H.bb.oovv[:, ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mnEF,FEainJ->amiJ', H.bb.oovv[:, ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNEf,EfaiJN->amiJ', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaiJN->amiJ', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    I2C_vooo[vb, :, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFe,FeaniJ->amiJ', H.ab.oovv[oa, :, Va, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NmFe,FeaNiJ->amiJ', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('nmfE,fEaniJ->amiJ', H.ab.oovv[oa, :, va, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nmFE,FEaniJ->amiJ', H.ab.oovv[oa, :, Va, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmfE,fEaNiJ->amiJ', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFE,FEaNiJ->amiJ', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    I2C_vooo[vb, :, Ob, ob] = -1.0*np.transpose(I2C_vooo[vb, :, ob, Ob], (0, 1, 3, 2))
    # amij
    I2C_vooo[vb, :, ob, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaijN->amij', H.bb.oovv[:, Ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaijN->amij', H.bb.oovv[:, Ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    I2C_vooo[vb, :, ob, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('NmFe,FeaNij->amij', H.ab.oovv[Oa, :, Va, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('NmfE,fEaNij->amij', H.ab.oovv[Oa, :, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NmFE,FEaNij->amij', H.ab.oovv[Oa, :, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    ##### I2C_vvov #####
    I2C_vvov = H.bb.vvov.copy()
    # ABIe
    I2C_vvov[Vb, Vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,BAfmnI->ABIe', H.bb.oovv[ob, ob, :, vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('mneF,FBAmnI->ABIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnef,BAfnIM->ABIe', H.bb.oovv[Ob, ob, :, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNef,BAfIMN->ABIe', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MneF,FBAnIM->ABIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FBAIMN->ABIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVVOOO, optimize=True)
    )
    I2C_vvov[Vb, Vb, Ob, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfe,fBAnmI->ABIe', H.ab.oovv[oa, ob, va, :], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nmFe,FBAnmI->ABIe', H.ab.oovv[oa, ob, Va, :], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Nmfe,fBANmI->ABIe', H.ab.oovv[Oa, ob, va, :], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('NmFe,FBANmI->ABIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('nMfe,fBAnIM->ABIe', H.ab.oovv[oa, Ob, va, :], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nMFe,FBAnIM->ABIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NMfe,fBANIM->ABIe', H.ab.oovv[Oa, Ob, va, :], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NMFe,FBANIM->ABIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVVOOO, optimize=True)
    )
    # AbIe
    I2C_vvov[Vb, vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnef,AfbmnI->AbIe', H.bb.oovv[ob, ob, :, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('mneF,FAbmnI->AbIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnef,AfbnIM->AbIe', H.bb.oovv[Ob, ob, :, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,FAbnIM->AbIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNef,AfbIMN->AbIe', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNeF,FAbIMN->AbIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVvOOO, optimize=True)
    )
    I2C_vvov[Vb, vb, Ob, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfe,fAbnmI->AbIe', H.ab.oovv[oa, ob, va, :], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nmFe,FAbnmI->AbIe', H.ab.oovv[oa, ob, Va, :], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Nmfe,fAbNmI->AbIe', H.ab.oovv[Oa, ob, va, :], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NmFe,FAbNmI->AbIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('nMfe,fAbnIM->AbIe', H.ab.oovv[oa, Ob, va, :], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nMFe,FAbnIM->AbIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NMfe,fAbNIM->AbIe', H.ab.oovv[Oa, Ob, va, :], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NMFe,FAbNIM->AbIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVvOOO, optimize=True)
    )
    I2C_vvov[vb, Vb, Ob, :] = -1.0*np.transpose(I2C_vvov[Vb, vb, Ob, :], (1, 0, 2, 3))
    # abIe
    I2C_vvov[vb, vb, Ob, :] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mneF,FbamnI->abIe', H.bb.oovv[ob, ob, :, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MneF,FbanIM->abIe', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FbaIMN->abIe', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VvvOOO, optimize=True)
    )
    I2C_vvov[vb, vb, Ob, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFe,FbanmI->abIe', H.ab.oovv[oa, ob, Va, :], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NmFe,FbaNmI->abIe', H.ab.oovv[Oa, ob, Va, :], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('nMFe,FbanIM->abIe', H.ab.oovv[oa, Ob, Va, :], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFe,FbaNIM->abIe', H.ab.oovv[Oa, Ob, Va, :], T.abb.VvvOOO, optimize=True)
    )
    # ABie
    I2C_vvov[Vb, Vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Mnef,BAfinM->ABie', H.bb.oovv[Ob, ob, :, vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNef,BAfiMN->ABie', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,FBAinM->ABie', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FBAiMN->ABie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVVoOO, optimize=True)
    )
    I2C_vvov[Vb, Vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Nmfe,fBANim->ABie', H.ab.oovv[Oa, ob, va, :], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('NmFe,FBANim->ABie', H.ab.oovv[Oa, ob, Va, :], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('nMfe,fBAniM->ABie', H.ab.oovv[oa, Ob, va, :], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('NMfe,fBANiM->ABie', H.ab.oovv[Oa, Ob, va, :], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('nMFe,FBAniM->ABie', H.ab.oovv[oa, Ob, Va, :], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('NMFe,FBANiM->ABie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVVOoO, optimize=True)
    )
    # Abie
    I2C_vvov[Vb, vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,AfbimN->Abie', H.bb.oovv[ob, Ob, :, vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNef,AfbiMN->Abie', H.bb.oovv[Ob, Ob, :, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FAbimN->Abie', H.bb.oovv[ob, Ob, :, Vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNeF,FAbiMN->Abie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VVvoOO, optimize=True)
    )
    I2C_vvov[Vb, vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfe,fAbniM->Abie', H.ab.oovv[oa, Ob, va, :], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nMFe,FAbniM->Abie', H.ab.oovv[oa, Ob, Va, :], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Nmfe,fAbNim->Abie', H.ab.oovv[Oa, ob, va, :], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NMfe,fAbNiM->Abie', H.ab.oovv[Oa, Ob, va, :], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFe,FAbNim->Abie', H.ab.oovv[Oa, ob, Va, :], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('NMFe,FAbNiM->Abie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VVvOoO, optimize=True)
    )
    I2C_vvov[vb, Vb, ob, :] = -1.0*np.transpose(I2C_vvov[Vb, vb, ob, :], (1, 0, 2, 3))
    # abie
    I2C_vvov[vb, vb, ob, :] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MneF,FbainM->abie', H.bb.oovv[Ob, ob, :, Vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FbaiMN->abie', H.bb.oovv[Ob, Ob, :, Vb], T.bbb.VvvoOO, optimize=True)
    )
    I2C_vvov[vb, vb, ob, :] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NmFe,FbaNim->abie', H.ab.oovv[Oa, ob, Va, :], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('nMFe,FbaniM->abie', H.ab.oovv[oa, Ob, Va, :], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('NMFe,FbaNiM->abie', H.ab.oovv[Oa, Ob, Va, :], T.abb.VvvOoO, optimize=True)
    )

    H.aa.vooo = I2A_vooo
    H.aa.vvov = I2A_vvov
    H.ab.ovoo = I2B_ovoo
    H.ab.vvvo = I2B_vvvo
    H.ab.vooo = I2B_vooo
    H.ab.vvov = I2B_vvov
    H.bb.vooo = I2C_vooo
    H.bb.vvov = I2C_vvov

    return H