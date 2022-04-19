import numpy as np
from ccpy.models.integrals import Integral

def calc_rdm1(T, L, system):

    rdm1 = Integral.from_empty(system, 1, use_none=True)


    # HF 1-RDM: 2 in each doubly occoupied orbital, 1 in each singly occupied orbital (done in a naive way)

    # oo block: <j|gamma|i> => i->-.->-j
    rdm1.a.oo = (
                    -np.einsum('ei,ej->ij', L.a, T.a)
                    - 0.5 * np.einsum('efin,efjn->ij', L.aa, T.aa)
                    - np.einsum('efin,efjn->ij', L.ab, T.ab)
        )
    rdm1.b.oo = (
                    -np.einsum('ei,ej->ij', L.b, T.b)
                    - 0.5 * np.einsum('efin,efjn->ij', L.bb, T.bb)
                    - np.einsum('feni,fenj->ij', L.ab, T.ab)
        )

    rdm1.a.oo += np.eye(system.noccupied_alpha, dtype=rdm1.a.oo.dtype)
    rdm1.b.oo += np.eye(system.noccupied_beta, dtype=rdm1.a.oo.dtype)

    # vv block: <a|gamma|b> => a-<-.-<-b
    rdm1.a.vv = (
                    np.einsum('bm,am->ab', L.a, T.a)
                    + 0.5 * np.einsum('bfmn,afmn->ab', L.aa, T.aa)
                    + np.einsum('bfmn,afmn->ab', L.ab, T.ab)
    )
    rdm1.b.vv = (
                    np.einsum('bm,am->ab', L.b, T.b)
                    + 0.5 * np.einsum('bfmn,afmn->ab', L.bb, T.bb)
                    + np.einsum('fbnm,fanm->ab', L.ab, T.ab)
    )

    # vo block: <i|gamma|a> ia>.
    rdm1.a.ov = L.a.transpose()
    rdm1.b.ov = L.b.transpose()

    # ov block: <a|gamma|i> .<ia
    rdm1.a.vo = (
                    np.einsum('em,aeim->ai', L.a, T.aa)
                    + np.einsum('em,aeim->ai', L.b, T.ab)
                    - np.einsum('em,ei,am->ai', L.a, T.a, T.a)
                    - 0.5 * np.einsum('efmn,afmn,ei->ai', L.aa, T.aa, T.a)
                    - np.einsum('efmn,afmn,ei->ai', L.ab, T.ab, T.a)
                    - 0.5 * np.einsum('efmn,efin,am->ai', L.aa, T.aa, T.a)
                    - np.einsum('efmn,efin,am->ai', L.ab, T.ab, T.a)
        )
    rdm1.b.vo = (
                    np.einsum('em,aeim->ai', L.b, T.bb)
                    + np.einsum('em,eami->ai', L.a, T.ab)
                    - np.einsum('em,ei,am->ai', L.b, T.b, T.b)
                    - 0.5 * np.einsum('efmn,afmn,ei->ai', L.bb, T.bb, T.b)
                    - np.einsum('fenm,fanm,ei->ai', L.ab, T.ab, T.b)
                    - 0.5 * np.einsum('efmn,efin,am->ai', L.bb, T.bb, T.b)
                    - np.einsum('fenm,feni,am->ai', L.ab, T.ab, T.b)
        )

    rdm1.a.vo += T.a
    rdm1.b.vo += T.b

    return rdm1