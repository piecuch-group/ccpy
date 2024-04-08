import numpy as np

def get_lefteaeom3_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-EA-EOMCCSD(3p-2h) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(m)
    X["a"]["o"] = (
        0.5 * np.einsum("efn,efmn->m", L.aa, T.aa, optimize=True)
        + np.einsum("efn,efmn->m", L.ab, T.ab, optimize=True)
    )

    # x2a(ibj)
    X["aa"]["ovo"] = (
        0.5 * np.einsum("ebfjn,efin->ibj", L.aaa, T.aa, optimize=True)
        + np.einsum("ebfjn,efin->ibj", L.aab, T.ab, optimize=True)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
        -0.5 * np.einsum("aefmn,bfmn->abe", L.aaa, T.aa, optimize=True)
        - np.einsum("aefmn,bfmn->abe", L.aab, T.ab, optimize=True)
    )

    # x2b(ab~e~)
    X["ab"]["vvv"] = (
        -np.einsum("afenm,fbnm->abe", L.aab, T.ab, optimize=True)
        - 0.5 * np.einsum("afenm,bfmn->abe", L.abb, T.bb, optimize=True)
    )

    # x2b(ib~j~)
    X["ab"]["ovo"] = (
        0.5 * np.einsum("efbnj,efin->ibj", L.aab, T.aa, optimize=True)
        + np.einsum("ebfjn,efin->ibj", L.abb, T.ab, optimize=True)
    )

    # x2b(ak~m~)
    X["ab"]["voo"] = (
        np.einsum("afenk,fenm->akm", L.aab, T.ab, optimize=True)
        + 0.5 * np.einsum("afenk,fenm->akm", L.abb, T.bb, optimize=True)
    )
    return X
