import numpy as np
from ccpy.lib.core import lefteaeom3_p_intermediates

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

def get_lefteaeom3_p_intermediates(L, l3_excitations, T, do_l3, system):
    """Calculate the L*T intermediates used in the left-EA-EOMCCSD(3p-2h) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(m)
    X["a"]["o"] = (
        0.5 * np.einsum("efn,efmn->m", L.aa, T.aa, optimize=True)
        + np.einsum("efn,efmn->m", L.ab, T.ab, optimize=True)
    )
    # x2a(ibj)
    X["aa"]["ovo"] = lefteaeom3_p_intermediates.get_x2a_ovo(L.aaa, l3_excitations["aaa"],
                                                                                       L.aab, l3_excitations["aab"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aaa"], do_l3["aab"])
    # x2a(abe)
    X["aa"]["vvv"] = lefteaeom3_p_intermediates.get_x2a_vvv(L.aaa, l3_excitations["aaa"],
                                                                                       L.aab, l3_excitations["aab"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aaa"], do_l3["aab"])
    # x2b(ib~j~)
    X["ab"]["ovo"] = lefteaeom3_p_intermediates.get_x2b_ovo(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aab"], do_l3["abb"])
    # x2b(ak~m~)
    X["ab"]["voo"] = lefteaeom3_p_intermediates.get_x2b_voo(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.ab, T.bb,
                                                                                       do_l3["aab"], do_l3["abb"])
    # x2b(ae~b~)
    X["ab"]["vvv"] = lefteaeom3_p_intermediates.get_x2b_vvv(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.ab, T.bb,
                                                                                       do_l3["aab"], do_l3["abb"])
    return X
