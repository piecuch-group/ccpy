import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import lefteaeom3_p_intermediates

def get_lefteaeom3_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-EA-EOMCCSD(3p-2h) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(m)
    X["a"]["o"] = (
        0.5 * ccpy_einsum("efn,efmn->m", L.aa, T.aa)
        + ccpy_einsum("efn,efmn->m", L.ab, T.ab)
    )

    # x2a(ibj)
    X["aa"]["ovo"] = (
        0.5 * ccpy_einsum("ebfjn,efin->ibj", L.aaa, T.aa)
        + ccpy_einsum("ebfjn,efin->ibj", L.aab, T.ab)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
        -0.5 * ccpy_einsum("aefmn,bfmn->abe", L.aaa, T.aa)
        - ccpy_einsum("aefmn,bfmn->abe", L.aab, T.ab)
    )

    # x2b(ab~e~)
    X["ab"]["vvv"] = (
        -ccpy_einsum("afenm,fbnm->abe", L.aab, T.ab)
        - 0.5 * ccpy_einsum("afenm,bfmn->abe", L.abb, T.bb)
    )

    # x2b(ib~j~)
    X["ab"]["ovo"] = (
        0.5 * ccpy_einsum("efbnj,efin->ibj", L.aab, T.aa)
        + ccpy_einsum("ebfjn,efin->ibj", L.abb, T.ab)
    )

    # x2b(ak~m~)
    X["ab"]["voo"] = (
        ccpy_einsum("afenk,fenm->akm", L.aab, T.ab)
        + 0.5 * ccpy_einsum("afenk,fenm->akm", L.abb, T.bb)
    )
    return X

def get_lefteaeom3_p_intermediates(L, l3_excitations, T, do_l3, system):
    """Calculate the L*T intermediates used in the left-EA-EOMCCSD(3p-2h) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(m)
    X["a"]["o"] = (
        0.5 * ccpy_einsum("efn,efmn->m", L.aa, T.aa)
        + ccpy_einsum("efn,efmn->m", L.ab, T.ab)
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
