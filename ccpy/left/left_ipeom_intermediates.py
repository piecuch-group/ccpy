import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import leftipeom3_p_intermediates

def get_leftipeom3_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-IP-EOMCCSD(3h-2p) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(e)
    X["a"]["v"] = (
        -0.5 * ccpy_einsum("mfn,efmn->e", L.aa, T.aa)
        - ccpy_einsum("mfn,efmn->e", L.ab, T.ab)
    )

    # x2a(abj), a, b->-j
    X["aa"]["vvo"] = (
        -0.5 * ccpy_einsum("mbfjn,afmn->abj", L.aaa, T.aa)
        - ccpy_einsum("mbfjn,afmn->abj", L.aab, T.ab)
    )

    # x2a(ijk), i, k->-j
    X["aa"]["ooo"] = (
        0.5 * ccpy_einsum("iefjn,efkn->ijk", L.aaa, T.aa)
        + ccpy_einsum("iefjn,efkn->ijk", L.aab, T.ab)
    )

    # x2b(ij~k~), i, (k~)->-(j~)
    X["ab"]["ooo"] = (
        ccpy_einsum("ifenj,fenk->ijk", L.aab, T.ab)
        + 0.5 * ccpy_einsum("iefjn,efkn->ijk", L.abb, T.bb)
    )

    # x2b(ab~j~), a, (b~)->(j~)
    X["ab"]["vvo"] = (
        -0.5 * ccpy_einsum("mfbnj,afmn->abj", L.aab, T.aa)
        - ccpy_einsum("mfbnj,afmn->abj", L.abb, T.ab)

    )

    # x2b(ic~b~), i, (b~)->-(c~)
    X["ab"]["ovv"] = (
        -ccpy_einsum("ifbnm,fcnm->icb", L.aab, T.ab)
        - 0.5 * ccpy_einsum("ifbnm,fcnm->icb", L.abb, T.bb)
    )
    return X

def get_leftipeom3_p_intermediates(L, l3_excitations, T, do_l3, system):
    """Calculate the L*T intermediates used in the left-IP-EOMCCSD(3h-2p) equations"""

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(e)
    X["a"]["v"] = (
        -0.5 * ccpy_einsum("mfn,efmn->e", L.aa, T.aa)
        - ccpy_einsum("mfn,efmn->e", L.ab, T.ab)
    )
    # x2a(abj)
    X["aa"]["vvo"] = leftipeom3_p_intermediates.get_x2a_vvo(L.aaa, l3_excitations["aaa"],
                                                                                       L.aab, l3_excitations["aab"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aaa"], do_l3["aab"])
    # x2a(ijk)
    X["aa"]["ooo"] = leftipeom3_p_intermediates.get_x2a_ooo(L.aaa, l3_excitations["aaa"],
                                                                                       L.aab, l3_excitations["aab"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aaa"], do_l3["aab"])
    # x2b(ab~j~)
    X["ab"]["vvo"] = leftipeom3_p_intermediates.get_x2b_vvo(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.aa, T.ab,
                                                                                       do_l3["aab"], do_l3["abb"])
    # x2b(ic~b~)
    X["ab"]["ovv"] = leftipeom3_p_intermediates.get_x2b_ovv(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.ab, T.bb,
                                                                                       do_l3["aab"], do_l3["abb"])
    # x2b(ij~k~)
    X["ab"]["ooo"] = leftipeom3_p_intermediates.get_x2b_ooo(L.aab, l3_excitations["aab"],
                                                                                       L.abb, l3_excitations["abb"],
                                                                                       T.ab, T.bb,
                                                                                       do_l3["aab"], do_l3["abb"])
    return X
