import numpy as np
from ccpy.models.integrals import Integral
from ccpy.utilities.updates import leftccsdt_p_intermediates


def build_left_ccsdt_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-CCSDT equations"""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)

    # (L2 * T3)_C
    X.a.vo = (
               0.25 * np.einsum("efmn,aefimn->ai", L.aa, T.aaa, optimize=True)
             + np.einsum("efmn,aefimn->ai", L.ab, T.aab, optimize=True)
             + 0.25 * np.einsum("efmn,aefimn->ai", L.bb, T.abb, optimize=True)
    )

    X.b.vo = (
               0.25 * np.einsum("efmn,efamni->ai", L.aa, T.aab, optimize=True)
               + np.einsum("efmn,efamni->ai", L.ab, T.abb, optimize=True)
               + 0.25 * np.einsum("efmn,efamni->ai", L.bb, T.bbb, optimize=True)
    )

    # (L3 * T2)_C
    X.aa.ooov = (
                0.5 * np.einsum('aefijn,efmn->jima', L.aaa, T.aa, optimize=True)
                + np.einsum('aefijn,efmn->jima', L.aab, T.ab, optimize=True)
    )

    X.ab.oovo = (
                np.einsum('afeinj,fenm->ijam', L.aab, T.ab, optimize=True)
               + 0.5 * np.einsum('afeinj,fenm->ijam', L.abb, T.bb, optimize=True)
    )

    X.ab.ooov = (
                0.5 * np.einsum('efajni,efmn->jima', L.aab, T.aa, optimize=True)
              + np.einsum('efajni,efmn->jima', L.abb, T.ab, optimize=True)
    )

    X.bb.ooov = (
                0.5 * np.einsum('aefijn,efmn->jima', L.bbb, T.bb, optimize=True)
                + np.einsum('feanji,fenm->jima', L.abb, T.ab, optimize=True)
    )

    X.aa.vovv = (
                -0.5 * np.einsum('abfimn,efmn->eiba', L.aaa, T.aa, optimize=True)
                - np.einsum('abfimn,efmn->eiba', L.aab, T.ab, optimize=True)
    )

    X.ab.vovv = (
                -0.5 * np.einsum('bfamni,efmn->eiba', L.aab, T.aa, optimize=True)
                - np.einsum('bfamni,efmn->eiba', L.abb, T.ab, optimize=True)
    )

    X.ab.ovvv = (
                -np.einsum('afbinm,fenm->ieab', L.aab, T.ab, optimize=True)
                -0.5 * np.einsum('afbinm,fenm->ieab', L.abb, T.bb, optimize=True)
    )

    X.bb.vovv = (
                -0.5 * np.einsum('abfimn,efmn->eiba', L.bbb, T.bb, optimize=True)
                - np.einsum('fbanmi,fenm->eiba', L.abb, T.ab, optimize=True)
    )

    # (L3 * T3)_C
    # l = 3, h = 4 -> s = -1
    X.a.oo = (
              (1.0 / 12.0) * np.einsum("efgmno,efgino->mi", L.aaa, T.aaa, optimize=True)
              + 0.5 * np.einsum("efgmno,efgino->mi", L.aab, T.aab, optimize=True)
              + 0.25 * np.einsum("efgmno,efgino->mi", L.abb, T.abb, optimize=True)
    )
    X.b.oo = (
              (1.0 / 12.0) * np.einsum("efgmno,efgino->mi", L.bbb, T.bbb, optimize=True)
              + 0.5 * np.einsum("gfeonm,gfeoni->mi", L.abb, T.abb, optimize=True)
              + 0.25 * np.einsum("gfeonm,gfeoni->mi", L.aab, T.aab, optimize=True)
    )

    # l = 3, h = 3 -> s = +1
    X.a.vv = (
            -(1.0 / 12.0) * np.einsum("efgmno,afgmno->ae", L.aaa, T.aaa, optimize=True)
            - 0.5 * np.einsum("efgmno,afgmno->ae", L.aab, T.aab, optimize=True)
            - 0.25 * np.einsum("efgmno,afgmno->ae", L.abb, T.abb, optimize=True)
    )
    X.b.vv = (
            -(1.0 / 12.0) * np.einsum("efgmno,afgmno->ae", L.bbb, T.bbb, optimize=True)
            - 0.5 * np.einsum("gfeonm,gfaonm->ae", L.abb, T.abb, optimize=True)
            - 0.25 * np.einsum("gfeonm,gfaonm->ae", L.aab, T.aab, optimize=True)
    )

    X.aa.oooo = (
                (1.0 / 6.0) * np.einsum("efgmno,efgijo->mnij", L.aaa, T.aaa, optimize=True)
                + 0.5 * np.einsum("efgmno,efgijo->mnij", L.aab, T.aab, optimize=True)
    )
    X.ab.oooo = (
                0.5 * np.einsum("egfmon,egfioj->mnij", L.aab, T.aab, optimize=True)
              + 0.5 * np.einsum("egfmon,egfioj->mnij", L.abb, T.abb, optimize=True)
    )
    X.bb.oooo = (
                (1.0 / 6.0) * np.einsum("efgmno,efgijo->mnij", L.bbb, T.bbb, optimize=True)
                + 0.5 * np.einsum("gfeonm,gfeoji->mnij", L.abb, T.abb, optimize=True)
    )

    X.aa.vvvv = (
            (1.0 / 6.0) * np.einsum("efgmno,abgmno->abef", L.aaa, T.aaa, optimize=True)
            + 0.5 * np.einsum("efgmno,abgmno->abef", L.aab, T.aab, optimize=True)
    )
    X.ab.vvvv = (
            0.5 * np.einsum("egfmon,agbmon->abef", L.aab, T.aab, optimize=True)
            + 0.5 * np.einsum("egfmon,agbmon->abef", L.abb, T.abb, optimize=True)
    )
    X.bb.vvvv = (
            (1.0 / 6.0) * np.einsum("efgmno,abgmno->abef", L.bbb, T.bbb, optimize=True)
            + 0.5 * np.einsum("gfeonm,gbaonm->abef", L.abb, T.abb, optimize=True)
    )

    X.aa.voov = (
            0.25 * np.einsum("efgmno,afgino->amie", L.aaa, T.aaa, optimize=True)
            + np.einsum("efgmno,afgino->amie", L.aab, T.aab, optimize=True)
            + 0.25 * np.einsum("efgmno,afgino->amie", L.abb, T.abb, optimize=True)
    )
    X.ab.voov = (
            0.25 * np.einsum("fgenom,afgino->amie", L.aab, T.aaa, optimize=True)
            + np.einsum("fgenom,afgino->amie", L.abb, T.aab, optimize=True)
            + 0.25 * np.einsum("efgmno,afgino->amie", L.bbb, T.abb, optimize=True)
    )
    X.ab.vovo = (
            -0.5 * np.einsum("gefonm,agfnoi->amei", L.aab, T.aab, optimize=True)
            -0.5 * np.einsum("egfnom,agfnoi->amei", L.abb, T.abb, optimize=True)
    )
    X.ab.ovov = (
            -0.5 * np.einsum("fgemon,fgaion->maie", L.aab, T.aab, optimize=True)
            -0.5 * np.einsum("fgemon,fgaion->maie", L.abb, T.abb, optimize=True)
    )
    X.ab.ovvo = (
            0.25 * np.einsum("efgmno,fganoi->maei", L.aaa, T.aab, optimize=True)
            + np.einsum("efgmno,fganoi->maei", L.aab, T.abb, optimize=True)
            + 0.25 * np.einsum("efgmno,fganoi->maei", L.abb, T.bbb, optimize=True)
    )
    X.bb.voov = (
            0.25 * np.einsum("efgmno,afgino->amie", L.bbb, T.bbb, optimize=True)
            + np.einsum("gfeonm,gfaoni->amie", L.abb, T.abb, optimize=True)
            + 0.25 * np.einsum("gfeonm,gfaoni->amie", L.aab, T.aab, optimize=True)
    )

    return X

def build_left_ccsdt_p_intermediates(L, l3_excitations, T, t3_excitations, system, RHF_symmetry=False):
    """Calculate the L*T intermediates used in the left-CCSDT equations"""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)
    ## a intermediates ##
    X.a.vo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1a_vo(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aa, L.ab, L.bb,
    )
    X.a.oo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1a_oo(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    X.a.vv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1a_vv(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    ## aa intermediates ##
    X.aa.ooov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2a_ooov(
                                                               T.aa, T.ab,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
    )
    X.aa.vovv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2a_vovv(
                                                               T.aa, T.ab,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
    )
    X.aa.oooo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2a_oooo(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               noa, nua, nob, nub,
    )
    X.aa.vvvv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2a_vvvv(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               noa, nua, nob, nub,
    )
    X.aa.voov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2a_voov(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    ## ab intermediates ##
    X.ab.oovo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_oovo(
                                                               T.ab, T.bb,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
    )
    X.ab.ooov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_ooov(
                                                               T.aa, T.ab,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
    )
    X.ab.ovvv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_ovvv(
                                                               T.ab, T.bb,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
    )
    X.ab.vovv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_vovv(
                                                               T.aa, T.ab,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
    )
    X.ab.oooo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_oooo(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    X.ab.vvvv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_vvvv(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    X.ab.voov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_voov(
                                                               T.aaa, t3_excitations["aaa"].T,
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
    )
    X.ab.ovvo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_ovvo(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.aaa, l3_excitations["aaa"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    X.ab.vovo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_vovo(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    X.ab.ovov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2b_ovov(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               noa, nua, nob, nub,
    )
    if RHF_symmetry:
        X.b.vo = X.a.vo.copy()
        X.b.oo = X.a.oo.copy()
        X.b.vv = X.b.vv.copy()
        X.bb.ooov = X.aa.ooov.copy()
        X.bb.vovv = X.aa.vovv.copy()
        X.bb.oooo = X.aa.oooo.copy()
        X.bb.vvvv = X.aa.vvvv.copy()
        X.bb.voov = X.aa.voov.copy()
    else:
        ## b intermediates ##
        X.b.vo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1b_vo(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.aa, L.ab, L.bb,
        )
        X.b.oo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1b_oo(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
        )
        X.b.vv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x1b_vv(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
        )
        ## bb intermediates ##
        X.bb.ooov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2c_ooov(
                                                               T.ab, T.bb,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
        )
        X.bb.vovv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2c_vovv(
                                                               T.ab, T.bb,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
        )
        X.bb.oooo = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2c_oooo(
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
        )
        X.bb.vvvv = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2c_vvvv(
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
        )
        X.bb.voov = leftccsdt_p_intermediates.leftccsdt_p_intermediates.compute_x2c_voov(
                                                               T.aab, t3_excitations["aab"].T,
                                                               T.abb, t3_excitations["abb"].T,
                                                               T.bbb, t3_excitations["bbb"].T,
                                                               L.aab, l3_excitations["aab"].T,
                                                               L.abb, l3_excitations["abb"].T,
                                                               L.bbb, l3_excitations["bbb"].T,
                                                               noa, nua, nob, nub,
        )

    return X
