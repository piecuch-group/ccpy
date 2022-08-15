import numpy as np

from ccpy.models.integrals import Integral

def build_left_ccsd_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-CCSD equations"""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=T.a.oo.dtype, use_none=True)

    # (L2 * T3)_C


    return X


def build_left_ccsdt_intermediates(L, T, system):
    """Calculate the L*T intermediates used in the left-CCSDT equations"""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=T.a.oo.dtype, use_none=True)

    # (L2 * T2)_C

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
                np.einsum('efajni,efmn->jima', L.aab, T.aa, optimize=True)
              + 0.5 * np.einsum('efajni,efmn->jima', L.abb, T.ab, optimize=True)
    )

    X.bb.ooov = (
                0.5 * np.einsum('aefijn,efmn->jima', L.bbb, T.bb, optimize=True)
                + np.einsum('efanji,fenm->jima', L.abb, T.bb, optimize=True)
    )

    X.aa.ovvv = (
                -0.5 * np.einsum('abfimn,efmn->ieab', L.aaa, T.aa, optimize=True)
                - np.einsum('abfimn,efmn->ieab', L.aab, T.ab, optimize=True)
    )

    X.ab.vovv = (
                -0.5 * np.einsum('bfamni,efmn->eiba', L.aab, T.aa, optimize=True)
                - np.einsum('bfamni,efmn->eiba', L.abb, T.ab, optimize=True)
    )

    X.ab.ovvv = (
                -np.einsum('afbinm,fenm->ieab', L.aab, T.ab, optimize=True)
                -0.5 * np.einsum('afbinm,fenm->ieab', L.abb, T.bb, optimize=True)
    )

    X.bb.ovvv = (
                -0.5 * np.einsum('abfimn,efmn->ieab', L.bbb, T.bb, optimize=True)
                - np.einsum('fbanmi,fenm->ieab', L.abb, T.ab, optimize=True)
    )

    # (L3 * T3)_C
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
              + 0.5 * np.einsum("egfmon,egfioj->mnij", L.abb, T.abb, optimize=True
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
            0.25 * np.einsum("efgmno,fganoi->amie", L.aaa, T.aab, optimize=True)
            + np.einsum("efgmno,fganoi->amie", L.aab, T.abb, optimize=True)
            + 0.25 * np.einsum("efgmno,fganoi->amie", L.abb, T.bbb, optimize=True)
    )
    X.bb.voov = (
            0.25 * np.einsum("efgmno,afgino->amie", L.bbb, T.bbb, optimize=True)
            + np.einsum("gfeonm,gfaoni->amie", L.abb, T.abb, optimize=True)
            + 0.25 * np.einsum("gfeonm,gfaoni->amie", L.aab, T.aab, optimize=True)
    )

    return X
