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

    # (L2 * T3)_C

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

    return X
