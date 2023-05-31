import numpy as np

from ccpy.models.integrals import Integral


def get_deaeom4_intermediates(H, R, T, system):

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype)

    # x(mb~)
    X.a.ov = (
            np.einsum("mbef,ef->mb", H.ab.ovvv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,ebfn->mb", H.aa.oovv, R.aba, optimize=True)
            + np.einsum("mnef,ebfn->mb", H.ab.oovv, R.abb, optimize=True)
    )
    # x(am~)
    X.a.vo = (
            np.einsum("amef,ef->am", H.ab.vovv, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,aefn->am", H.bb.oovv, R.abb, optimize=True)
            + np.einsum("nmfe,aefn->am", H.ab.oovv, R.aba, optimize=True)
    )

    return X
