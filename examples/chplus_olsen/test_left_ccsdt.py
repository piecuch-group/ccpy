import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.models.operators import ClusterOperator
from ccpy.models.integrals import Integral

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt

from ccpy.left.left_cc_intermediates import build_left_ccsdt_intermediates_v2

if __name__ == "__main__":

    system, H = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=3,
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsdt(T, H)

    L = ClusterOperator(system, 3)
    L.unflatten(T.flatten()[:L.ndim])

    X = build_left_ccsdt_intermediates_v2(L, T, system)
    #X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)

    # -1/12 * h3a(ifgmno) * l3a(mnoafg)
    I3A_ovvooo = (
        0.5 * np.einsum("bmje,ecik->mbcijk", H.aa.voov, T.aa, optimize=True)
        - 0.25 * np.einsum("mnij,bcnk->mbcijk", H.aa.oooo, T.aa, optimize=True)
        + (1.0 / 12.0) * np.einsum("me,ebcijk->mbcijk", H.a.ov, T.aaa, optimize=True)
        + (1.0 / 12.0) * np.einsum("bmfe,efcijk->mbcijk", H.aa.vovv, T.aaa, optimize=True)
        + 0.25 * np.einsum("mnif,fbcnjk->mbcijk", H.aa.ooov, T.aaa, optimize=True)
        + 0.25 * np.einsum("mnif,bcfjkn->mbcijk", H.ab.ooov, T.aab, optimize=True)
    )
    I3A_ovvooo -= np.transpose(I3A_ovvooo, (0, 2, 1, 3, 4, 5))
    I3A_ovvooo -= np.transpose(I3A_ovvooo, (0, 1, 2, 4, 3, 5))
    I3A_ovvooo -= np.transpose(I3A_ovvooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_ovvooo, (0, 1, 2, 3, 5, 4))
    X1A = -(1.0 / 12.0) * np.einsum("ifgmno,afgmno->ai", I3A_ovvooo, L.aaa, optimize=True)

    x1a = np.einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov, optimize=True)
    x1a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    x1a += np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    x1a -= 0.5 * np.einsum("gifa,efag->ai", H.aa.vovv, X.aa.vvvv, optimize=True)
    x1a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    x1a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    print("Error on h3a_ovvooo = ",np.linalg.norm(X1A.flatten() - x1a.flatten()))

    # -1/2 * h3b(ifgmno) * l3b(mnoafg)
    I3B_ovvooo = (
          np.einsum("bmje,ecik->mbcijk", H.aa.voov, T.ab, optimize=True)
        + 0.5 * np.einsum("mcek,beji->mbcijk", H.ab.ovvo, T.aa, optimize=True)
        + np.einsum("mcie,bcjk->mbcijk", H.ab.ovov, T.ab, optimize=True)
        - 0.5 * np.einsum("mnij,bcnk->mbcijk", H.aa.oooo, T.ab, optimize=True)
        - np.einsum("mnik,bcjn->mbcijk", H.ab.oooo, T.ab, optimize=True)
        + 0.5 * np.einsum("me,ebcijk->mbcijk", H.a.ov, T.aab, optimize=True)
        + 0.25 * np.einsum("bmfe,efcijk->mbcijk", H.aa.vovv, T.aab, optimize=True)
        + 0.5 * np.einsum("mcef,ebfijk->mbcijk", H.ab.ovvv, T.aab, optimize=True)
        + np.einsum("mnif,bfcjnk->mbcijk", H.aa.ooov, T.aab, optimize=True)
        + np.einsum("mnif,bfcjnk->mbcijk", H.ab.ooov, T.abb, optimize=True)
        - 0.5 * np.einsum("mnek,ebcijn->mbcijk", H.ab.oovo, T.aab, optimize=True)
    )
    I3B_ovvooo -= np.transpose(I3B_ovvooo, (0, 1, 2, 4, 3, 5))
    X1A = -(1.0 / 2.0) * np.einsum("ifgmno,afgmno->ai", I3B_ovvooo, L.aab, optimize=True)

    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)
