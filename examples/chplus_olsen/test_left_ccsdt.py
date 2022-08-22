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

    #X = build_left_ccsdt_intermediates_v2(L, T, system)

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

    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)
    X.aa.vovv = -0.5 * np.einsum("abcijk,ecik->ejab", L.aaa, T.aa, optimize=True)
    x1a = np.einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov, optimize=True)
    X.aa.ooov = 0.5 * np.einsum("abcijk,bcnk->jina", L.aaa, T.aa, optimize=True)
    x1a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    X.a.vv = -(1.0 / 12.0) * np.einsum("abcijk,ebcijk->ea", L.aaa, T.aaa, optimize=True)
    x1a += np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    X.aa.vvvv = (1.0 / 6.0) * np.einsum("abcijk,efcijk->efab", L.aaa, T.aaa, optimize=True)
    x1a -= 0.5 * np.einsum("bife,efab->ai", H.aa.vovv, X.aa.vvvv, optimize=True)
    X.aa.voov = 0.25 * np.einsum("abcijk,fbcnjk->fina", L.aaa, T.aaa, optimize=True)
    x1a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    X.ab.ovvo = 0.25 * np.einsum("abcijk,bcfjkn->ifan", L.aaa, T.aab, optimize=True)
    x1a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)

    print("Error on h3a_ovvooo = ",np.linalg.norm(X1A.flatten() - x1a.flatten()))

    # -1/2 * h3b(ifgmno) * l3b(mnoafg)
    I3B_ovvooo = (
          np.einsum("bmje,ecik->mbcijk", H.aa.voov, T.ab, optimize=True)
         + 0.5 * np.einsum("mcek,beji->mbcijk", H.ab.ovvo, T.aa, optimize=True)
         + np.einsum("mcie,bejk->mbcijk", H.ab.ovov, T.ab, optimize=True)
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
    X.aa.vovv = -np.einsum('abfimn,efmn->eiba', L.aab, T.ab, optimize=True)
    x1a = np.einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov, optimize=True)
    X.ab.vovv = -0.5 * np.einsum('bfamni,efmn->eiba', L.aab, T.aa, optimize=True)
    x1a += np.einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo, optimize=True)
    X.ab.ovvv = -np.einsum('afbinm,fenm->ieab', L.aab, T.ab, optimize=True)
    x1a += np.einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov, optimize=True)
    X.aa.ooov = + np.einsum('aefijn,efmn->jima', L.aab, T.ab, optimize=True)
    x1a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    X.ab.oovo = np.einsum('afeinj,fenm->ijam', L.aab, T.ab, optimize=True)
    x1a += np.einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo, optimize=True)
    X.a.vv = - 0.5 * np.einsum("efgmno,afgmno->ae", L.aab, T.aab, optimize=True)
    x1a += np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    X.aa.vvvv = + 0.5 * np.einsum("abcijk,efcijk->efab", L.aab, T.aab, optimize=True)
    x1a -= 0.5 * np.einsum("bife,efab->ai", H.aa.vovv, X.aa.vvvv, optimize=True)
    X.ab.vvvv = 0.5 * np.einsum("egfmon,agbmon->abef", L.aab, T.aab, optimize=True)
    x1a -= np.einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv, optimize=True)
    X.aa.voov = + np.einsum("efgmno,afgino->amie", L.aab, T.aab, optimize=True)
    x1a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    X.ab.ovvo = + np.einsum("efgmno,fganoi->maei", L.aab, T.abb, optimize=True)
    x1a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    X.ab.vovo = -0.5 * np.einsum("gefonm,agfnoi->amei", L.aab, T.aab, optimize=True)
    x1a -= np.einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo, optimize=True)

    print("Error on h3b_ovvooo = ", np.linalg.norm(X1A.flatten() - x1a.flatten()))

    # -1/4 h3c(ifgmno) * l3c(mnoafg)
    I3C_ovvooo = (
        np.einsum("mcek,ebij->mbcijk", H.ab.ovvo, T.ab, optimize=True)
        + 0.5 * np.einsum("mbie,ecjk->mbcijk", H.ab.ovov, T.bb, optimize=True)
        - 0.5 * np.einsum("mnij,bcnk->mbcijk", H.ab.oooo, T.bb, optimize=True)
        + 0.25 * np.einsum("me,ebcijk->mbcijk", H.a.ov, T.abb, optimize=True)
        + 0.25 * np.einsum("mnif,fbcnjk->mbcijk", H.aa.ooov, T.abb, optimize=True)
        + 0.25 * np.einsum("mnif,fbcnjk->mbcijk", H.ab.ooov, T.bbb, optimize=True)
        - 0.5 * np.einsum("mnej,ebcink->mbcijk", H.ab.oovo, T.abb, optimize=True)
        + 0.5 * np.einsum("mbef,efcijk->mbcijk", H.ab.ovvv, T.abb, optimize=True)
    )
    I3C_ovvooo -= np.transpose(I3C_ovvooo, (0, 2, 1, 3, 4, 5))
    I3C_ovvooo -= np.transpose(I3C_ovvooo, (0, 1, 2, 3, 5, 4))
    X1A = -0.25 * np.einsum("ifgmno,afgmno->ai", I3C_ovvooo, L.abb, optimize=True)

    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)
    X.ab.vovv = - np.einsum('bfamni,efmn->eiba', L.abb, T.ab, optimize=True)
    x1a = np.einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo, optimize=True)
    X.ab.ovvv = -0.5 * np.einsum('afbinm,fenm->ieab', L.abb, T.bb, optimize=True)
    x1a += np.einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov, optimize=True)
    X.ab.oovo = + 0.5 * np.einsum('afeinj,fenm->ijam', L.abb, T.bb, optimize=True)
    x1a += np.einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo, optimize=True)
    X.a.vv = - 0.25 * np.einsum("efgmno,afgmno->ae", L.abb, T.abb, optimize=True)
    x1a += np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    X.aa.voov = + 0.25 * np.einsum("efgmno,afgino->amie", L.abb, T.abb, optimize=True)
    x1a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    X.ab.ovvo = + 0.25 * np.einsum("efgmno,fganoi->maei", L.abb, T.bbb, optimize=True)
    x1a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    X.ab.vovo = -0.5 * np.einsum("egfnom,agfnoi->amei", L.abb, T.abb, optimize=True)
    x1a -= np.einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo, optimize=True)
    X.ab.vvvv = + 0.5 * np.einsum("egfmon,agbmon->abef", L.abb, T.abb, optimize=True)
    x1a -= np.einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv, optimize=True)

    print("Error on h3c_ovvooo = ", np.linalg.norm(X1A.flatten() - x1a.flatten()))

    # 1/12 * h3a(efgano) * l3a(inoefg)
    I3A_vvvvoo = (
        - (6.0 / 12.0) * np.einsum("cmke,abmj->abcejk", H.aa.voov, T.aa, optimize=True)
        + (3.0 / 12.0) * np.einsum("abef,fcjk->abcejk", H.aa.vvvv, T.aa, optimize=True)
        - (1.0 / 12.0) * np.einsum("me,abcmjk->abcejk", H.a.ov, T.aaa, optimize=True)
        + (3.0 / 12.0) * np.einsum("anef,fbcnjk->abcejk", H.aa.vovv, T.aaa, optimize=True)
        + (3.0 / 12.0) * np.einsum("anef,bcfjkn->abcejk", H.ab.vovv, T.aab, optimize=True)
        + (2.0 / 24.0) * np.einsum("nmje,abcmnk->abcejk", H.aa.ooov, T.aaa, optimize=True)
    )
    I3A_vvvvoo -= np.transpose(I3A_vvvvoo, (0, 1, 2, 3, 5, 4))
    I3A_vvvvoo -= np.transpose(I3A_vvvvoo, (1, 0, 2, 3, 4, 5))
    I3A_vvvvoo -= np.transpose(I3A_vvvvoo, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvvoo, (0, 2, 1, 3, 4, 5))
    X1A = (1.0 / 12.0) * np.einsum("efgano,efgino->ai", I3A_vvvvoo, L.aaa, optimize=True)

    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)
    X.aa.ooov = 0.5 * np.einsum("abcijk,bcnk->jina", L.aaa, T.aa, optimize=True)
    x1a = -np.einsum("imne,enma->ai", X.aa.ooov, H.aa.voov, optimize=True)
    X.aa.vovv = -0.5 * np.einsum("abcijk,ecik->ejab", L.aaa, T.aa, optimize=True)
    x1a -= 0.5 * np.einsum("gife,efag->ai", X.aa.vovv, H.aa.vvvv, optimize=True)
    X.a.oo = (1.0 / 12.0) * np.einsum("efgmno,efgino->mi", L.aaa, T.aaa, optimize=True)
    x1a -= np.einsum("ma,im->ai", H.a.ov, X.a.oo, optimize=True)
    X.aa.voov = 0.25 * np.einsum("abcijk,fbcnjk->fina", L.aaa, T.aaa, optimize=True)
    x1a += np.einsum("fmae,eimf->ai", H.aa.vovv, X.aa.voov, optimize=True)
    X.ab.ovvo = 0.25 * np.einsum("abcijk,bcfjkn->ifan", L.aaa, T.aab, optimize=True)
    x1a += np.einsum("fmae,iefm->ai", H.ab.vovv, X.ab.ovvo, optimize=True)
    X.aa.oooo = (1.0 / 6.0) * np.einsum("efgmno,efgijo->mnij", L.aaa, T.aaa, optimize=True)
    x1a += 0.5 * np.einsum("nmoa,iomn->ai", H.aa.ooov, X.aa.oooo, optimize=True)

    print("Error on h3a_vvvvoo = ", np.linalg.norm(X1A.flatten() - x1a.flatten()))

    # 1/2 h3b(efgano) * l3b(inoefg)
    I3B_vvvvoo = (
        - np.einsum("bmje,acmk->abcejk", H.aa.voov, T.ab, optimize=True)
        - 0.5 * np.einsum("mcek,abmj->abcejk", H.ab.ovvo, T.aa, optimize=True)
        # - np.einsum("amek,bcjm->abcejk", H.ab.vovo, T.ab, optimize=True)
        # + np.einsum("acef,bfjk->abcejk", H.ab.vvvv, T.ab, optimize=True)
        # + 0.5 * np.einsum("abef,fcjk->abcejk", H.aa.vvvv, T.ab, optimize=True)
        # - 0.5 * np.einsum("me,abcmjk->abcejk", H.a.ov, T.aab, optimize=True)
        # + np.einsum("anef,fbcnjk->abcejk", H.aa.vovv, T.aab, optimize=True)
        # + np.einsum("anef,bfcjnk->abcejk", H.ab.vovv, T.abb, optimize=True)
        # - 0.5 * np.einsum("mcef,abfmjk->abcejk", H.ab.ovvv, T.aab, optimize=True)
        # + 0.25 * np.einsum("nmje,abcmnk->abcejk", H.aa.ooov, T.aab, optimize=True)
        # + 0.5 * np.einsum("mnek,abcmjn->abcejk", H.ab.oovo, T.aab, optimize=True)
    )
    I3B_vvvvoo -= np.transpose(I3B_vvvvoo, (1, 0, 2, 3, 4, 5))
    X1A = 0.5 * np.einsum("efgano,efgino->ai", I3B_vvvvoo, L.aab, optimize=True)

    X = Integral.from_empty(system, 2, data_type=T.a.dtype, use_none=True)

    X.aa.ooov = + np.einsum('aefijn,efmn->jima', L.aab, T.ab, optimize=True)
    x1a = -np.einsum("imne,enma->ai", X.aa.ooov, H.aa.voov, optimize=True)
    X.ab.ooov = 0.5 * np.einsum('efajni,efmn->jima', L.aab, T.aa, optimize=True)
    x1a -= np.einsum("imne,neam->ai", X.ab.ooov, H.ab.ovvo, optimize=True)


    print("Error on h3b_vvvvoo = ", np.linalg.norm(X1A.flatten() - x1a.flatten()))