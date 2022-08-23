import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.models.operators import ClusterOperator

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt

from ccpy.left.left_cc_intermediates import build_left_ccsdt_intermediates

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
        RHF_symmetry=True,
    )

    T, total_energy, _ = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsdt(T, H)

    L = ClusterOperator(system, 3)
    L.unflatten(T.flatten()[:L.ndim])

    X = build_left_ccsdt_intermediates(L, T, system)

    ################ CHECKING UPDATE L1A ###################
    # [1] -1/12 * h3a(ifgmno) * l3a(mnoafg)
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

    # [2] -1/2 * h3b(ifgmno) * l3b(mnoafg)
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
    X1A -= (1.0 / 2.0) * np.einsum("ifgmno,afgmno->ai", I3B_ovvooo, L.aab, optimize=True)

    # [3] -1/4 h3c(ifgmno) * l3c(mnoafg)
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
    X1A -= 0.25 * np.einsum("ifgmno,afgmno->ai", I3C_ovvooo, L.abb, optimize=True)

    # [4] 1/12 * h3a(efgano) * l3a(inoefg)
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
    X1A += (1.0 / 12.0) * np.einsum("efgano,efgino->ai", I3A_vvvvoo, L.aaa, optimize=True)

    # [5] 1/2 h3b(efgano) * l3b(inoefg)
    I3B_vvvvoo = (
        - np.einsum("bmje,acmk->abcejk", H.aa.voov, T.ab, optimize=True)
        - 0.5 * np.einsum("mcek,abmj->abcejk", H.ab.ovvo, T.aa, optimize=True)
        - np.einsum("amek,bcjm->abcejk", H.ab.vovo, T.ab, optimize=True)
        + np.einsum("acef,bfjk->abcejk", H.ab.vvvv, T.ab, optimize=True)
        + 0.5 * np.einsum("abef,fcjk->abcejk", H.aa.vvvv, T.ab, optimize=True)
        - 0.5 * np.einsum("me,abcmjk->abcejk", H.a.ov, T.aab, optimize=True)
        + np.einsum("anef,fbcnjk->abcejk", H.aa.vovv, T.aab, optimize=True)
        + np.einsum("anef,bfcjnk->abcejk", H.ab.vovv, T.abb, optimize=True)
        - 0.5 * np.einsum("mcef,abfmjk->abcejk", H.ab.ovvv, T.aab, optimize=True)
        + 0.25 * np.einsum("nmje,abcmnk->abcejk", H.aa.ooov, T.aab, optimize=True)
        + 0.5 * np.einsum("mnek,abcmjn->abcejk", H.ab.oovo, T.aab, optimize=True)
    )
    I3B_vvvvoo -= np.transpose(I3B_vvvvoo, (1, 0, 2, 3, 4, 5))
    X1A += 0.5 * np.einsum("efgano,efgino->ai", I3B_vvvvoo, L.aab, optimize=True)

    # [6] 1/4 h3c(efgano) * l3c(inoefg)
    I3C_vvvvoo = (
        - np.einsum("mcek,abmj->abcejk", H.ab.ovvo, T.ab, optimize=True)
        - 0.5 * np.einsum("amej,bcmk->abcejk", H.ab.vovo, T.bb, optimize=True)
        + 0.5 * np.einsum("abef,fcjk->abcejk", H.ab.vvvv, T.bb, optimize=True)
        - 0.25 * np.einsum("me,abcmjk->abcejk", H.a.ov, T.abb, optimize=True)
        + 0.25 * np.einsum("anef,fbcnjk->abcejk", H.aa.vovv, T.abb, optimize=True)
        + 0.25 * np.einsum("anef,fbcnjk->abcejk", H.ab.vovv, T.bbb, optimize=True)
        - 0.5 * np.einsum("mbef,afcmjk->abcejk", H.ab.ovvv, T.abb, optimize=True)
        + 0.5 * np.einsum("mnej,abcmnk->abcejk", H.ab.oovo, T.abb, optimize=True)
    )
    I3C_vvvvoo -= np.transpose(I3C_vvvvoo, (0, 2, 1, 3, 4, 5))
    I3C_vvvvoo -= np.transpose(I3C_vvvvoo, (0, 1, 2, 3, 5, 4))
    X1A += 0.25 * np.einsum("efgano,efgino->ai", I3C_vvvvoo, L.abb, optimize=True)

    # 4-body HBar
    X1A += 0.25 * np.einsum("imae,efno,ghmp,fghnop->ai", H.aa.oovv, T.aa, T.aa, L.aaa, optimize=True)
    X1A += 0.5 * np.einsum("imae,efno,ghmp,fghnop->ai", H.aa.oovv, T.aa, T.ab, L.aab, optimize=True)
    X1A += 0.5 * np.einsum("imae,fgmn,ehop,fghnop->ai", H.aa.oovv, T.aa, T.ab, L.aab, optimize=True)
    X1A -= 0.5 * np.einsum("ioag,egmp,fhno,efhmnp->ai", H.ab.oovv, T.ab, T.ab, L.aab, optimize=True)
    X1A -= 0.5 * np.einsum("ioag,ehmo,fgnp,efhmnp->ai", H.ab.oovv, T.ab, T.ab, L.aab, optimize=True)
    X1A -= np.einsum("imae,egno,fhmp,fghnop->ai", H.aa.oovv, T.ab, T.ab, L.abb, optimize=True)
    X1A -= 0.5 * np.einsum("inaf,efmo,ghnp,eghmop->ai", H.ab.oovv, T.ab, T.bb, L.abb, optimize=True)
    X1A -= 0.5 * np.einsum("inaf,egmn,fhop,eghmop->ai", H.ab.oovv, T.ab, T.bb, L.abb, optimize=True)
    X1A += 0.25 * np.einsum("imae,efno,ghmp,fghnop->ai", H.ab.oovv, T.bb, T.bb, L.bbb, optimize=True)

    X1A += (1.0 / 12.0) * np.einsum("fiea,fghmno,eghmno->ai", H.aa.vovv, L.aaa, T.aaa, optimize=True)
    X1A += (1.0 / 2.0) * np.einsum("fiea,fghmno,eghmno->ai", H.aa.vovv, L.aab, T.aab, optimize=True)
    X1A += (1.0 / 4.0) * np.einsum("fiea,fghmno,eghmno->ai", H.aa.vovv, L.abb, T.abb, optimize=True)

    X1A += (1.0 / 12.0) * np.einsum("ifae,fghmno,eghmno->ai", H.ab.ovvv, L.bbb, T.bbb, optimize=True)
    X1A += (1.0 / 2.0) * np.einsum("ifae,hgfonm,hgeonm->ai", H.ab.ovvv, L.abb, T.abb, optimize=True)
    X1A += (1.0 / 4.0) * np.einsum("ifae,hgfonm,hgeonm->ai", H.ab.ovvv, L.aab, T.aab, optimize=True)

    X1A -= (1.0 / 12.0) * np.einsum("mina,efgnop,efgmop->ai", H.aa.ooov, L.aaa, T.aaa, optimize=True)
    X1A -= (1.0 / 2.0) * np.einsum("mina,efgnop,efgmop->ai", H.aa.ooov, L.aab, T.aab, optimize=True)
    X1A -= (1.0 / 4.0) * np.einsum("mina,efgnop,efgmop->ai", H.aa.ooov, L.abb, T.abb, optimize=True)

    X1A -= (1.0 / 12.0) * np.einsum("iman,gfepon,gfepom->ai", H.ab.oovo, L.bbb, T.bbb, optimize=True)
    X1A -= (1.0 / 2.0) * np.einsum("iman,gfepon,gfepom->ai", H.ab.oovo, L.abb, T.abb, optimize=True)
    X1A -= (1.0 / 4.0) * np.einsum("iman,gfepon,gfepom->ai", H.ab.oovo, L.aab, T.aab, optimize=True)

    x1a = np.einsum("ie,ea->ai", H.a.ov, X.a.vv, optimize=True)
    x1a -= np.einsum("ma,im->ai", H.a.ov, X.a.oo, optimize=True)

    x1a += 0.5 * np.einsum("nmoa,iomn->ai", X.aa.ooov, H.aa.oooo, optimize=True)
    x1a += np.einsum("mnao,iomn->ai", X.ab.oovo, H.ab.oooo, optimize=True)
    x1a += np.einsum("fmae,eimf->ai", X.aa.vovv, H.aa.voov, optimize=True)
    x1a += np.einsum("fmae,iefm->ai", X.ab.vovv, H.ab.ovvo, optimize=True)
    x1a += np.einsum("mfae,iemf->ai", X.ab.ovvv, H.ab.ovov, optimize=True)

    x1a -= 0.5 * np.einsum("gife,efag->ai", X.aa.vovv, H.aa.vvvv, optimize=True)
    x1a -= np.einsum("igef,efag->ai", X.ab.ovvv, H.ab.vvvv, optimize=True)
    x1a -= np.einsum("imne,enma->ai", X.aa.ooov, H.aa.voov, optimize=True)
    x1a -= np.einsum("imne,neam->ai", X.ab.ooov, H.ab.ovvo, optimize=True)
    x1a -= np.einsum("imen,enam->ai", X.ab.oovo, H.ab.vovo, optimize=True)

    x1a += 0.5 * np.einsum("nmoa,iomn->ai", H.aa.ooov, X.aa.oooo, optimize=True)
    x1a += np.einsum("mnao,iomn->ai", H.ab.oovo, X.ab.oooo, optimize=True)
    x1a += np.einsum("fmae,eimf->ai", H.aa.vovv, X.aa.voov, optimize=True)
    x1a += np.einsum("fmae,iefm->ai", H.ab.vovv, X.ab.ovvo, optimize=True)
    x1a += np.einsum("mfae,iemf->ai", H.ab.ovvv, X.ab.ovov, optimize=True)

    x1a -= 0.5 * np.einsum("gife,efag->ai", H.aa.vovv, X.aa.vvvv, optimize=True)
    x1a -= np.einsum("igef,efag->ai", H.ab.ovvv, X.ab.vvvv, optimize=True)
    x1a -= np.einsum("imne,enma->ai", H.aa.ooov, X.aa.voov, optimize=True)
    x1a -= np.einsum("imne,neam->ai", H.ab.ooov, X.ab.ovvo, optimize=True)
    x1a -= np.einsum("imen,enam->ai", H.ab.oovo, X.ab.vovo, optimize=True)

    I1A_vo = (
          -0.5 * np.einsum("nomg,egno->em", X.aa.ooov, T.aa, optimize=True)
          - np.einsum("nomg,egno->em", X.ab.ooov, T.ab, optimize=True)
    )
    I1B_vo = (
          -0.5 * np.einsum("nomg,egno->em", X.bb.ooov, T.bb, optimize=True)
          - np.einsum("ongm,geon->em", X.ab.oovo, T.ab, optimize=True)
    )
    x1a += np.einsum("em,imae->ai", I1A_vo, H.aa.oovv, optimize=True)
    x1a += np.einsum("em,imae->ai", I1B_vo, H.ab.oovv, optimize=True)

    x1a -= np.einsum("nm,mina->ai", X.a.oo, H.aa.ooov, optimize=True)
    x1a -= np.einsum("nm,iman->ai", X.b.oo, H.ab.oovo, optimize=True)
    x1a -= np.einsum("ef,fiea->ai", X.a.vv, H.aa.vovv, optimize=True)
    x1a -= np.einsum("ef,ifae->ai", X.b.vv, H.ab.ovvv, optimize=True)

    print("Error on L1A update = ", np.linalg.norm(X1A.flatten() - x1a.flatten()))


    ################ CHECKING UPDATE L2A ###################
    # [1] 1/4 A(ij)A(ab) h3a(iefamn) * l3a(jmnbef)
    I3A_ovvvoo = (
        0.5 * np.einsum("amfe,bfji->mabeij", H.aa.vovv, T.aa, optimize=True)
        - 0.5 * np.einsum("nmje,abin->mabeij", H.aa.ooov, T.aa, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.aa.oovv, T.aaa, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_ovvvoo -= np.transpose(I3A_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3A_ovvvoo -= np.transpose(I3A_ovvvoo, (0, 1, 2, 3, 5, 4))
    X2A = (1.0 / 4.0) * np.einsum("iefamn,befjmn->abij", I3A_ovvvoo, L.aaa, optimize=True)

    # [2] A(ij)A(ab) h3b(iefamn) * l3b(jmnbef)
    I3B_ovvvoo = (
        np.einsum("amfe,fbij->mabeij", H.aa.vovv, T.ab, optimize=True)
        + np.einsum("mbef,afij->mabeij", H.ab.ovvv, T.ab, optimize=True)
        - np.einsum("nmie,abnj->mabeij", H.aa.ooov, T.ab, optimize=True)
        - np.einsum("mnej,abin->mabeij", H.ab.oovo, T.ab, optimize=True)
        + np.einsum("mnef,afbinj->mabeij", H.aa.oovv, T.aab, optimize=True)
        + np.einsum("mnef,afbinj->mabeij", H.ab.oovv, T.abb, optimize=True)
    )
    X2A += np.einsum("iefamn,befjmn->abij", I3B_ovvvoo, L.aab, optimize=True)

    # [3] 1/4 A(ij)A(ab) h3c(iefamn) * l3c(jmnbef)
    I3C_ovvvoo = (
        0.5 * np.einsum("maef,bfji->mabeij", H.ab.ovvv, T.bb, optimize=True)
        - 0.5 * np.einsum("mnei,abnj->mabeij", H.ab.oovo, T.bb, optimize=True)
        + 0.25 * np.einsum("mnef,fabnij->mabeij", H.aa.oovv, T.abb, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.ab.oovv, T.bbb, optimize=True)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    X2A += (1.0 / 4.0) * np.einsum("iefamn,befjmn->abij", I3C_ovvvoo, L.abb, optimize=True)

    # [4] 1/6 h3a(ijgmno) * l3a(mnoabg)
    I3A_oovooo = (
        np.einsum("mnif,fcjk->mncijk", H.aa.ooov, T.aa, optimize=True)
        + (1.0 / 6.0) * np.einsum("mnef,efcijk->mncijk", H.aa.oovv, T.aaa, optimize=True)
    )
    I3A_oovooo -= np.transpose(I3A_oovooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_oovooo, (0, 1, 2, 5, 4, 3))
    X2A += (1.0 / 24.0) * np.einsum("ijgmno,abgmno->abij", I3A_oovooo, L.aaa, optimize=True)

    # [5] 1/2 h3b(ijgmno) * l3b(mnoabg)
    I3B_oovooo = (
        np.einsum("mnif,fcjk->mncijk", H.aa.ooov, T.ab, optimize=True)
        + 0.25 * np.einsum("mnef,efcijk->mncijk", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    X2A += (1.0 / 8.0) * np.einsum("ijgmno,abgmno->abij", I3B_oovooo, L.aab, optimize=True)

    # [6] 1/6 h3a(efgabo) * l3a(ijoefg)
    I3A_vvvvvo = (
        -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa, optimize=True)
        + (1.0 / 6.0) * np.einsum("mnef,abcmnk->abcefk", H.aa.oovv, T.aaa, optimize=True)
    )
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    X2A += (1.0 / 24.0) * np.einsum("efgabo,efgijo->abij", I3A_vvvvvo, L.aaa, optimize=True)

    # [7] 1/2 h3b(efgabo) * l3b(ijoefg)
    I3B_vvvvvo = (
        -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.ab, optimize=True)
        + 0.25 * np.einsum("mnef,abcmnk->abcefk", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    X2A += (1.0 / 8.0) * np.einsum("efgabo,efgijo->abij", I3B_vvvvvo, L.aab, optimize=True)

    # 4-body HBar
    X2A -= (1.0 / 24.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.aa.oovv, L.aaa, T.aaa, optimize=True)
    X2A -= (1.0 / 4.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.aa.oovv, L.aab, T.aab, optimize=True)
    X2A -= (1.0 / 8.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.aa.oovv, L.abb, T.abb, optimize=True)

    X2A -= (1.0 / 24.0) * np.einsum("mjab,efgino,efgmno->abij", H.aa.oovv, L.aaa, T.aaa, optimize=True)
    X2A -= (1.0 / 4.0) * np.einsum("mjab,efgino,efgmno->abij", H.aa.oovv, L.aab, T.aab, optimize=True)
    X2A -= (1.0 / 8.0) * np.einsum("mjab,efgino,efgmno->abij", H.aa.oovv, L.abb, T.abb, optimize=True)

    X2A -= np.transpose(X2A, (1, 0, 2, 3)) + np.transpose(X2A, (0, 1, 3, 2)) - np.transpose(X2A, (1, 0, 3, 2))

    # < 0 | L3 * H(2) | ijab >
    x2a = -np.einsum("ejfb,fiea->abij", X.aa.vovv, H.aa.vovv, optimize=True) # 1
    x2a -= np.einsum("njmb,mina->abij", X.aa.ooov, H.aa.ooov, optimize=True) # 2
    x2a -= 0.25 * np.einsum("enab,jine->abij", X.aa.vovv, H.aa.ooov, optimize=True) # 3
    x2a -= 0.25 * np.einsum("jine,enab->abij", X.aa.ooov, H.aa.vovv, optimize=True) # 4
    x2a -= np.einsum("jebf,ifae->abij", X.ab.ovvv, H.ab.ovvv, optimize=True) # 5
    x2a -= np.einsum("jnbm,iman->abij", X.ab.oovo, H.ab.oovo, optimize=True) # 6

    # < 0 | L3 * (H(2) * T3) | ijab >
    x2a += np.einsum("ejmb,imae->abij", X.aa.voov, H.aa.oovv, optimize=True) # 1
    x2a += np.einsum("jebm,imae->abij", X.ab.ovvo, H.ab.oovv, optimize=True) # 2
    x2a += 0.125 * np.einsum("efab,ijef->abij", X.aa.vvvv, H.aa.oovv, optimize=True) # 3
    x2a += 0.125 * np.einsum("ijmn,mnab->abij", X.aa.oooo, H.aa.oovv, optimize=True) # 4

    # 4-body HBar
    x2a += 0.5 * np.einsum("ea,ijeb->abij", X.a.vv, H.aa.oovv, optimize=True) # 1
    x2a -= 0.5 * np.einsum("im,jmba->abij", X.a.oo, H.aa.oovv, optimize=True) # 2

    x2a -= np.transpose(x2a, (1, 0, 2, 3)) + np.transpose(x2a, (0, 1, 3, 2)) - np.transpose(x2a, (1, 0, 3, 2))

    print("Error on L2A update = ", np.linalg.norm(X2A.flatten() - x2a.flatten()))

    ################ CHECKING UPDATE L2B ###################
    # [1] 1/4 h3a(iefamn) * l3b(mnjefb)
    I3A_ovvvoo = (
        0.5 * np.einsum("amfe,bfji->mabeij", H.aa.vovv, T.aa, optimize=True)
        - 0.5 * np.einsum("nmje,abin->mabeij", H.aa.ooov, T.aa, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.aa.oovv, T.aaa, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_ovvvoo -= np.transpose(I3A_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3A_ovvvoo -= np.transpose(I3A_ovvvoo, (0, 1, 2, 3, 5, 4))
    X2B = 0.25 * np.einsum("iefamn,efbmnj->abij", I3A_ovvvoo, L.aab, optimize=True)

    # [2] h3b(iefamn) * l3c(mjnebf)
    I3B_ovvvoo = (
        np.einsum("amfe,fbij->mabeij", H.aa.vovv, T.ab, optimize=True)
        + np.einsum("mbef,afij->mabeij", H.ab.ovvv, T.ab, optimize=True)
        - np.einsum("nmie,abnj->mabeij", H.aa.ooov, T.ab, optimize=True)
        - np.einsum("mnej,abin->mabeij", H.ab.oovo, T.ab, optimize=True)
        + np.einsum("mnef,afbinj->mabeij", H.aa.oovv, T.aab, optimize=True)
        + np.einsum("mnef,afbinj->mabeij", H.ab.oovv, T.abb, optimize=True)
    )
    X2B += np.einsum("iefamn,ebfmjn->abij", I3B_ovvvoo, L.abb, optimize=True)

    # [3] 1/4 h3c(iefamn) * l3d(mnjefb)
    I3C_ovvvoo = (
        0.5 * np.einsum("maef,bfji->mabeij", H.ab.ovvv, T.bb, optimize=True)
        - 0.5 * np.einsum("mnei,abnj->mabeij", H.ab.oovo, T.bb, optimize=True)
        + 0.25 * np.einsum("mnef,fabnij->mabeij", H.aa.oovv, T.abb, optimize=True)
        + 0.25 * np.einsum("mnef,abfijn->mabeij", H.ab.oovv, T.bbb, optimize=True)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    X2B += (1.0 / 4.0) * np.einsum("iefamn,efbmnj->abij", I3C_ovvvoo, L.bbb, optimize=True)

    # [4] 1/4 h3b(efjmnb) * l3a(imnaef)
    I3B_vvooov = (
        -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa)
        +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa)
        +0.25 * np.einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa)
        +0.25 * np.einsum("mnef,abfijn->abmije", H.bb.oovv, T.aab)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    X2B += (1.0 / 4.0) * np.einsum("efjmnb,aefimn->abij", I3B_vvooov, L.aaa)

    # [5] h3c(efjmnb) * l3b(imnaef)
    I3C_vvooov = (
        -np.einsum("nmie,abnj->abmije", H.ab.ooov, T.ab)
        -np.einsum("nmje,abin->abmije", H.bb.ooov, T.ab)
        +np.einsum("amfe,fbij->abmije", H.ab.vovv, T.ab)
        +np.einsum("bmfe,afij->abmije", H.bb.vovv, T.ab)
        +np.einsum("nmfe,afbinj->abmije", H.ab.oovv, T.aab)
        +np.einsum("mnef,abfijn->abmije", H.bb.oovv, T.abb)
    )
    X2B += np.einsum("efjmnb,aefimn->abij", I3C_vvooov, L.aab)

    # [6] 1/4 h3d(jefbmn) * l3c(imnaef)
    I3D_ovvvoo = (
        -0.5 * np.einsum("nmje,abin->mabeij", H.bb.ooov, T.bb)
        +0.5 * np.einsum("bmfe,afij->mabeij", H.bb.vovv, T.bb)
        +0.25 * np.einsum("nmfe,fabnij->mabeij", H.ab.oovv, T.abb)
        +0.25 * np.einsum("mnef,afbinj->mabeij", H.bb.oovv, T.bbb)
    )
    I3D_ovvvoo -= np.transpose(I3D_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3D_ovvvoo -= np.transpose(I3D_ovvvoo, (0, 1, 2, 3, 5, 4))
    X2B += (1.0 / 4.0) * np.einsum("jefbmn,aefimn->abij", I3D_ovvvoo, L.abb)

    # [7] -1/2 h3b(efjanm) * l3b(inmefb)
    I3B_vvovoo = (
        -0.5 * np.einsum("nmej,acnk->acmekj", H.ab.oovo, T.aa)
        +np.einsum("amef,cfkj->acmekj", H.ab.vovv, T.ab)
        -0.5 * np.einsum("nmef,acfnkj->acmekj", H.ab.oovv, T.aab)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    X2B -= (1.0 / 2.0) * np.einsum("efjanm,efbinm->abij", I3B_vvovoo, L.aab)

    # [8] -1/2 h3c(efjanm) * l3c(imnebf)
    I3C_vvovoo = (
        -np.einsum("nmej,acnk->acmekj", H.ab.oovo, T.ab)
        +0.5 * np.einsum("amef,fcjk->acmekj", H.ab.vovv, T.bb)
        -0.5 * np.einsum("nmef,afcnjk->acmekj", H.ab.oovv, T.abb)
    )
    I3C_vvovoo -= np.transpose(I3C_vvovoo, (0, 1, 2, 3, 5, 4))
    X2B -= (1.0 / 2.0) * np.einsum("efjanm,ebfimn->abij", I3C_vvovoo, L.abb)

    # [9] -1/2 h3b(ifemnb) * l3b(mnjafe)
    I3B_ovvoov = (
        0.5 * np.einsum("mbfe,fcik->mcbike", H.ab.ovvv, T.aa)
        -np.einsum("mnie,cbkn->mcbike", H.ab.ooov, T.ab)
        -0.5 * np.einsum("mnfe,cfbkin->mcbike", H.ab.oovv, T.aab)
    )
    I3B_ovvoov -= np.transpose(I3B_ovvoov, (0, 1, 2, 4, 3, 5))
    X2B -= (1.0 / 2.0) * np.einsum("ifemnb,afemnj->abij", I3B_ovvoov, L.aab)

    # [10] -1/2 h3c(iefmbn) * l3c(mnjafe)
    I3C_ovvovo = (
        np.einsum("mbfe,fcik->mbciek", H.ab.ovvv, T.ab)
        -0.5 * np.einsum("mnie,bcnk->mbciek", H.ab.ooov, T.bb)
        -0.5 * np.einsum("mnfe,fbcink->mbciek", H.ab.oovv, T.abb)
    )
    I3C_ovvovo -= np.transpose(I3C_ovvovo, (0, 2, 1, 3, 4, 5))
    X2B -= (1.0 / 2.0) * np.einsum("iefmbn,afemnj->abij", I3C_ovvovo, L.abb)

    # [11] 1/2 h3b(igjmon) * l3b(monagb)
    I3B_ovoooo = (
        np.einsum("mnif,cfkj->mcnikj", H.ab.ooov, T.ab)
        +0.5 * np.einsum("mnfj,fcik->mcnikj", H.ab.oovo, T.aa)
        +0.5 * np.einsum("mnef,ecfikj->mcnikj", H.ab.oovv, T.aab)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    X2B += (1.0 / 2.0) * np.einsum("igjmon,agbmon->abij", I3B_ovoooo, L.aab)

    # [12] 1/2 h3c(igjmon) * l3c(monagb)
    I3C_ovoooo = (
        0.5 * np.einsum("mnif,fcjk->mcnikj", H.ab.ooov, T.bb)
        +np.einsum("mnfj,fcik->mcnikj", H.ab.oovo, T.ab)
        +0.5 * np.einsum("mnef,efcijk->mcnikj", H.ab.oovv, T.abb)
    )
    I3C_ovoooo -= np.transpose(I3C_ovoooo, (0, 1, 2, 3, 5, 4))
    X2B += (1.0 / 2.0) * np.einsum("igjmon,agbmon->abij", I3C_ovoooo, L.abb)

    # [13] 1/2 h3b(egfaob) * l3b(iojegf)
    I3B_vvvvov = (
        -np.einsum("anef,cbkn->acbekf", H.ab.vovv, T.ab)
        -0.5 * np.einsum("nbef,acnk->acbekf", H.ab.ovvv, T.aa)
        +0.5 * np.einsum("mnef,acbmkn->acbekf", H.ab.oovv, T.aab)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    X2B += (1.0 / 2.0) * np.einsum("egfaob,egfioj->abij", I3B_vvvvov, L.aab)

    # [14] 1/2 h3c(efgabo) * l3c(iojegf)
    I3C_vvvvvo = (
        -0.5 * np.einsum("anef,bcnk->abcefk", H.ab.vovv, T.bb)
        -np.einsum("nbef,acnk->abcefk", H.ab.ovvv, T.ab)
        +0.5 * np.einsum("mnef,abcmnk->abcefk", H.ab.oovv, T.abb)
    )
    I3C_vvvvvo -= np.transpose(I3C_vvvvvo, (0, 2, 1, 3, 4, 5))
    X2B += (1.0 / 2.0) * np.einsum("efgabo,egfioj->abij", I3C_vvvvvo, L.abb)

    # 4-body HBar
    X2B -= (1.0 / 12.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.ab.oovv, L.aaa, T.aaa, optimize=True)
    X2B -= (1.0 / 2.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.ab.oovv, L.aab, T.aab, optimize=True)
    X2B -= (1.0 / 4.0) * np.einsum("ijeb,afgmno,efgmno->abij", H.ab.oovv, L.abb, T.abb, optimize=True)

    X2B -= (1.0 / 12.0) * np.einsum("mjab,efgino,efgmno->abij", H.ab.oovv, L.aaa, T.aaa, optimize=True)
    X2B -= (1.0 / 2.0) * np.einsum("mjab,efgino,efgmno->abij", H.ab.oovv, L.aab, T.aab, optimize=True)
    X2B -= (1.0 / 4.0) * np.einsum("mjab,efgino,efgmno->abij", H.ab.oovv, L.abb, T.abb, optimize=True)

    X2B -= (1.0 / 12.0) * np.einsum("ijae,fgbnom,fgenom->abij", H.ab.oovv, L.bbb, T.bbb, optimize=True)
    X2B -= (1.0 / 2.0) * np.einsum("ijae,fgbnom,fgenom->abij", H.ab.oovv, L.abb, T.abb, optimize=True)
    X2B -= (1.0 / 4.0) * np.einsum("ijae,fgbnom,fgenom->abij", H.ab.oovv, L.aab, T.aab, optimize=True)

    X2B -= (1.0 / 12.0) * np.einsum("imab,fgenoj,fgenom->abij", H.ab.oovv, L.bbb, T.bbb, optimize=True)
    X2B -= (1.0 / 2.0) * np.einsum("imab,fgenoj,fgenom->abij", H.ab.oovv, L.abb, T.abb, optimize=True)
    X2B -= (1.0 / 4.0) * np.einsum("imab,fgenoj,fgenom->abij", H.ab.oovv, L.aab, T.aab, optimize=True)

    # < 0 | L3 * H(2) | ij~ab~ >
    x2b = -np.einsum("ejfb,fiea->abij", X.ab.vovv, H.aa.vovv, optimize=True)  # 1
    x2b -= np.einsum("ejfb,ifae->abij", X.bb.vovv, H.ab.ovvv, optimize=True)  # 2
    x2b -= np.einsum("eifa,fjeb->abij", X.aa.vovv, H.ab.vovv, optimize=True)  # 3
    x2b -= np.einsum("ieaf,fjeb->abij", X.ab.ovvv, H.bb.vovv, optimize=True)  # 4
    x2b -= np.einsum("njmb,mina->abij", X.ab.ooov, H.aa.ooov, optimize=True)  # 5
    x2b -= np.einsum("njmb,iman->abij", X.bb.ooov, H.ab.oovo, optimize=True)  # 6
    x2b -= np.einsum("nima,mjnb->abij", X.aa.ooov, H.ab.ooov, optimize=True)  # 7
    x2b -= np.einsum("inam,mjnb->abij", X.ab.oovo, H.bb.ooov, optimize=True)  # 8
    x2b += np.einsum("inmb,mjan->abij", X.ab.ooov, H.ab.oovo, optimize=True)  # 9
    x2b += np.einsum("ifeb,ejaf->abij", X.ab.ovvv, H.ab.vovv, optimize=True)  # 10
    x2b += np.einsum("ejaf,ifeb->abij", X.ab.vovv, H.ab.ovvv, optimize=True)  # 11
    x2b += np.einsum("mjan,inmb->abij", X.ab.oovo, H.ab.ooov, optimize=True)  # 12
    x2b -= np.einsum("enab,ijen->abij", X.ab.vovv, H.ab.oovo, optimize=True)  # 13
    x2b -= np.einsum("mfab,ijmf->abij", X.ab.ovvv, H.ab.ooov, optimize=True)  # 14
    x2b -= np.einsum("ijmf,mfab->abij", X.ab.ooov, H.ab.ovvv, optimize=True)  # 15
    x2b -= np.einsum("ijen,enab->abij", X.ab.oovo, H.ab.vovv, optimize=True)  # 16

    # < 0 | L3 * (H(2) * T3)_C | ij~ab~ >
    x2b += (
            np.einsum("ejmb,miea->abij", X.ab.voov, H.aa.oovv, optimize=True)
            + np.einsum("ejmb,imae->abij", X.bb.voov, H.ab.oovv, optimize=True)
    )  # 1
    x2b += (
            np.einsum("eima,mjeb->abij", X.aa.voov, H.ab.oovv, optimize=True)
            + np.einsum("ieam,mjeb->abij", X.ab.ovvo, H.bb.oovv, optimize=True)
    )  # 2
    x2b -= np.einsum("iemb,mjae->abij", X.ab.ovov, H.ab.oovv, optimize=True)  # 3
    x2b -= np.einsum("ejam,imeb->abij", X.ab.vovo, H.ab.oovv, optimize=True)  # 4
    x2b += np.einsum("efab,ijef->abij", X.ab.vvvv, H.ab.oovv, optimize=True)  # 5
    x2b += np.einsum("ijmn,mnab->abij", X.ab.oooo, H.ab.oovv, optimize=True)  # 6

    # 4-body HBar
    x2b += np.einsum("ea,ijeb->abij", X.a.vv, H.ab.oovv, optimize=True) # 1
    x2b += np.einsum("eb,ijae->abij", X.b.vv, H.ab.oovv, optimize=True) # 2
    x2b -= np.einsum("im,mjab->abij", X.a.oo, H.ab.oovv, optimize=True) # 3
    x2b -= np.einsum("jm,imab->abij", X.b.oo, H.ab.oovv, optimize=True) # 4

    print("Error on L2B update = ", np.linalg.norm(X2B.flatten() - x2b.flatten()))