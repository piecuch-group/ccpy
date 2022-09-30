import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

def get_index_arrays(system):

    ct = 0

    slices = {}

    idx1A = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct += 1
            idx1A[a, i] = ct
    slices['a'] = slice(0, ct)

    idx1B = np.zeros((system.nunoccupied_beta, system.noccupied_beta), dtype=np.int8)
    ct1 = ct
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct += 1
            idx1B[a, i] = ct
    slices['b'] = slice(ct1, ct)

    idx2A = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    ct1 = ct
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_alpha):
                    if i == j or a == b: continue
                    ct += 1
                    idx2A[a, b, i, j] = ct
    slices['aa'] = slice(ct1, ct)

    idx2B = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta), dtype=np.int8)
    ct1 = ct
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    ct += 1
                    idx2B[a, b, i, j] = ct
    slices['ab'] = slice(ct1, ct)

    idx2C = np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta), dtype=np.int8)
    ct1 = ct
    for a in range(system.nunoccupied_beta):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(system.noccupied_beta):
                    if i == j or a == b: continue
                    ct += 1
                    idx2C[a, b, i, j] = ct
    slices['bb'] = slice(ct1, ct)

    return idx1A, idx1B, idx2A, idx2B, idx2C, ct, slices

if __name__ == "__main__":


    system, H0 = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H0)

    H = build_hbar_ccsd(T, H0)

    idx1A, idx1B, idx2A, idx2B, idx2C, ndim, slices = get_index_arrays(system)


    Hmat = np.zeros((ndim, ndim))

    # 1A block
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):

            if idx1A[a, i] == 0: continue
            idet = idx1A[a, i] - 1

            #X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
            for m in range(system.noccupied_alpha):
                if idx1A[a, m] == 0: continue
                jdet = idx1A[a, m] - 1
                Hmat[idet, jdet] -= H.a.oo[m, i]
            #X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
            for e in range(system.nunoccupied_alpha):
                if idx1A[e, i] == 0: continue
                jdet = idx1A[e, i] - 1
                Hmat[idet, jdet] += H.a.vv[a, e]
            #X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
            for e in range(system.nunoccupied_alpha):
                for m in range(system.noccupied_alpha):
                    if idx1A[e, m] == 0: continue
                    jdet = idx1A[e, m] - 1
                    Hmat[idet, jdet] += H.aa.voov[a, m, i, e]
            #X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
            for e in range(system.nunoccupied_beta):
                for m in range(system.noccupied_beta):
                    if idx1B[e, m] == 0: continue
                    jdet = idx1B[e, m] - 1
                    Hmat[idet, jdet] += H.ab.voov[a, m, i, e]
            #X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
            for m in range(system.noccupied_alpha):
                for n in range(m + 1, system.noccupied_alpha):
                    for f in range(system.nunoccupied_alpha):
                        if idx2A[a, f, m, n] == 0: continue
                        jdet = idx2A[a, f, m, n] - 1
                        Hmat[idet, jdet] -= H.aa.ooov[m, n, i, f]
            #X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
            for m in range(system.noccupied_alpha):
                for n in range(system.noccupied_beta):
                    for f in range(system.nunoccupied_beta):
                        if idx2B[a, f, m, n] == 0: continue
                        jdet = idx2B[a, f, m, n] - 1
                        Hmat[idet, jdet] -= H.ab.ooov[m, n, i, f]
            # #X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
            # for n in range(system.noccupied_alpha):
            #     for e in range(system.nunoccupied_alpha):
            #         for f in range(e + 1, system.nunoccupied_alpha):
            #             if idx2A[e, f, i, n] == 0: continue
            #             jdet = idx2A[e, f, i, n] - 1
            #             Hmat[idet, jdet] += H.aa.vovv[a, n, e, f]
            # #X1A += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
            # for n in range(system.noccupied_beta):
            #     for e in range(system.nunoccupied_alpha):
            #         for f in range(system.nunoccupied_beta):
            #             if idx2B[e, f, i, n] == 0: continue
            #             jdet = idx2B[e, f, i, n] - 1
            #             Hmat[idet, jdet] += H.ab.vovv[a, n, e, f]
            # #X1A += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
            # for m in range(system.noccupied_alpha):
            #     for e in range(system.nunoccupied_alpha):
            #         if idx2A[a, e, i, m] == 0: continue
            #         jdet = idx2A[a, e, i, m] - 1
            #         Hmat[idet, jdet] += H.a.ov[m, e]
            # #X1A += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)
            # for m in range(system.noccupied_beta):
            #     for e in range(system.nunoccupied_beta):
            #         if idx2B[a, e, i, m] == 0: continue
            #         jdet = idx2B[a, e, i, m] - 1
            #         Hmat[idet, jdet] += H.b.ov[m, e]

    # 1B block
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):

            if idx1B[a, i] == 0: continue
            idet = idx1B[a, i] - 1

            #X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
            for m in range(system.noccupied_beta):
                if idx1B[a, m] == 0: continue
                jdet = idx1B[a, m] - 1
                Hmat[idet, jdet] -= H.b.oo[m, i]
            #X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
            for e in range(system.nunoccupied_beta):
                if idx1B[e, i] == 0: continue
                jdet = idx1B[e, i] - 1
                Hmat[idet, jdet] += H.b.vv[a, e]
            #X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
            for e in range(system.nunoccupied_alpha):
                for m in range(system.noccupied_alpha):
                    if idx1A[e, m] == 0: continue
                    jdet = idx1A[e, m] - 1
                    Hmat[idet, jdet] += H.ab.ovvo[m, a, e, i]
            #X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
            for e in range(system.nunoccupied_beta):
                for m in range(system.noccupied_beta):
                    if idx1B[e, m] == 0: continue
                    jdet = idx1B[e, m] - 1
                    Hmat[idet, jdet] += H.bb.voov[a, m, i, e]
            #X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
            for n in range(system.noccupied_alpha):
                for m in range(system.noccupied_beta):
                    for f in range(system.nunoccupied_beta):
                        if idx2B[f, a, n, m] == 0: continue
                        jdet = idx2B[f, a, n, m] - 1
                        Hmat[idet, jdet] -= H.ab.oovo[n, m, f, i]
            #X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
            for m in range(system.noccupied_beta):
                for n in range(m + 1, system.noccupied_beta):
                    for f in range(system.nunoccupied_beta):
                        if idx2C[a, f, m, n] == 0: continue
                        jdet = idx2C[a, f, m, n] - 1
                        Hmat[idet, jdet] -= H.bb.ooov[m, n, i, f]
            # #X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
            # for n in range(system.noccupied_alpha):
            #     for f in range(system.nunoccupied_alpha):
            #         for e in range(system.nunoccupied_beta):
            #             if idx2B[f, e, n, i] == 0: continue
            #             jdet = idx2B[f, e, n, i] - 1
            #             Hmat[idet, jdet] += H.ab.ovvv[n, a, f, e]
            # #X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
            # for n in range(system.noccupied_beta):
            #     for e in range(system.nunoccupied_beta):
            #         for f in range(e + 1, system.nunoccupied_beta):
            #             if idx2C[e, f, i, n] == 0: continue
            #             jdet = idx2C[e, f, i, n] - 1
            #             Hmat[idet, jdet] += H.bb.vovv[a, n, e, f]
            # #X1B += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
            # for m in range(system.noccupied_alpha):
            #     for e in range(system.noccupied_alpha):
            #         if idx2B[e, a, m, i] == 0: continue
            #         jdet = idx2B[e, a, m, i] - 1
            #         Hmat[idet, jdet] += H.a.ov[m, e]
            # #X1B += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)
            # for m in range(system.noccupied_beta):
            #     for e in range(system.nunoccupied_beta):
            #         if idx2C[a, e, i, m] == 0: continue
            #         jdet = idx2C[a, e, i, m] - 1
            #         Hmat[idet, jdet] += H.b.ov[m, e]


    print("Diagonalizing matrix")
    H1aa = Hmat[slices['a'], slices['a']]
    H1ab = Hmat[slices['a'], slices['b']]
    H1ba = Hmat[slices['b'], slices['a']]
    H1bb = Hmat[slices['b'], slices['b']]
    H1 = np.vstack((np.hstack((H1aa, H1ab)), np.hstack((H1ba, H1bb))))
    E, V = np.linalg.eig(H1)
    idx = np.argsort(E)
    for i in range(len(E)):
        print("Eigval ", i + 1, " = ", E[idx[i]])

    # # 2A block
    # for a in range(system.nunoccupied_alpha):
    #     for b in range(a + 1, system.nunoccupied_alpha):
    #         for i in range(system.noccupied_alpha):
    #             for j in range(i + 1, system.noccupied_alpha):
    #
    #                 if idx2A[a, b, i, j] == 0: continue
    #                 idet = idx2A[a, b, i, j] - 1
    #
    #                 # X2A = -np.einsum("mi,abmj->abij", H.a.oo, R.aa, optimize=True)  # A(ij)
    #                 for m in range(system.noccupied_alpha):
    #                     if idx2A[a, b, m, j] == 0: continue
    #                     jdet = idx2A[a, b, m, j] - 1
    #                     Hmat[idet, jdet] -= H.a.oo[m, i]
    #                 for m in range(system.noccupied_alpha):
    #                     if idx2A[a, b, m, i] == 0: continue
    #                     jdet = idx2A[a, b, m, i] - 1
    #                     Hmat[idet, jdet] += H.a.oo[m, j]
    #                 # X2A += np.einsum("ae,ebij->abij", H.a.vv, R.aa, optimize=True)  # A(ab)
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx2A[e, b, i, j] == 0: continue
    #                     jdet = idx2A[e, b, i, j] - 1
    #                     Hmat[idet, jdet] += H.a.vv[a, e]
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx2A[e, a, i, j] == 0: continue
    #                     jdet = idx2A[e, a, i, j] - 1
    #                     Hmat[idet, jdet] -= H.a.vv[b, e]
    #                 # X2A += 0.5 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.aa, optimize=True)
    #                 for m in range(system.noccupied_alpha):
    #                     for n in range(m + 1, system.noccupied_alpha):
    #                         if idx2A[a, b, m, n] == 0: continue
    #                         jdet = idx2A[a, b, m, n] - 1
    #                         Hmat[idet, jdet] += H.aa.oooo[m, n, i, j]
    #                 # X2A += 0.5 * np.einsum("abef,efij->abij", H.aa.vvvv, R.aa, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for f in range(e + 1, system.nunoccupied_alpha):
    #                         if idx2A[e, f, i, j] == 0: continue
    #                         jdet = idx2A[e, f, i, j] - 1
    #                         Hmat[idet, jdet] += H.aa.vvvv[a, b, e, f]
    #                 # X2A += np.einsum("amie,ebmj->abij", H.aa.voov, R.aa, optimize=True)  # A(ij)A(ab)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2A[e, b, m, j] == 0: continue
    #                         jdet = idx2A[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.aa.voov[a, m, i, e]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2A[e, b, m, i] == 0: continue
    #                         jdet = idx2A[e, b, m, i] - 1
    #                         Hmat[idet, jdet] -= H.aa.voov[a, m, j, e]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2A[e, a, m, j] == 0: continue
    #                         jdet = idx2A[e, a, m, j] - 1
    #                         Hmat[idet, jdet] -= H.aa.voov[b, m, i, e]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2A[e, a, m, i] == 0: continue
    #                         jdet = idx2A[e, a, m, i] - 1
    #                         Hmat[idet, jdet] += H.aa.voov[b, m, j, e]
    #                 # X2A += np.einsum("amie,bejm->abij", H.ab.voov, R.ab, optimize=True)  # A(ij)A(ab)
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[e, b, m, j] == 0: continue
    #                         jdet = idx2B[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.ab.voov[a, m, i, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[e, b, m, i] == 0: continue
    #                         jdet = idx2B[e, b, m, i] - 1
    #                         Hmat[idet, jdet] -= H.ab.voov[a, m, j, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[e, a, m, j] == 0: continue
    #                         jdet = idx2B[e, a, m, j] - 1
    #                         Hmat[idet, jdet] -= H.ab.voov[b, m, i, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[e, a, m, i] == 0: continue
    #                         jdet = idx2B[e, a, m, i] - 1
    #                         Hmat[idet, jdet] += H.ab.voov[b, m, j, e]
    #                 # X2A -= np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    #                 for m in range(system.noccupied_alpha):
    #                     if idx1A[a, m] == 0: continue
    #                     jdet = idx1A[a, m] - 1
    #                     Hmat[idet, jdet] -= H.aa.vooo[b, m, j, i]
    #                 for m in range(system.noccupied_alpha):
    #                     if idx1A[b, m] == 0: continue
    #                     jdet = idx1A[b, m] - 1
    #                     Hmat[idet, jdet] += H.aa.vooo[a, m, j, i]
    #                 # X2A += np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx1A[e, i] == 0: continue
    #                     jdet = idx1A[e, i] - 1
    #                     Hmat[idet, jdet] += H.aa.vvov[b, a, j, e]
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx1A[e, j] == 0: continue
    #                     jdet = idx1A[e, j] - 1
    #                     Hmat[idet, jdet] -= H.aa.vvov[b, a, i, e]
    #
    # # 2B block
    # for a in range(system.nunoccupied_alpha):
    #     for b in range(system.nunoccupied_beta):
    #         for i in range(system.noccupied_alpha):
    #             for j in range(system.noccupied_beta):
    #
    #                 if idx2B[a, b, i, j] == 0: continue
    #                 idet = idx2B[a, b, i, j] - 1
    #
    #                 # X2B = np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx2B[e, b, i, j] == 0: continue
    #                     jdet = idx2B[e, b, i, j] - 1
    #                     Hmat[idet, jdet] += H.a.vv[a, e]
    #                 # X2B += np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx2B[a, e, i, j] == 0: continue
    #                     jdet = idx2B[a, e, i, j] - 1
    #                     Hmat[idet, jdet] += H.b.vv[b, e]
    #                 # X2B -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    #                 for m in range(system.noccupied_alpha):
    #                     if idx2B[a, b, m, j] == 0: continue
    #                     jdet = idx2B[a, b, m, j] - 1
    #                     Hmat[idet, jdet] -= H.a.oo[m, i]
    #                 # X2B -= np.einsum("mj,abim->abij", H.b.oo, R.ab, optimize=True)
    #                 for m in range(system.noccupied_beta):
    #                     if idx2B[a, b, i, m] == 0: continue
    #                     jdet = idx2B[a, b, i, m] - 1
    #                     Hmat[idet, jdet] -= H.b.oo[m, j]
    #                 # X2B += np.einsum("mnij,abmn->abij", H.ab.oooo, R.ab, optimize=True)
    #                 for m in range(system.noccupied_alpha):
    #                     for n in range(system.noccupied_beta):
    #                         if idx2B[a, b, m, n] == 0: continue
    #                         jdet = idx2B[a, b, m, n] - 1
    #                         Hmat[idet, jdet] += H.ab.oooo[m, n, i, j]
    #                 # X2B += np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for f in range(system.nunoccupied_beta):
    #                         if idx2B[e, f, i, j] == 0: continue
    #                         jdet = idx2B[e, f, i, j] - 1
    #                         Hmat[idet, jdet] += H.ab.vvvv[a, b, e, f]
    #                 # X2B += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[e, b, m, j] == 0: continue
    #                         jdet = idx2B[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.aa.voov[a, m, i, e]
    #                 # X2B += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2C[e, b, m, j] == 0: continue
    #                         jdet = idx2C[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.ab.voov[a, m, i, e]
    #                 # X2B += np.einsum("mbej,aeim->abij", H.ab.ovvo, R.aa, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2A[a, e, i, m] == 0: continue
    #                         jdet = idx2A[a, e, i, m] - 1
    #                         Hmat[idet, jdet] += H.ab.ovvo[m, b, e, j]
    #                 # X2B += np.einsum("bmje,aeim->abij", H.bb.voov, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[a, e, i, m] == 0: continue
    #                         jdet = idx2B[a, e, i, m] - 1
    #                         Hmat[idet, jdet] += H.bb.voov[b, m, j, e]
    #                 # X2B -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[a, e, m, j] == 0: continue
    #                         jdet = idx2B[a, e, m, j] - 1
    #                         Hmat[idet, jdet] -= H.ab.ovov[m, b, i, e]
    #                 # X2B -= np.einsum("amej,ebim->abij", H.ab.vovo, R.ab, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2B[e, b, i, m] == 0: continue
    #                         jdet = idx2B[e, b, i, m] - 1
    #                         Hmat[idet, jdet] -= H.ab.vovo[a, m, e, j]
    #                 # X2B += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    #                 for e in range(system.nunoccupied_alpha):
    #                     if idx1A[e, i] == 0: continue
    #                     jdet = idx1A[e, i] - 1
    #                     Hmat[idet, jdet] += H.ab.vvvo[a, b, e, j]
    #                 # X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx1B[e, j] == 0: continue
    #                     jdet = idx1B[e, j] - 1
    #                     Hmat[idet, jdet] += H.ab.vvov[a, b, i, e]
    #                 # X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    #                 for m in range(system.noccupied_alpha):
    #                     if idx1A[a, m] == 0: continue
    #                     jdet = idx1A[a, m] - 1
    #                     Hmat[idet, jdet] -= H.ab.ovoo[m, b, i, j]
    #                 # X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)
    #                 for m in range(system.noccupied_beta):
    #                     if idx1B[b, m] == 0: continue
    #                     jdet = idx1B[b, m] - 1
    #                     Hmat[idet, jdet] -= H.ab.vooo[a, m, i, j]
    #
    # # 2C block
    # for a in range(system.nunoccupied_beta):
    #     for b in range(a + 1, system.nunoccupied_beta):
    #         for i in range(system.noccupied_beta):
    #             for j in range(i + 1, system.noccupied_beta):
    #
    #                 if idx2C[a, b, i, j] == 0: continue
    #                 idet = idx2C[a, b, i, j] - 1
    #
    #                 # X2C = -np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)  # A(ij)
    #                 for m in range(system.noccupied_beta):
    #                     if idx2C[a, b, m, j] == 0: continue
    #                     jdet = idx2C[a, b, m, j] - 1
    #                     Hmat[idet, jdet] -= H.b.oo[m, i]
    #                 for m in range(system.noccupied_beta):
    #                     if idx2C[a, b, m, i] == 0: continue
    #                     jdet = idx2C[a, b, m, i] - 1
    #                     Hmat[idet, jdet] += H.b.oo[m, j]
    #                 # X2C += np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)  # A(ab)
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx2C[e, b, i, j] == 0: continue
    #                     jdet = idx2C[e, b, i, j] - 1
    #                     Hmat[idet, jdet] += H.b.vv[a, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx2C[e, a, i, j] == 0: continue
    #                     jdet = idx2C[e, a, i, j] - 1
    #                     Hmat[idet, jdet] -= H.b.vv[b, e]
    #                 # X2C += 0.5 * np.einsum("mnij,abmn->abij", H.bb.oooo, R.bb, optimize=True)
    #                 for m in range(system.noccupied_beta):
    #                     for n in range(m + 1, system.noccupied_beta):
    #                         if idx2C[a, b, m, n] == 0: continue
    #                         jdet = idx2C[a, b, m, n] - 1
    #                         Hmat[idet, jdet] += H.bb.oooo[m, n, i, j]
    #                 # X2C += 0.5 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    #                 for e in range(system.nunoccupied_beta):
    #                     for f in range(e + 1, system.nunoccupied_beta):
    #                         if idx2C[e, f, i, j] == 0: continue
    #                         jdet= idx2C[e, f, i, j] - 1
    #                         Hmat[idet, jdet] += H.bb.vvvv[a, b, e, f]
    #                 # X2C += np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)  # A(ij)A(ab)
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2C[e, b, m, j] == 0: continue
    #                         jdet = idx2C[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.bb.voov[a, m, i, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2C[e, b, m, i] == 0: continue
    #                         jdet = idx2C[e, b, m, i] - 1
    #                         Hmat[idet, jdet] -= H.bb.voov[a, m, j, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2C[e, a, m, j] == 0: continue
    #                         jdet = idx2C[e, a, m, j] - 1
    #                         Hmat[idet, jdet] -= H.bb.voov[b, m, i, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     for m in range(system.noccupied_beta):
    #                         if idx2C[e, a, m, i] == 0: continue
    #                         jdet = idx2C[e, a, m, i] - 1
    #                         Hmat[idet, jdet] += H.bb.voov[b, m, j, e]
    #                 # X2C += np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)  # A(ij)A(ab)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[e, b, m, j] == 0: continue
    #                         jdet = idx2B[e, b, m, j] - 1
    #                         Hmat[idet, jdet] += H.ab.ovvo[m, a, e, i]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[e, b, m, i] == 0: continue
    #                         jdet = idx2B[e, b, m, i] - 1
    #                         Hmat[idet, jdet] -= H.ab.ovvo[m, a, e, j]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[e, a, m, j] == 0: continue
    #                         jdet = idx2B[e, a, m, j] - 1
    #                         Hmat[idet, jdet] -= H.ab.ovvo[m, b, e, i]
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         if idx2B[e, a, m, i] == 0: continue
    #                         jdet = idx2B[e, a, m, i] - 1
    #                         Hmat[idet, jdet] += H.ab.ovvo[m, b, e, j]
    #                 # X2C -= np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    #                 for m in range(system.noccupied_beta):
    #                     if idx1B[a, m] == 0: continue
    #                     jdet = idx1B[a, m] - 1
    #                     Hmat[idet, jdet] -= H.bb.vooo[b, m, j, i]
    #                 for m in range(system.noccupied_beta):
    #                     if idx1B[b, m] == 0: continue
    #                     jdet = idx1B[b, m] - 1
    #                     Hmat[idet, jdet] += H.bb.vooo[a, m, j, i]
    #                 # X2C += np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx1B[e, i] == 0: continue
    #                     jdet = idx1B[e, i] - 1
    #                     Hmat[idet, jdet] += H.bb.vvov[b, a, j, e]
    #                 for e in range(system.nunoccupied_beta):
    #                     if idx1B[e, j] == 0: continue
    #                     jdet = idx1B[e, j] - 1
    #                     Hmat[idet, jdet] -= H.bb.vvov[b, a, i, e]
    #
    #
    # print(np.linalg.norm(Hmat[slices['a'], slices['a']] - Hmat[slices['b'], slices['b']]))
    # print(np.linalg.norm(Hmat[slices['aa'], slices['aa']] - Hmat[slices['bb'], slices['bb']]))
    # # print("Diagonalizing matrix")
    # # E, V = np.linalg.eig(Hmat)
    # # idx = np.argsort(E)
    # # for i in range(ndim):
    # #     print("Eigval ", i + 1, " = ", E[idx[i]])
