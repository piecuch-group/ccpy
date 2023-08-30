
def HR(R, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    r1a, r1b, r2a, r2b, r2c, r2d = unflatten_R(R, sys)

    X1A = build_HR_1A(
        r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
    )
    X2A = build_HR_2A(
        r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
    )
    X2B = build_HR_2B(
        r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
    )
    X2C = build_HR_2C(
        r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
    )
    X2D = build_HR_2D(
        r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
    )

    return flatten_R(X1A, X1B, X2A, X2B, X2C, X2D)


def build_HR_1A(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    X1A = 0.0
    X1A += np.einsum("ae,e->a", H1A["vv"], r1a, optimize=True)
    X1A += 0.5 * np.einsum("anef,efn->a", H2A["vovv"], r2a, optimize=True)
    X1A += np.einsum("anef,efn->a", H2B["vovv"], r2c, optimize=True)
    X1A += np.einsum("me,aem->a", H1A["ov"], r2a, optimize=True)
    X1A += np.einsum("me,aem->a", H1B["ov"], r2c, optimize=True)

    return X1A


def build_HR_1B(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    X1B = 0.0
    X1B += np.einsum("ae,e->a", H1B["vv"], r1b, optimize=True)
    X1B += np.einsum("nafe,efn->a", H2B["ovvv"], r2b, optimize=True)
    X1B += 0.5 * np.einsum("anef,efn->a", H2C["vovv"], r2d, optimize=True)
    X1B += np.einsum("me,aem->a", H1A["ov"], r2b, optimize=True)
    X1B += np.einsum("me,aem->a", H1B["ov"], r2d, optimize=True)

    return X1B


def build_HR_2A(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    vA = ints["vA"]
    vB = ints["vB"]
    t2a = cc_t["t2a"]

    X2A = 0.0
    X2A += np.einsum("baje,e->abj", H2A["vvov"], r1a, optimize=True)
    X2A += np.einsum("mj,abm->abj", H1A["oo"], r2a, optimize=True)
    X2A += 0.5 * np.einsum("abef,efj->abj", H2A["vvvv"], r2a, optimize=True)
    I1 = 0.5 * np.einsum("mnef,efn->m", vA["oovv"], r2a, optimize=True) + np.einsum(
        "mnef,efn->m", vB["oovv"], r2c, optimize=True
    )
    X2A == np.einsum("m,abmj->abj", I1, t2a, optimize=True)

    D_ab = 0.0
    D_ab -= np.einsum("ae,ebj->abj", H1A["vv"], r2a, optimize=True)
    D_ab += np.einsum("bmje,aem->abj", H2A["voov"], r2a, optimize=True)
    D_ab += np.einsum("bmje,aem->abj", H2B["voov"], r2c, optimize=True)
    D_ab -= np.transpose(D_ab, (1, 0, 2))

    X2A += D_ab

    return X2A


def build_HR_2B(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    t2b = cc_t["t2b"]
    vB = ints["vB"]
    vC = ints["vC"]

    X2B = 0.0
    X2B -= np.einsum("bmji,m->bij", H2B["vooo"], r1b, optimize=True)
    X2B -= np.einsum("mi,bmj->bij", H1B["oo"], r2b, optimize=True)
    X2B -= np.einsum("mj,bim->bij", H1A["oo"], r2b, optimize=True)
    X2B += np.einsum("be,eij->bij", H1A["vv"], r2b, optimize=True)
    X2B += np.einsum("nmji,bmn->bij", H2B["oooo"], r2b, optimize=True)
    X2B += np.einsum("bmje,eim->bij", H2A["voov"], r2b, optimize=True)
    X2B += np.einsum("bmje,eim->bij", H2B["voov"], r2d, optimize=True)
    X2B -= np.einsum("bmei,emj->bij", H2B["vovo"], r2b, optimize=True)
    I1 = -np.einsum("nmfe,fmn->e", vB["oovv"], r2b, optimize=True) - 0.5 * np.einsum(
        "mnef,fmn->e", vC["oovv"], r2d, optimize=True
    )
    X2B += np.einsum("e,beji->bij", I1, t2b, optimize=True)

    return X2B


def build_HR_2C(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    t2b = cc_t["t2b"]
    vA = ints["vA"]
    vB = ints["vB"]

    X2C = 0.0
    X2C -= np.einsum("mbij,m->bij", H2B["ovoo"], r1a, optimize=True)
    X2C -= np.einsum("mi,bmj->bij", H1A["oo"], r2c, optimize=True)
    X2C -= np.einsum("mj,bim->bij", H1B["oo"], r2c, optimize=True)
    X2C += np.einsum("be,eij->bij", H1B["vv"], r2c, optimize=True)
    X2C += np.einsum("mnij,bmn->bij", H2B["oooo"], r2c, optimize=True)
    X2C += np.einsum("mbej,eim->bij", H2B["ovvo"], r2a, optimize=True)
    X2C += np.einsum("bmje,eim->bij", H2C["voov"], r2c, optimize=True)
    X2C -= np.einsum("mbie,emj->bij", H2B["ovov"], r2c, optimize=True)
    I1 = -0.5 * np.einsum("mnef,fmn->e", vA["oovv"], r2a, optimize=True) - np.einsum(
        "mnef,fmn->e", vB["oovv"], r2c, optimize=True
    )
    X2C += np.einsum("e,ebij->bij", I1, t2b, optimize=True)

    return X2C


def build_HR_2D(r1a, r1b, r2a, r2b, r2c, r2d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    vB = ints["vB"]
    vC = ints["vC"]
    t2c = cc_t["t2c"]

    X2D = 0.0
    X2D -= np.einsum("bmji,m->bij", H2C["vooo"], r1b, optimize=True)
    X2D += np.einsum("be,eij->bij", H1B["vv"], r2d, optimize=True)
    X2D += 0.5 * np.einsum("mnij,bmn->bij", H2C["oooo"], r2d, optimize=True)
    I1 = -0.5 * np.einsum("mnef,fmn->e", vC["oovv"], r2d, optimize=True) - np.einsum(
        "nmfe,fmn->e", vB["oovv"], r2b, optimize=True
    )
    X2D += np.einsum("e,ebij->bij", I1, t2c, optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum("mi,bmj->bij", H1B["oo"], r2d, optimize=True)
    D_ij += np.einsum("bmje,eim->bij", H2C["voov"], r2d, optimize=True)
    D_ij += np.einsum("mbej,eim->bij", H2B["ovvo"], r2b, optimize=True)
    D_ij -= np.transpose(D_ij, (0, 2, 1))

    X2D += D_ij

    return X2D


def guess_1p(ints, sys):

    fA = ints["fA"]
    fB = ints["fB"]

    n1a = sys["Nunocc_a"]
    n1b = sys["Nunocc_b"]

    HAA = np.zeros((n1a, n1a))
    HAB = np.zeros((n1a, n1b))
    HBA = np.zeros((n1b, n1a))
    HBB = np.zeros((n1b, n1b))

    ct1 = 0
    for a in range(sys["Nunocc_a"]):
        ct2 = 0
        for b in range(sys["Nunocc_a"]):
            HAA[ct1, ct2] = fA["vv"][a, b]
            ct2 += 1
        ct1 += 1

    ct1 = 0
    for a in range(sys["Nunocc_b"]):
        ct2 = 0
        for b in range(sys["Nunocc_b"]):
            HBB[ct1, ct2] = fB["vv"][a, b]
            ct2 += 1
        ct1 += 1

    H = np.hstack((np.vstack((HAA, HBA)), np.vstack((HAB, HBB))))

    E_1p, C = np.linalg.eigh(H)
    idx = np.argsort(E_1p)
    E_1p = E_1p[idx]
    C = C[:, idx]

    return C, E_1p
