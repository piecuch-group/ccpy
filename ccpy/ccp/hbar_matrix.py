import numpy as np

# Thoughts on how to do this "efficiently":
#
#   1. We cannot have a double loop over triples in the P space; even CIPSI people do not do this.
#      The most efficient way to operate is to have a single loop over the P space and then, for every
#      determinant in the P space, try to efficiently generate all other connected determinants by,
#      for example, applying all allowed single, double, and triple excitations that keep you in the P space.
#
#   2. In lieu of this complicated scheme, we use one expensive operation to precompute the connected determinants
#      for each diagrammatic term. This incurs a large storage cost. How to get around this?
#
#      E.g.,:
#      X3A(abcijk) = 1/2 * A(c/ab) h2a(abef) * t3a(efcijk)
#
#      -> First, define a hashing function that maps a tuple (a, b, c, i, j, k) into the index of t3 in the P space
#     def hash(a, b, c, i, j, k):
#           return idx
#
#     for a, b, c, i, j, k in triples_P_space:
#         idet = hash(a, b, c, i, j, k)
#          for e, f in connected_dets_dgm4[idet]:
#              idx1 = hash(e, f, c, i, j, k)
#              idx2 = hash(e, f, a, i, j, k)
#              idx3 = hash(e, f, b, i, j, k)
#              X3A[idet] += 0.5 * h2a[a, b, e, f] * t3a[idx1]
#              X3A[idet] -= 0.5 * h2a[c, b, e, f] * t3a[idx2]
#              X3A[idet] -= 0.5 * h2a[a, c, e, f] * t3a[idx3]

def update_hbar_t3a(t3_aaa, list_aaa, idx3A, idx3B, H, H0, shift, system):

    n3a_p = len(idx3A)

    Hmat = np.zeros((n3a_p, n3a_p))

    # loop over aaa
    for idet in range(n3a_p):
        a, b, c, i, j, k = list_aaa[idet]

        # -A(i/jk) h1a(mi) * t3a(abcmjk)
        #  A(k/ij) 1/2 h2a(mnij) * t3a(abcmnk)
        for m in range(system.noccupied_alpha):
            jdet = idx3A[a, b, c, m, j, k]
            if jdet != -1:
                Hmat[idet, jdet] -= H.a.oo[m, i]
            jdet = idx3A[a, b, c, m, i, k]
            if jdet != -1:
                Hmat[idet, jdet] += H.a.oo[m, j]
            jdet = idx3A[a, b, c, m, j, i]
            if jdet != -1:
                Hmat[idet, jdet] -= H.a.oo[m, k]

            for n in range(m + 1, system.noccupied_alpha):
                jdet = idx3A[a, b, c, m, n, k]
                if jdet != -1:
                    Hmat[idet, jdet] += H.aa.oooo[m, n, i, j]
                jdet = idx3A[a, b, c, m, n, i]
                if jdet != -1:
                    Hmat[idet, jdet] -= H.aa.oooo[m, n, k, j]
                jdet = idx3A[a, b, c, m, n, j]
                if jdet != -1:
                    Hmat[idet, jdet] -= H.aa.oooo[m, n, i, k]


        # A(a/bc) h1a(ae) * t3a(ebcijk)
        # A(c/ab) 1/2 h2a(abef) * t3a(efcijk)
        # A(i/jk)A(a/bc) h2a(amie) * t3a(ebcmjk)
        for e in range(system.nunoccupied_alpha):

            jdet = idx3A[e, b, c, i, j, k]
            if jdet != -1:
                Hmat[idet, jdet] += H.a.vv[a, e]
            jdet = idx3A[e, a, c, i, j, k]
            if jdet != -1:
                Hmat[idet, jdet] -= H.a.vv[b, e]
            jdet = idx3A[e, b, a, i, j, k]
            if jdet != -1:
                Hmat[idet, jdet] -= H.a.vv[c, e]

            for f in range(e + 1, system.nunoccupied_alpha):
                jdet = idx3A[e, f, c, i, j, k]
                if jdet != -1:
                    Hmat[idet, jdet] += H.aa.vvvv[a, b, e, f]
                jdet = idx3A[e, f, a, i, j, k]
                if jdet != -1:
                    Hmat[idet, jdet] -= H.aa.vvvv[c, b, e, f]
                jdet = idx3A[e, f, b, i, j, k]
                if jdet != -1:
                    Hmat[idet, jdet] -= H.aa.vvvv[a, c, e, f]

            for m in range(system.noccupied_alpha):
                jdet = idx3A[e, b, c, m, j, k]
                if jdet != -1:
                    Hmat[idet, jdet] += H.aa.voov[a, m, i, e]
                jdet = idx3A[e, a, c, m, j, k]
                if jdet != -1:
                    Hmat[idet, jdet] -= H.aa.voov[b, m, i, e]


    # end loop

    # compute residual using one-time matrix-vector product
    residual = np.dot(Hmat, t3_aaa)

    # t3_aaa, residual = ccp_loops.ccp_loops.update_t3a(
    #     t3_aaa,
    #     residual,
    #     H0.a.oo,
    #     H0.a.vv,
    #     H0.b.oo,
    #     H0.b.vv,
    #     shift,
    # )
    # return t3_aaa, residual