"""
Calculate the triples diagonal <ijkabc|H3|ijkabc>, where H3
is the 3-body component of (H_N e^(T1+T2))_C corresponding to
(V_N*T2)_C diagrams.
"""

import numpy as np

def aaa_H3_aaa_diagonal(T, H, system):
    """< ijkabc | (V_V*T2)_C | ijkabc > diagonal"""

    d3A_V = lambda a, i, b: -np.dot(H.aa.oovv[i, :, a, b].T, T.aa[a, b, i, :])
    d3A_O = lambda a, i, j: np.dot(H.aa.oovv[i, j, a, :].T, T.aa[a, :, i, j])

    D3A_V = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha, system.nunoccupied_alpha))
    D3A_O = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))

    # A diagonal
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for b in range(system.nunoccupied_alpha):
                D3A_V[a, i, b] = d3A_V(a, i, b)
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_alpha):
                D3A_O[a, i, j] = d3A_O(a, i, j)

    return D3A_V, D3A_O

def aab_H3_aab_diagonal(T, H, system):
    """< ijk~abc~ | (V_V*T2)_C | ijk~abc~ > diagonal"""

    d3B_V = lambda a, i, c: -np.dot(H.ab.oovv[i, :, a, c].T, T.ab[a, c, i, :])
    d3B_O = lambda a, i, k: np.dot(H.ab.oovv[i, k, a, :].T, T.ab[a, :, i, k])

    D3B_V = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha, system.nunoccupied_beta))
    D3B_O = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_beta))

    # B diagonal
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for c in range(system.nunoccupied_beta):
                D3B_V[a, i, c] = d3B_V(a, i, c)
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                D3B_O[a, i, k] = d3B_O(a, i, k)

    return D3B_V, D3B_O

def abb_H3_abb_diagonal(T, H, system):
    """< ij~k~ab~c~ | (V_V*T2)_C | ij~k~ab~c~ > diagonal"""

    d3C_V = lambda a, k, c: -np.dot(H.ab.oovv[:, k, a, c].T, T.ab[a, c, :, k])
    d3C_O = lambda c, i, k: np.dot(H.ab.oovv[i, k, :, c].T, T.ab[:, c, i, k])

    D3C_V = np.zeros((system.nunoccupied_alpha, system.noccupied_beta, system.nunoccupied_beta))
    D3C_O = np.zeros((system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))

    # C diagonal
    for a in range(system.nunoccupied_alpha):
        for k in range(system.noccupied_beta):
            for c in range(system.nunoccupied_beta):
                D3C_V[a, k, c] = d3C_V(a, k, c)
    for c in range(system.nunoccupied_beta):
        for i in range(system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                D3C_O[c, i, k] = d3C_O(c, i, k)

    return D3C_V, D3C_O

def bbb_H3_bbb_diagonal(T, H, system):
    """< i~j~k~a~b~c~ | (V_V*T2)_C | i~j~k~a~b~c~ > diagonal"""

    d3D_V = lambda a, i, b: -np.dot(H.bb.oovv[i, :, a, b].T, T.bb[a, b, i, :])
    d3D_O = lambda a, i, j: np.dot(H.bb.oovv[i, j, a, :].T, T.bb[a, :, i, j])

    D3D_V = np.zeros((system.nunoccupied_beta, system.noccupied_beta, system.nunoccupied_beta))
    D3D_O = np.zeros((system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    # D diagonal
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            for b in range(system.nunoccupied_beta):
                D3D_V[a, i, b] = d3D_V(a, i, b)
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            for j in range(system.noccupied_beta):
                D3D_O[a, i, j] = d3D_O(a, i, j)

    return D3D_V, D3D_O