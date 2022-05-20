import numpy as np
from numba import njit

@njit(nopython=True)
def calc_aaa_H_aaa(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_holes_alpha = exc[1, 0, 0, 0]
    num_holes_beta = exc[1, 0, 1, 0]
    num_particles_alpha = exc[1, 1, 0, 0]
    num_particles_beta = exc[1, 1, 1, 0]

    common_holes_alpha = common[0, 0, 1:common[0, 0, 0] + 1]
    common_holes_beta = common[0, 1, 1:common[0, 1, 0] + 1]
    common_particles_alpha = common[1, 0, 1:common[1, 0, 0] + 1]
    common_particles_beta = common[1, 1, 1:common[1, 1, 0] + 1]

    degree = int(sum(exc[:, :, :, 0].flatten())/2 )
    
    onebody = 0.0
    twobody = 0.0
    threebody = 0.0

    j1, k1 = exc[0, 0, 0, 1:]
    b1, c1 = exc[0, 1, 0, 1:]
    j2, k2 = exc[1, 0, 0, 1:]
    b2, c2 = exc[1, 1, 0, 1:]

    if degree == 0:
        # diagonal
        for i in common_holes_alpha:
            onebody -= H1A['oo'][i, i]
            for j in common_holes_alpha:
                twobody += 0.5 * H2A['oooo'][i, j, i, j]
        for a in common_particles_alpha:
            onebody += H1A['vv'][a, a]
            for b in common_particles_alpha:
                twobody += 0.5 * H2A['vvvv'][a, b, a, b]
                for i in common_holes_alpha:
                    threebody += 0.5 * H3A['vovovv'][a, i, b, i, a, b]
            for i in common_holes_alpha:
                twobody += H2A['voov'][a, i, i, a]
                for j in common_holes_alpha:
                    threebody -= 0.5 * H3A['vooovo'][a, i, j, i, a, j]

    elif degree == 1:
        # case: 1h(a)
        if num_holes_alpha == 1:
            onebody -= H1A['oo'][j2, j1]
            for a in common_particles_alpha:
                twobody += H2A['voov'][a, j2, j1, a]
                for i in common_particles_alpha:
                    threebody -= H3A['vooovo'][a, i, j2, i, a, j1]
                for b in common_particles_alpha:
                    threebody += 0.5 * H3A['vovovv'][a, j2, b, j1, a, b]
        # case: 1p(a)
        if num_particles_alpha == 1:
            onebody += H1A['vv'][b1, b2]
            for i in common_holes_alpha:
                twobody += H2A['voov'][b1, i, i, b2]
                for j in common_holes_alpha:
                    threebody -= 0.5 * H3A['vooovo'][b1, i, j, i, b2, j]
                for a in common_particles_alpha:
                    threebody += H3A['vovovv'][a, i, b1, i, a, b2]

    elif degree == 2:
        # case: 2h(a)
        if num_holes_alpha == 2:
            twobody += H2A['oooo'][j2, k2, j1, k1]
            for a in common_particles_alpha:
                threebody -= H3A['vooovo'][a, j2, k2, j1, a, k1]
        # case: 2p(a)
        if num_particles_alpha == 2:
            twobody += H2A['vvvv'][b1, c1, b2, c2]
            for i in common_holes_alpha:
                threebody += H3A['vovovv'][b1, i, c1, i, b2, c2]
        # case: 1h(a)-1p(a)
        if num_particles_alpha == 1 and num_holes_alpha == 1:
            twobody += H2A['voov'][b1, j2, j1, b2]
            for a in common_particles_alpha:
                threebody += H3A['vovovv'][a, j2, b1, j1, a, b2]
            for i in common_holes_alpha:
                threebody -= H3A['vooovo'][b1, i, j2, i, b2, j1]

    elif degree == 3:
        # case: 2h(a)-1p(a)
        if num_holes_alpha == 2 and num_particles_alpha == 1:
            threebody -= H3A['vooovo'][b1, j2, k2, j1, b2, k1]
        # case: 1h(a)-2p(a)
        if num_holes_alpha == 1 and num_particles_alpha == 2:
            threebody += H3A['vovovv'][b1, j2, c1, j1, b2, c2]

    return onebody + twobody + threebody

@njit(nopython=True)
def calc_aaa_H_aab(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_holes_alpha = exc[1, 0, 0, 0]
    num_holes_beta = exc[1, 0, 1, 0]
    num_particles_alpha = exc[1, 1, 0, 0]
    num_particles_beta = exc[1, 1, 1, 0]

    common_holes_alpha = common[0, 0, 1:common[0, 0, 0] + 1]
    common_holes_beta = common[0, 1, 1:common[0, 1, 0] + 1]
    common_particles_alpha = common[1, 0, 1:common[1, 0, 0] + 1]
    common_particles_beta = common[1, 1, 1:common[1, 1, 0] + 1]

    degree = int(sum(exc[:, :, :, 0].flatten())/2 )
    
    onebody = 0.0
    twobody = 0.0
    threebody = 0.0

    j1, k1 = exc[0, 0, 0, 1:]
    b1, c1 = exc[0, 1, 0, 1:]
    j2a, k2a = exc[1, 0, 0, 1:]
    b2a, c2a = exc[1, 1, 0, 1:]
    j2b, k2b = exc[1, 0, 1, 1:]
    b2b, c2b = exc[1, 1, 1, 1:]

    if degree == 2:
        # case: 1h(b)-1p(b)
        if num_holes_beta == 1 and num_particles_beta == 1:
            twobody += H2B['voov'][b1, j2b, j1, b2b]
            for a in common_particles_alpha:
                threebody += H3B['vvoovv'][b1, a, j2b, j1, a, b2b]
            for i in common_particles_alpha:
                threebody -= H3B['voooov'][b1, i, j2b, j1, i, b2b]

    elif degree == 3:
        # case: 1h(a)-1h(b)-1p(b)
        if num_holes_alpha == 1 and num_holes_beta == 1 and num_particles_beta == 1:
            threebody -= H3B['voooov'][b1, j2a, j2b, j1, k1, b2b] 
        # case: 1h(b)-1p(a)-1p(b)
        if num_holes_beta == 1 and num_particles_alpha == 1 and num_particles_beta == 1:
            threebody += H3B['vvoovv'][b1, c1, j2b, j1, b2a, b2b]

    return onebody + twobody + threebody


@njit(nopython=True)
def calc_aab_H_aaa(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_holes_alpha = exc[1, 0, 0, 0]
    num_holes_beta = exc[1, 0, 1, 0]
    num_particles_alpha = exc[1, 1, 0, 0]
    num_particles_beta = exc[1, 1, 1, 0]

    common_holes_alpha = common[0, 0, 1:common[0, 0, 0] + 1]
    common_holes_beta = common[0, 1, 1:common[0, 1, 0] + 1]
    common_particles_alpha = common[1, 0, 1:common[1, 0, 0] + 1]
    common_particles_beta = common[1, 1, 1:common[1, 1, 0] + 1]

    degree = int(sum(exc[:, :, :, 0].flatten())/2 )
    
    onebody = 0.0
    twobody = 0.0
    threebody = 0.0

    k1a, j1a = exc[0, 0, 0, 1:]
    k1b, j1b = exc[0, 0, 1, 1:]
    c1a, b1a = exc[0, 1, 0, 1:]
    c1b, b1b = exc[0, 1, 1, 1:]
    k2, j2 = exc[1, 0, 0, 1:]
    c2, b2 = exc[1, 1, 0, 1:]

    if degree == 2:
        # case: 1h(a)-1p(a)
        if num_holes_alpha == 1 and num_particles_alpha == 1:
            twobody += H2B['ovvo'][k2, c1b, c2, k1b]
            for a in common_particles_alpha:
                threebody += H3B['vovvvo'][a, k2, c1b, a, c2, k1b]
            for i in common_holes_alpha:
                threebody -= H3B['oovovo'][i, k2, c1b, i, c2, k1b]

    elif degree == 3:
        # case: 2h(a)-1p(a)
        if num_holes_alpha == 2 and num_particles_alpha == 1:
            threebody -= H3B['oovovo'][j2, k2, c1b, k1a, c2, k1b]
        # case: 1h(a)-2p(a)
        if num_holes_alpha == 1 and num_particles_alpha == 2:
            threebody += H3B['vovvvo'][c1a, k2, c1b, b2, c2, k1b]

    return onebody + twobody + threebody


def calc_aab_H_aab(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_holes_alpha = exc[1, 0, 0, 0]
    num_holes_beta = exc[1, 0, 1, 0]
    num_particles_alpha = exc[1, 1, 0, 0]
    num_particles_beta = exc[1, 1, 1, 0]

    common_holes_alpha = common[0, 0, 1:common[0, 0, 0] + 1]
    common_holes_beta = common[0, 1, 1:common[0, 1, 0] + 1]
    common_particles_alpha = common[1, 0, 1:common[1, 0, 0] + 1]
    common_particles_beta = common[1, 1, 1:common[1, 1, 0] + 1]

    degree = int(sum(exc[:, :, :, 0].flatten())/2 )

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0

    j1, i1 = exc[0, 0, 0, 1:] # different holes alpha 1
    k1, _ = exc[0, 0, 1, 1:]  # different hole beta 1
    b1, a1 = exc[0, 1, 0, 1:] # different particles alpha 1
    c1, _ = exc[0, 1, 1, 1:]  # different particle beta 1
    j2, i1 = exc[1, 0, 0, 1:] # different holes alpha 2
    k2, _ = exc[1, 0, 1, 1:]  # different hole beta 2
    b1, a1 = exc[1, 1, 0, 1:] # different particles alpha 2
    c2, _ = exc[1, 1, 1, 1:]  # different particle beta 2
    
    if degree == 0:
        # diagonal
        pass
    elif degree == 1:
        # case: 1h(a)
        if num_holes_alpha == 1:
            onebody -= H1A['oo'][j2, j1]

            for a in common_particles_alpha:
                twobody += H2A['voov'][a, j2, j1, a]
            for c in common_particles_beta:
                twobody -= H2B['ovov'][j2, c, j1, c]
            for i in common_holes_alpha:
                twobody += H2A['oooo'][i, j2, i, j1]
            for k in common_holes_beta:
                twobody += H2B['oooo'][j2, k, j1, k]

            for a in common_particles_alpha:
                for i in common_holes_alpha:
                    threebody -= H3A['vooovo'][a, i, j2, i, a, j1]
            for a in common_particles_alpha:
                for k in common_holes_beta:
                    threebody -= H3B['vooovo'][a, j2, k, j1, a, k]
            for c in common_particles_beta:
                for k in common_holes_beta:
                    threebody -= H3C['ovooov'][j2, c, k, j1, k, c]
            for a in common_particles_alpha:
                for b in common_particles_alpha:
                    threebody += 0.5 * H3A['vovovv'][a, j2, b, j1, a, b]
            for a in common_particles_alpha:
                for c in common_particles_beta:
                    threebody += H3B['vovovv'][a, j2, c, j1, a, c]


        # case: 1h(b)
        if num_holes_beta == 1:
            onebody = -H1B['oo'][k2, k1]

            for a in common_particles_alpha:
                twobody -= H2B['vovo'][a, k2, a, k1]
            for c in common_particles_beta:
                twobody += H2C['voov'][c, k2, k1, c]
            for i in common_holes_alpha:
                twobody += H2B['oooo'][i, k2, i, k1]
    
            for a in common_particles_alpha:
                for i in common_holes_alpha:
                    threebody -= H3B['vooovo'][a, i, k2, i, a, k1]
            for a in common_particles_alpha:
                for c in common_particles_beta:
                    threebody += H3C['vvovov'][a, c, k2, a, k1, c]
            for i in common_holes_alpha:
                for c in common_particles_beta:
                    threebody -= H3C['ovooov'][i, c, k2, i, k1, c]

        # case: 1p(a)
        onebody = H1A['vv'][b1, b2]

        for i in common_holes_alpha:
            twobody += H2A['voov'][b1, j, j, b2]
        for k in common_holes_beta:
            twobody -= H2B['vovo'][b1, k, k, b2]
        for a in common_particles_alpha:
            twobody += H2B['vvvv'][a, b1, a, b2]
        for c in common_particles_beta:
            twobody += H2B['vvvv'][b1, c, b2, c]

        for i in common_holes_alpha:
            for j in common_holes_alpha:
                threebody -= 0.5 * H3A['vooovo'][b1, i, j, i, b2, j]
        for i in common_holes_alpha:
            for k in common_holes_beta:
                threebody -= H3B['vooovo'][b1, i, k, i, b2, k]
        for a in common_particles_alpha:
            for i in common_holes_alpha:
                threebody += H3A['vovovv'][b1, i, a, i, b2, a]
        for i in common_holes_alpha:
            for c in common_particles_beta:
                threebody += H3B['vovovv'][b1, i, c, i, b2, c]
        for c in common_particles_beta:
            for k in common_holes_beta:
                threebody += H3C['vvovov'][b1, c, k, b2, k, c]

        # case: 1p(b)
        onebody += H1B['vv'][c1, c2]
        
        for k in common_holes_beta:
            twobody += H2C['voov'][c1, k, k, c2]
    elif degree == 2: pass
        # case: 1h(a)-1h(b)
        # case: 2h(a)
        # case: 1p(a)-1p(b)
        # case: 2p(a)
        # case: 1h(a)-1p(a)
        # case: 1h(a)-1p(b)
        # case: 1h(b)-1p(a)
        # case: 1h(b)-1p(b)
    elif degree == 3: pass
        # case: 2h(a)-1p(a)
        # case: 2h(a)-1p(b)
        # case: 1h(a)-1h(b)-1p(a)
        # case: 1h(a)-1h(b)-1p(b)
        # case: 1h(a)-2p(a)
        # case: 1h(b)-2p(a)
        # case: 1h(a)-1p(a)-1p(b)
        # case: 1h(b)-1p(a)-1p(b)

    return onebody + twobody + threebody


def calc_aab_H_abb(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 2: pass
        # case: 1h(b)-1p(b)
    elif degree == 3: pass
        # case: 1h(a)-1h(b)-1p(b)
        # case: 2h(b)-1p(b)
        # case: 1h(b)-1p(a)-1p(b)
        # case: 1h(b)-2p(b)

    return onebody + twobody + threebody

def calc_abb_H_aab(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 2: pass
        # case: 1h(a)-1p(a)
    elif degree == 3: pass
        # case: 2h(a)-1p(a)
        # case: 1h(a)-1h(b)-1p(a)
        # case: 1h(a)-2p(a)
        # case: 1h(a)-1p(a)-1p(b)

    return onebody + twobody + threebody

def calc_abb_H_abb(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 0: pass
        # diagonal
    elif degree == 1: pass
        # case: 1h(a)
        # case: 1h(b)
        # case: 1p(a)
        # case: 1p(b)
    elif degree == 2: pass
        # case: 1h(a)-1h(b)
        # case: 2h(b)
        # case: 1p(a)-1p(b)
        # case: 2p(b)
        # case: 1h(a)-1p(a)
        # case: 1h(a)-1p(b)
        # case: 1h(b)-1p(a)
        # case: 1h(b)-1p(b)
    elif degree == 3: pass
        # case: 1h(a)-1h(b)-1p(a)
        # case: 1h(a)-1h(b)-1p(b)
        # case: 2h(b)-1p(a)
        # case: 2h(b)-1p(b)
        # case: 1h(a)-1p(a)-1p(b)
        # case: 1h(b)-1p(a)-1p(b)
        # case: 1h(a)-2p(b)
        # case: 1h(b)-2p(b)

    return onebody + twobody + threebody

def calc_abb_H_bbb(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 2: pass
        # case: 1h(b)-1p(b)
    elif degree == 3: pass
        # case: 1h(a)-1h(b)-1p(b)
        # case: 1h(b)-1p(a)-1p(b)

    return onebody + twobody + threebody

def calc_bbb_H_abb(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 2: pass
        # case: 1h(a)-1p(a)
    elif degree == 3: pass
        # case: 1h(a)-1h(b)-1p(a)
        # case: 1h(a)-1p(a)-1p(b)

    return onebody + twobody + threebody


def calc_bbb_H_bbb(exc, common, H1A, H1B, H2A, H2B, H2C):

    num_ha1 = exc[0, 0, 0, 0]
    num_ha2 = exc[1, 0, 0, 0]
    num_hb1 = exc[0, 0, 1, 0]
    num_hb2 = exc[1, 0, 1, 0]
    num_pa1 = exc[0, 1, 0, 0]
    num_pa2 = exc[1, 1, 0, 0]
    num_pb1 = exc[0, 1, 1, 0]
    num_pb2 = exc[1, 1, 1, 0]

    num_comm_ha = common[0, 0, 0]
    num_comm_hb = common[0, 1, 0]
    num_comm_pa = common[1, 0, 0]
    num_comm_pb = common[1, 1, 0]

    common_holes_alpha = common[0, 0, 1:num_comm_ha+1]
    common_holes_beta = common[0, 1, 1:num_comm_hb+1]
    common_particles_alpha = common[1, 0, 1:num_comm_pa+1]
    common_particles_beta = common[1, 1, 1:num_comm_pb+1]


    degree = (num_ha1 + num_ha2 + num_hb1 + num_hb2 + num_pa1 + num_pa2 + num_pb1 + num_pb2)/2

    onebody = 0.0
    twobody = 0.0
    threebody = 0.0
    
    if degree == 0: pass
        # diagonal
    elif degree == 1: pass
        # case: 1h(b)
        # case: 1p(b)
    elif degree == 2: pass
        # case: 2h(b)
        # case: 2p(b)
        # case: 1h(b)-1p(b)
    elif degree == 3: pass
        # case: 2h(b)-1p(b)
        # case: 1h(b)-2p(b)

    return onebody + twobody + threebody
