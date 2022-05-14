import numpy as np

def select_triples_from_full_moments(moments_aaa, moments_aab, moments_abb, moments_bbb, pspace, num_add, system):

    n3_aaa = system.noccupied_alpha**3 * system.nunoccupied_alpha**3
    n3_aab = system.noccupied_alpha**2 * system.noccupied_beta * system.nunoccupied_alpha**2 * system.nunoccupied_beta
    n3_abb = system.noccupied_beta**2 * system.noccupied_alpha * system.nunoccupied_beta**2 * system.nunoccupied_alpha
    n3_bbb = system.noccupied_beta**3 * system.nunoccupied_beta**3

    # check if flattening/sorting/unraveling is correct
    mvec = np.hstack([moments_aaa.flatten(), moments_aab.flatten(), moments_abb.flatten(), moments_bbb.flatten()])
    idx = np.flip(np.argsort(abs(mvec)))

    # should these be copied?
    new_pspace = {
        "aaa": pspace["aaa"],
        "aab": pspace["aab"],
        "abb": pspace["abb"],
        "bbb": pspace["bbb"],
    }

    # this loop is not conserving spin symmetry!
    ct = 0
    ct2 = 0
    while ct < num_add:

        if idx[ct2] < n3_aaa:
            a, b, c, i, j, k = np.unravel_index(idx[ct2], moments_aaa.shape)
            if pspace["aaa"][a, b, c, i, j, k] == 1:
                ct2 += 1
                continue
            else:
                ct += 1
                new_pspace["aaa"][a, b, c, i, j, k] = 1
        elif idx[ct2] < n3_aaa + n3_aab:
            a, b, c, i, j, k = np.unravel_index(idx[ct2] - n3_aaa, moments_aab.shape)
            if pspace["aab"][a, b, c, i, j, k] == 1:
                ct2 += 1
                continue
            else:
                ct += 1
                new_pspace["aab"][a, b, c, i, j, k] = 1
        elif idx[ct2] < n3_aaa + n3_aab + n3_abb:
            a, b, c, i, j, k = np.unravel_index(idx[ct2] - n3_aaa - n3_aab, moments_abb.shape)
            if pspace["abb"][a, b, c, i, j, k] == 1:
                ct2 += 1
                continue
            else:
                ct += 1
                new_pspace["abb"][a, b, c, i, j, k] = 1
        else:
            a, b, c, i, j, k = np.unravel_index(idx[ct2] - n3_aaa - n3_aab - n3_abb, moments_bbb.shape)
            if pspace["bbb"][a, b, c, i, j, k] == 1:
                ct2 += 1
                continue
            else:
                ct += 1
                new_pspace["bbb"][a, b, c, i, j, k] = 1

    return new_pspace, ct

def select_triples_from_moments(triples_list, pspace):
    """Expand the size of the previous P space using the determinants contained in the list
    of triples (stored as a, b, c, i, j, k) in triples_list. The variable triples_list stores
    triples in spinorbital form, where all orbital indices start from 1 and odd indices
    correspond to alpha orbitals while even indices corresopnd to beta orbitals."""

    # should these be copied?
    new_pspace = {
        "aaa": pspace["aaa"],
        "aab": pspace["aab"],
        "abb": pspace["abb"],
        "bbb": pspace["bbb"],
    }

    num_add = triples_list.shape[0]
    for n in range(num_add):
        num_beta = sum([x % 2 for x in triples_list[n, :]])
        a, b, c, i, j, k = triples_list[n, :]
        if num_beta == 3:
            new_pspace['bbb'][a, b, c, i, j, k] = 1
        elif num_beta == 2:
            new_pspace['abb'][a, b, c, i, j, k] = 1
        elif num_beta == 1:
            new_pspace['aab'][a, b, c, i, j, k] = 1
        else:
            new_pspace['aaa'][a, b, c, i, j, k] = 1

    return new_pspace

