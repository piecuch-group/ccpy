import numpy as np

def select_triples_by_moments(momets_aaa, moments_aab, moments_abb, moments_bbb, pspace, num_add, system):

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
