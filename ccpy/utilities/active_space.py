
def get_active_slices(system):

    oa = slice(0, system.noccupied_alpha - system.num_act_occupied_alpha)
    Oa = slice(system.noccupied_alpha - system.num_act_occupied_alpha, system.noccupied_alpha)
    va = slice(system.num_act_unoccupied_alpha, system.nunoccupied_alpha)
    Va = slice(0, system.num_act_unoccupied_alpha)

    ob = slice(0, system.noccupied_beta - system.num_act_occupied_beta)
    Ob = slice(system.noccupied_beta - system.num_act_occupied_beta, system.noccupied_beta)
    vb = slice(system.num_act_unoccupied_beta, system.nunoccupied_beta)
    Vb = slice(0, system.num_act_unoccupied_beta)

    return oa, Oa, va, Va, ob, Ob, vb, Vb

def active_hole(x, nocc, nact):
    if x < nocc - nact:
        return 0
    else:
        return 1

def active_particle(x, nact):
    if x < nact:
        return 1
    else:
        return 0

def fill_t3aaa(T, T0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVVOOO = T0.aaa[Va, Va, Va, Oa, Oa, Oa]
    T.aaa.VVvOOO = T0.aaa[Va, Va, va, Oa, Oa, Oa]
    T.aaa.VVVoOO = T0.aaa[Va, Va, Va, oa, Oa, Oa]
    T.aaa.VVvoOO = T0.aaa[Va, Va, va, oa, Oa, Oa]
    T.aaa.VvvOOO = T0.aaa[Va, va, va, Oa, Oa, Oa]
    T.aaa.VVVooO = T0.aaa[Va, Va, Va, oa, oa, Oa]
    T.aaa.VvvoOO = T0.aaa[Va, va, va, oa, Oa, Oa]
    T.aaa.VVvooO = T0.aaa[Va, Va, va, oa, oa, Oa]
    T.aaa.VvvooO = T0.aaa[Va, va, va, oa, oa, Oa]

    return T

def fill_t3aab(T, T0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVVOOO = T0.aab[Va, Va, Vb, Oa, Oa, Ob]
    T.aab.VVVoOO = T0.aab[Va, Va, Vb, oa, Oa, Ob]
    T.aab.VVVOOo = T0.aab[Va, Va, Vb, Oa, Oa, ob]
    T.aab.VVVoOo = T0.aab[Va, Va, Vb, oa, Oa, ob]
    T.aab.VVVooO = T0.aab[Va, Va, Vb, oa, oa, Ob]

    T.aab.VvVOOO = T0.aab[Va, va, Vb, Oa, Oa, Ob]
    T.aab.VvVoOO = T0.aab[Va, va, Vb, oa, Oa, Ob]
    T.aab.VvVOOo = T0.aab[Va, va, Vb, Oa, Oa, ob]
    T.aab.VvVoOo = T0.aab[Va, va, Vb, oa, Oa, ob]
    T.aab.VvVooO = T0.aab[Va, va, Vb, oa, oa, Ob]

    T.aab.VVvOOO = T0.aab[Va, Va, vb, Oa, Oa, Ob]
    T.aab.VVvoOO = T0.aab[Va, Va, vb, oa, Oa, Ob]
    T.aab.VVvOOo = T0.aab[Va, Va, vb, Oa, Oa, ob]
    T.aab.VVvoOo = T0.aab[Va, Va, vb, oa, Oa, ob]
    T.aab.VVvooO = T0.aab[Va, Va, vb, oa, oa, Ob]

    T.aab.VvvOOO = T0.aab[Va, va, vb, Oa, Oa, Ob]
    T.aab.VvvoOO = T0.aab[Va, va, vb, oa, Oa, Ob]
    T.aab.VvvOOo = T0.aab[Va, va, vb, Oa, Oa, ob]
    T.aab.VvvoOo = T0.aab[Va, va, vb, oa, Oa, ob]
    T.aab.VvvooO = T0.aab[Va, va, vb, oa, oa, Ob]

    T.aab.vvVOOO = T0.aab[va, va, Vb, Oa, Oa, Ob]
    T.aab.vvVoOO = T0.aab[va, va, Vb, oa, Oa, Ob]
    T.aab.vvVOOo = T0.aab[va, va, Vb, Oa, Oa, ob]
    T.aab.vvVoOo = T0.aab[va, va, Vb, oa, Oa, ob]
    T.aab.vvVooO = T0.aab[va, va, Vb, oa, oa, Ob]

    return T

def fill_t3abb(T, T0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVVOOO = T0.abb[Va, Vb, Vb, Oa, Ob, Ob]
    T.abb.VVVoOO = T0.abb[Va, Vb, Vb, oa, Ob, Ob]
    T.abb.VVVOoO = T0.abb[Va, Vb, Vb, Oa, ob, Ob]
    T.abb.VVVooO = T0.abb[Va, Vb, Vb, oa, ob, Ob]
    T.abb.VVVOoo = T0.abb[Va, Vb, Vb, Oa, ob, ob]

    T.abb.vVVOOO = T0.abb[va, Vb, Vb, Oa, Ob, Ob]
    T.abb.vVVoOO = T0.abb[va, Vb, Vb, oa, Ob, Ob]
    T.abb.vVVOoO = T0.abb[va, Vb, Vb, Oa, ob, Ob]
    T.abb.vVVooO = T0.abb[va, Vb, Vb, oa, ob, Ob]
    T.abb.vVVOoo = T0.abb[va, Vb, Vb, Oa, ob, ob]

    T.abb.VVvOOO = T0.abb[Va, Vb, vb, Oa, Ob, Ob]
    T.abb.VVvoOO = T0.abb[Va, Vb, vb, oa, Ob, Ob]
    T.abb.VVvOoO = T0.abb[Va, Vb, vb, Oa, ob, Ob]
    T.abb.VVvooO = T0.abb[Va, Vb, vb, oa, ob, Ob]
    T.abb.VVvOoo = T0.abb[Va, Vb, vb, Oa, ob, ob]

    T.abb.vVvOOO = T0.abb[va, Vb, vb, Oa, Ob, Ob]
    T.abb.vVvoOO = T0.abb[va, Vb, vb, oa, Ob, Ob]
    T.abb.vVvOoO = T0.abb[va, Vb, vb, Oa, ob, Ob]
    T.abb.vVvooO = T0.abb[va, Vb, vb, oa, ob, Ob]
    T.abb.vVvOoo = T0.abb[va, Vb, vb, Oa, ob, ob]

    T.abb.VvvOOO = T0.abb[Va, vb, vb, Oa, Ob, Ob]
    T.abb.VvvoOO = T0.abb[Va, vb, vb, oa, Ob, Ob]
    T.abb.VvvOoO = T0.abb[Va, vb, vb, Oa, ob, Ob]
    T.abb.VvvooO = T0.abb[Va, vb, vb, oa, ob, Ob]
    T.abb.VvvOoo = T0.abb[Va, vb, vb, Oa, ob, ob]

    return T

def fill_t3bbb(T, T0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVVOOO = T0.bbb[Vb, Vb, Vb, Ob, Ob, Ob]
    T.bbb.VVvOOO = T0.bbb[Vb, Vb, vb, Ob, Ob, Ob]
    T.bbb.VVVoOO = T0.bbb[Vb, Vb, Vb, ob, Ob, Ob]
    T.bbb.VVvoOO = T0.bbb[Vb, Vb, vb, ob, Ob, Ob]
    T.bbb.VvvOOO = T0.bbb[Vb, vb, vb, Ob, Ob, Ob]
    T.bbb.VVVooO = T0.bbb[Vb, Vb, Vb, ob, ob, Ob]
    T.bbb.VvvoOO = T0.bbb[Vb, vb, vb, ob, Ob, Ob]
    T.bbb.VVvooO = T0.bbb[Vb, Vb, vb, ob, ob, Ob]
    T.bbb.VvvooO = T0.bbb[Vb, vb, vb, ob, ob, Ob]

    return T

def zero_t3aaa_outside_active_space(T, system, num_act):

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):

                            x1 = active_hole(i, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x2 = active_hole(j, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x3 = active_hole(k, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x4 = active_particle(a, system.num_act_unoccupied_alpha)
                            x5 = active_particle(b, system.num_act_unoccupied_alpha)
                            x6 = active_particle(c, system.num_act_unoccupied_alpha)

                            active_h = x1 + x2 + x3
                            active_p = x4 + x5 + x6
                            if active_h < num_act or active_p < num_act:
                                T.aaa[a, b, c, i, j, k] = 0.0
                                T.aaa[a, b, c, j, i, k] = 0.0
                                T.aaa[a, b, c, j, k, i] = 0.0
                                T.aaa[a, b, c, i, k, j] = 0.0
                                T.aaa[a, b, c, k, j, i] = 0.0
                                T.aaa[a, b, c, k, i, j] = 0.0

                                T.aaa[a, c, b, i, j, k] = 0.0
                                T.aaa[a, c, b, j, i, k] = 0.0
                                T.aaa[a, c, b, j, k, i] = 0.0
                                T.aaa[a, c, b, i, k, j] = 0.0
                                T.aaa[a, c, b, k, j, i] = 0.0
                                T.aaa[a, c, b, k, i, j] = 0.0

                                T.aaa[b, a, c, i, j, k] = 0.0
                                T.aaa[b, a, c, j, i, k] = 0.0
                                T.aaa[b, a, c, j, k, i] = 0.0
                                T.aaa[b, a, c, i, k, j] = 0.0
                                T.aaa[b, a, c, k, j, i] = 0.0
                                T.aaa[b, a, c, k, i, j] = 0.0

                                T.aaa[b, c, a, i, j, k] = 0.0
                                T.aaa[b, c, a, j, i, k] = 0.0
                                T.aaa[b, c, a, j, k, i] = 0.0
                                T.aaa[b, c, a, i, k, j] = 0.0
                                T.aaa[b, c, a, k, j, i] = 0.0
                                T.aaa[b, c, a, k, i, j] = 0.0

                                T.aaa[c, a, b, i, j, k] = 0.0
                                T.aaa[c, a, b, j, i, k] = 0.0
                                T.aaa[c, a, b, j, k, i] = 0.0
                                T.aaa[c, a, b, i, k, j] = 0.0
                                T.aaa[c, a, b, k, j, i] = 0.0
                                T.aaa[c, a, b, k, i, j] = 0.0

                                T.aaa[c, b, a, i, j, k] = 0.0
                                T.aaa[c, b, a, j, i, k] = 0.0
                                T.aaa[c, b, a, j, k, i] = 0.0
                                T.aaa[c, b, a, i, k, j] = 0.0
                                T.aaa[c, b, a, k, j, i] = 0.0
                                T.aaa[c, b, a, k, i, j] = 0.0
    return T

def zero_t3aab_outside_active_space(T, system, num_act):

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(system.nunoccupied_beta):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(system.noccupied_beta):

                            x1 = active_hole(i, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x2 = active_hole(j, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x3 = active_hole(k, system.noccupied_beta, system.num_act_occupied_beta)
                            x4 = active_particle(a, system.num_act_unoccupied_alpha)
                            x5 = active_particle(b, system.num_act_unoccupied_alpha)
                            x6 = active_particle(c, system.num_act_unoccupied_beta)

                            active_h = x1 + x2 + x3
                            active_p = x4 + x5 + x6

                            if active_h < num_act or active_p < num_act:
                                T.aab[a, b, c, i, j, k] = 0.0
                                T.aab[b, a, c, i, j, k] = 0.0
                                T.aab[a, b, c, j, i, k] = 0.0
                                T.aab[b, a, c, j, i, k] = 0.0
    return T

def zero_t3abb_outside_active_space(T, system, num_act):
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for c in range(b + 1, system.nunoccupied_beta):
                for i in range(system.noccupied_alpha):
                    for j in range(system.noccupied_beta):
                        for k in range(j + 1, system.noccupied_beta):

                            x1 = active_hole(i, system.noccupied_alpha, system.num_act_occupied_alpha)
                            x2 = active_hole(j, system.noccupied_beta, system.num_act_occupied_beta)
                            x3 = active_hole(k, system.noccupied_beta, system.num_act_occupied_beta)
                            x4 = active_particle(a, system.num_act_unoccupied_alpha)
                            x5 = active_particle(b, system.num_act_unoccupied_beta)
                            x6 = active_particle(c, system.num_act_unoccupied_beta)

                            active_h = x1 + x2 + x3
                            active_p = x4 + x5 + x6

                            if active_h < num_act or active_p < num_act:
                                T.abb[a, b, c, i, j, k] = 0.0
                                T.abb[a, c, b, i, j, k] = 0.0
                                T.abb[a, b, c, i, k, j] = 0.0
                                T.abb[a, c, b, i, k, j] = 0.0
    return T


def zero_t3bbb_outside_active_space(T, system, num_act):
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for c in range(b + 1, system.nunoccupied_beta):
                for i in range(system.noccupied_beta):
                    for j in range(i + 1, system.noccupied_beta):
                        for k in range(j + 1, system.noccupied_beta):

                            x1 = active_hole(i, system.noccupied_beta, system.num_act_occupied_beta)
                            x2 = active_hole(j, system.noccupied_beta, system.num_act_occupied_beta)
                            x3 = active_hole(k, system.noccupied_beta, system.num_act_occupied_beta)
                            x4 = active_particle(a, system.num_act_unoccupied_beta)
                            x5 = active_particle(b, system.num_act_unoccupied_beta)
                            x6 = active_particle(c, system.num_act_unoccupied_beta)

                            active_h = x1 + x2 + x3
                            active_p = x4 + x5 + x6
                            if active_h < num_act or active_p < num_act:
                                T.bbb[a, b, c, i, j, k] = 0.0
                                T.bbb[a, b, c, j, i, k] = 0.0
                                T.bbb[a, b, c, j, k, i] = 0.0
                                T.bbb[a, b, c, i, k, j] = 0.0
                                T.bbb[a, b, c, k, j, i] = 0.0
                                T.bbb[a, b, c, k, i, j] = 0.0

                                T.bbb[a, c, b, i, j, k] = 0.0
                                T.bbb[a, c, b, j, i, k] = 0.0
                                T.bbb[a, c, b, j, k, i] = 0.0
                                T.bbb[a, c, b, i, k, j] = 0.0
                                T.bbb[a, c, b, k, j, i] = 0.0
                                T.bbb[a, c, b, k, i, j] = 0.0

                                T.bbb[b, a, c, i, j, k] = 0.0
                                T.bbb[b, a, c, j, i, k] = 0.0
                                T.bbb[b, a, c, j, k, i] = 0.0
                                T.bbb[b, a, c, i, k, j] = 0.0
                                T.bbb[b, a, c, k, j, i] = 0.0
                                T.bbb[b, a, c, k, i, j] = 0.0

                                T.bbb[b, c, a, i, j, k] = 0.0
                                T.bbb[b, c, a, j, i, k] = 0.0
                                T.bbb[b, c, a, j, k, i] = 0.0
                                T.bbb[b, c, a, i, k, j] = 0.0
                                T.bbb[b, c, a, k, j, i] = 0.0
                                T.bbb[b, c, a, k, i, j] = 0.0

                                T.bbb[c, a, b, i, j, k] = 0.0
                                T.bbb[c, a, b, j, i, k] = 0.0
                                T.bbb[c, a, b, j, k, i] = 0.0
                                T.bbb[c, a, b, i, k, j] = 0.0
                                T.bbb[c, a, b, k, j, i] = 0.0
                                T.bbb[c, a, b, k, i, j] = 0.0

                                T.bbb[c, b, a, i, j, k] = 0.0
                                T.bbb[c, b, a, j, i, k] = 0.0
                                T.bbb[c, b, a, j, k, i] = 0.0
                                T.bbb[c, b, a, i, k, j] = 0.0
                                T.bbb[c, b, a, k, j, i] = 0.0
                                T.bbb[c, b, a, k, i, j] = 0.0
    return T