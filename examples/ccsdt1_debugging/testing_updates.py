import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update_t3a_full(T, dT, H0, system):

    for i in range(system.noccupied_alpha):
        for j in range(i+1, system.noccupied_alpha):
            for k in range(j+1, system.noccupied_alpha):
                for a in range(system.nunoccupied_alpha):
                    for b in range(a+1, system.nunoccupied_alpha):
                        for c in range(b+1, system.nunoccupied_alpha):

                            denom = (
                                    H0.a.oo[i, i] + H0.a.oo[j, j] + H0.a.oo[k, k]
                                   -H0.a.vv[a, a] - H0.a.vv[b, b] - H0.a.vv[c, c]
                            )

                            val = dT.aaa[a, b, c, i, j, k] / denom

                            T.aaa[a, b, c, i, j, k] += val
                            T.aaa[a, b, c, i, k, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[a, b, c, j, k, i] = T.aaa[a, b, c, i, j, k]
                            T.aaa[a, b, c, j, i, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[a, b, c, k, i, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[a, b, c, k, j, i] = -1.0 * T.aaa[a, b, c, i, j, k]

                            T.aaa[a, c, b, i, j, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[a, c, b, i, k, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[a, c, b, j, k, i] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[a, c, b, j, i, k] = T.aaa[a, b, c, i, j, k]
                            T.aaa[a, c, b, k, i, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[a, c, b, k, j, i] = T.aaa[a, b, c, i, j, k]

                            T.aaa[b, c, a, i, j, k] = T.aaa[a, b, c, i, j, k]
                            T.aaa[b, c, a, i, k, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[b, c, a, j, k, i] = T.aaa[a, b, c, i, j, k]
                            T.aaa[b, c, a, j, i, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[b, c, a, k, i, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[b, c, a, k, j, i] = -1.0 * T.aaa[a, b, c, i, j, k]

                            T.aaa[b, a, c, i, j, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[b, a, c, i, k, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[b, a, c, j, k, i] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[b, a, c, j, i, k] = T.aaa[a, b, c, i, j, k]
                            T.aaa[b, a, c, k, i, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[b, a, c, k, j, i] = T.aaa[a, b, c, i, j, k]

                            T.aaa[c, a, b, i, j, k] = T.aaa[a, b, c, i, j, k]
                            T.aaa[c, a, b, i, k, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[c, a, b, j, k, i] = T.aaa[a, b, c, i, j, k]
                            T.aaa[c, a, b, j, i, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[c, a, b, k, i, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[c, a, b, k, j, i] = -1.0 * T.aaa[a, b, c, i, j, k]

                            T.aaa[c, b, a, i, j, k] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[c, b, a, i, k, j] = T.aaa[a, b, c, i, j, k]
                            T.aaa[c, b, a, j, k, i] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[c, b, a, j, i, k] = T.aaa[a, b, c, i, j, k]
                            T.aaa[c, b, a, k, i, j] = -1.0 * T.aaa[a, b, c, i, j, k]
                            T.aaa[c, b, a, k, j, i] = T.aaa[a, b, c, i, j, k]

                            dT.aaa[a, b, c, i, j, k] = val
                            dT.aaa[a, b, c, i, k, j] = -1.0 * val
                            dT.aaa[a, b, c, j, k, i] = val
                            dT.aaa[a, b, c, j, i, k] = -1.0 * val
                            dT.aaa[a, b, c, k, i, j] = val
                            dT.aaa[a, b, c, k, j, i] = -1.0 * val

                            dT.aaa[a, c, b, i, j, k] = -1.0 * val
                            dT.aaa[a, c, b, i, k, j] = val
                            dT.aaa[a, c, b, j, k, i] = -1.0 * val
                            dT.aaa[a, c, b, j, i, k] = val
                            dT.aaa[a, c, b, k, i, j] = -1.0 * val
                            dT.aaa[a, c, b, k, j, i] = val

                            dT.aaa[b, c, a, i, j, k] = val
                            dT.aaa[b, c, a, i, k, j] = -1.0 * val
                            dT.aaa[b, c, a, j, k, i] = val
                            dT.aaa[b, c, a, j, i, k] = -1.0 * val
                            dT.aaa[b, c, a, k, i, j] = val
                            dT.aaa[b, c, a, k, j, i] = -1.0 * val

                            dT.aaa[b, a, c, i, j, k] = -1.0 * val
                            dT.aaa[b, a, c, i, k, j] = val
                            dT.aaa[b, a, c, j, k, i] = -1.0 * val
                            dT.aaa[b, a, c, j, i, k] = val
                            dT.aaa[b, a, c, k, i, j] = -1.0 * val
                            dT.aaa[b, a, c, k, j, i] = val

                            dT.aaa[c, a, b, i, j, k] = val
                            dT.aaa[c, a, b, i, k, j] = -1.0 * val
                            dT.aaa[c, a, b, j, k, i] = val
                            dT.aaa[c, a, b, j, i, k] = -1.0 * val
                            dT.aaa[c, a, b, k, i, j] = val
                            dT.aaa[c, a, b, k, j, i] = -1.0 * val

                            dT.aaa[c, b, a, i, j, k] = -1.0 * val
                            dT.aaa[c, b, a, i, k, j] = val
                            dT.aaa[c, b, a, j, k, i] = -1.0 * val
                            dT.aaa[c, b, a, j, i, k] = val
                            dT.aaa[c, b, a, k, i, j] = -1.0 * val
                            dT.aaa[c, b, a, k, j, i] = val


    return T, dT

def update_t3a_111111(T, dT, H0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    for i in range(system.num_act_occupied_alpha):
        for j in range(i+1, system.num_act_occupied_alpha):
            for k in range(j+1, system.num_act_occupied_alpha):
                for a in range(system.num_act_unoccupied_alpha):
                    for b in range(a+1, system.num_act_unoccupied_alpha):
                        for c in range(b+1, system.num_act_unoccupied_alpha):

                            denom = (
                                    H0.a.oo[Oa, Oa][i, i] + H0.a.oo[Oa, Oa][j, j] + H0.a.oo[Oa, Oa][k, k]
                                    - H0.a.vv[Va, Va][a, a] - H0.a.vv[Va, Va][b, b] - H0.a.vv[Va, Va][c, c]
                            )

                            val = dT.aaa.VVVOOO[a, b, c, i, j, k] / denom

                            T.aaa.VVVOOO[a, b, c, i, j, k] += val
                            T.aaa.VVVOOO[a, b, c, i, k, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, b, c, j, k, i] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, b, c, j, i, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, b, c, k, i, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, b, c, k, j, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]

                            T.aaa.VVVOOO[a, c, b, i, j, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, c, b, i, k, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, c, b, j, k, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, c, b, j, i, k] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, c, b, k, i, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[a, c, b, k, j, i] = T.aaa.VVVOOO[a, b, c, i, j, k]

                            T.aaa.VVVOOO[b, c, a, i, j, k] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, c, a, i, k, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, c, a, j, k, i] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, c, a, j, i, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, c, a, k, i, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, c, a, k, j, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]

                            T.aaa.VVVOOO[b, a, c, i, j, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, a, c, i, k, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, a, c, j, k, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, a, c, j, i, k] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, a, c, k, i, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[b, a, c, k, j, i] = T.aaa.VVVOOO[a, b, c, i, j, k]

                            T.aaa.VVVOOO[c, a, b, i, j, k] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, a, b, i, k, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, a, b, j, k, i] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, a, b, j, i, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, a, b, k, i, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, a, b, k, j, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]

                            T.aaa.VVVOOO[c, b, a, i, j, k] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, b, a, i, k, j] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, b, a, j, k, i] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, b, a, j, i, k] = T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, b, a, k, i, j] = -1.0 * T.aaa.VVVOOO[a, b, c, i, j, k]
                            T.aaa.VVVOOO[c, b, a, k, j, i] = T.aaa.VVVOOO[a, b, c, i, j, k]

                            dT.aaa.VVVOOO[a, b, c, i, j, k] = val
                            dT.aaa.VVVOOO[a, b, c, i, k, j] = -1.0 * val
                            dT.aaa.VVVOOO[a, b, c, j, k, i] = val
                            dT.aaa.VVVOOO[a, b, c, j, i, k] = -1.0 * val
                            dT.aaa.VVVOOO[a, b, c, k, i, j] = val
                            dT.aaa.VVVOOO[a, b, c, k, j, i] = -1.0 * val

                            dT.aaa.VVVOOO[a, c, b, i, j, k] = -1.0 * val
                            dT.aaa.VVVOOO[a, c, b, i, k, j] = val
                            dT.aaa.VVVOOO[a, c, b, j, k, i] = -1.0 * val
                            dT.aaa.VVVOOO[a, c, b, j, i, k] = val
                            dT.aaa.VVVOOO[a, c, b, k, i, j] = -1.0 * val
                            dT.aaa.VVVOOO[a, c, b, k, j, i] = val

                            dT.aaa.VVVOOO[b, c, a, i, j, k] = val
                            dT.aaa.VVVOOO[b, c, a, i, k, j] = -1.0 * val
                            dT.aaa.VVVOOO[b, c, a, j, k, i] = val
                            dT.aaa.VVVOOO[b, c, a, j, i, k] = -1.0 * val
                            dT.aaa.VVVOOO[b, c, a, k, i, j] = val
                            dT.aaa.VVVOOO[b, c, a, k, j, i] = -1.0 * val

                            dT.aaa.VVVOOO[b, a, c, i, j, k] = -1.0 * val
                            dT.aaa.VVVOOO[b, a, c, i, k, j] = val
                            dT.aaa.VVVOOO[b, a, c, j, k, i] = -1.0 * val
                            dT.aaa.VVVOOO[b, a, c, j, i, k] = val
                            dT.aaa.VVVOOO[b, a, c, k, i, j] = -1.0 * val
                            dT.aaa.VVVOOO[b, a, c, k, j, i] = val

                            dT.aaa.VVVOOO[c, a, b, i, j, k] = val
                            dT.aaa.VVVOOO[c, a, b, i, k, j] = -1.0 * val
                            dT.aaa.VVVOOO[c, a, b, j, k, i] = val
                            dT.aaa.VVVOOO[c, a, b, j, i, k] = -1.0 * val
                            dT.aaa.VVVOOO[c, a, b, k, i, j] = val
                            dT.aaa.VVVOOO[c, a, b, k, j, i] = -1.0 * val

                            dT.aaa.VVVOOO[c, b, a, i, j, k] = -1.0 * val
                            dT.aaa.VVVOOO[c, b, a, i, k, j] = val
                            dT.aaa.VVVOOO[c, b, a, j, k, i] = -1.0 * val
                            dT.aaa.VVVOOO[c, b, a, j, i, k] = val
                            dT.aaa.VVVOOO[c, b, a, k, i, j] = -1.0 * val
                            dT.aaa.VVVOOO[c, b, a, k, j, i] = val


    return T, dT

def update_t3a_110111(T, dT, H0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    for i in range(system.num_act_occupied_alpha):
        for j in range(i+1, system.num_act_occupied_alpha):
            for k in range(j+1, system.num_act_occupied_alpha):
                for a in range(system.num_act_unoccupied_alpha):
                    for b in range(a+1, system.num_act_unoccupied_alpha):
                        for c in range(system.num_virt_alpha):

                            denom = (
                                    H0.a.oo[Oa, Oa][i, i] + H0.a.oo[Oa, Oa][j, j] + H0.a.oo[Oa, Oa][k, k]
                                   -H0.a.vv[Va, Va][a, a] - H0.a.vv[Va, Va][b, b] - H0.a.vv[va, va][c, c]
                            )
                            val = dT.aaa.VVvOOO[a, b, c, i, j, k] / denom

                            T.aaa.VVvOOO[a, b, c, i, j, k] += val
                            T.aaa.VVvOOO[a, b, c, i, k, j] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[a, b, c, j, k, i] = T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[a, b, c, j, i, k] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[a, b, c, k, i, j] = T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[a, b, c, k, j, i] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]

                            T.aaa.VVvOOO[b, a, c, i, j, k] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[b, a, c, i, k, j] = T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[b, a, c, j, k, i] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[b, a, c, j, i, k] = T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[b, a, c, k, i, j] = -1.0 * T.aaa.VVvOOO[a, b, c, i, j, k]
                            T.aaa.VVvOOO[b, a, c, k, j, i] = T.aaa.VVvOOO[a, b, c, i, j, k]

                            dT.aaa.VVvOOO[a, b, c, i, j, k] = val
                            dT.aaa.VVvOOO[a, b, c, i, k, j] = -1.0 * val
                            dT.aaa.VVvOOO[a, b, c, j, k, i] = val
                            dT.aaa.VVvOOO[a, b, c, j, i, k] = -1.0 * val
                            dT.aaa.VVvOOO[a, b, c, k, i, j] = val
                            dT.aaa.VVvOOO[a, b, c, k, j, i] = -1.0 * val

                            dT.aaa.VVvOOO[b, a, c, i, j, k] = -1.0 * val
                            dT.aaa.VVvOOO[b, a, c, i, k, j] = val
                            dT.aaa.VVvOOO[b, a, c, j, k, i] = -1.0 * val
                            dT.aaa.VVvOOO[b, a, c, j, i, k] = val
                            dT.aaa.VVvOOO[b, a, c, k, i, j] = -1.0 * val
                            dT.aaa.VVvOOO[b, a, c, k, j, i] = val
    return T, dT

def update_t3a_111011(T, dT, H0, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    for i in range(system.num_core_alpha):
        for j in range(system.num_act_occupied_alpha):
            for k in range(j+1, system.num_act_occupied_alpha):
                for a in range(system.num_act_unoccupied_alpha):
                    for b in range(a+1, system.num_act_unoccupied_alpha):
                        for c in range(b+1, system.num_act_unoccupied_alpha):

                            denom = (
                                    H0.a.oo[oa, oa][i, i] + H0.a.oo[Oa, Oa][j, j] + H0.a.oo[Oa, Oa][k, k]
                                   -H0.a.vv[Va, Va][a, a] - H0.a.vv[Va, Va][b, b] - H0.a.vv[Va, Va][c, c]
                            )

                            val = dT.aaa.VVVoOO[a, b, c, i, j, k] / denom

                            T.aaa.VVVoOO[a, b, c, i, j, k] += val
                            T.aaa.VVVoOO[a, b, c, i, k, j] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]

                            T.aaa.VVVoOO[a, c, b, i, j, k] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]
                            T.aaa.VVVoOO[a, c, b, i, k, j] = T.aaa.VVVoOO[a, b, c, i, j, k]

                            T.aaa.VVVoOO[b, c, a, i, j, k] = T.aaa.VVVoOO[a, b, c, i, j, k]
                            T.aaa.VVVoOO[b, c, a, i, k, j] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]

                            T.aaa.VVVoOO[b, a, c, i, j, k] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]
                            T.aaa.VVVoOO[b, a, c, i, k, j] = T.aaa.VVVoOO[a, b, c, i, j, k]

                            T.aaa.VVVoOO[c, a, b, i, j, k] = T.aaa.VVVoOO[a, b, c, i, j, k]
                            T.aaa.VVVoOO[c, a, b, i, k, j] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]

                            T.aaa.VVVoOO[c, b, a, i, j, k] = -1.0 * T.aaa.VVVoOO[a, b, c, i, j, k]
                            T.aaa.VVVoOO[c, b, a, i, k, j] = T.aaa.VVVoOO[a, b, c, i, j, k]

                            dT.aaa.VVVoOO[a, b, c, i, j, k] = val
                            dT.aaa.VVVoOO[a, b, c, i, k, j] = -1.0 * val

                            dT.aaa.VVVoOO[a, c, b, i, j, k] = -1.0 * val
                            dT.aaa.VVVoOO[a, c, b, i, k, j] = val

                            dT.aaa.VVVoOO[b, c, a, i, j, k] = val
                            dT.aaa.VVVoOO[b, c, a, i, k, j] = -1.0 * val

                            dT.aaa.VVVoOO[b, a, c, i, j, k] = -1.0 * val
                            dT.aaa.VVVoOO[b, a, c, i, k, j] = val

                            dT.aaa.VVVoOO[c, a, b, i, j, k] = val
                            dT.aaa.VVVoOO[c, a, b, i, k, j] = -1.0 * val

                            dT.aaa.VVVoOO[c, b, a, i, j, k] = -1.0 * val
                            dT.aaa.VVVoOO[c, b, a, i, k, j] = val

    return T, dT
