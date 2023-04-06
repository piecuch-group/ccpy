import datetime
import numpy as np

WHITESPACE = "  "

ITERATION_HEADER_FMT = "{:>10} {:>12} {:>14} {:>17} {:>19}"
ITERATION_FMT = "{:>8} {:>17.10f} {:>17.10f} {:>17.10f} {:>15}"

CC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "δE", "ΔE", "Wall time"
)
EOMCC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "ω", "δω", "Wall time"
)

def get_timestamp():
        return datetime.datetime.strptime(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

def cc_calculation_summary(reference_energy, cc_energy):
    DATA_FMT = "{:<30} {:>20.8f}"
    print("\n   CC Calculation Summary")
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Reference energy", reference_energy))
    print(DATA_FMT.format("   CC correlation energy", cc_energy))
    print(DATA_FMT.format("   Total CC energy", reference_energy + cc_energy))
    print("")

def eomcc_calculation_summary(R, omega, r0, is_converged, system):
    print("\n   EOMCC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   omega = %.8f" % omega, "  r0 = %.8f" % r0)
    print_ee_amplitudes(R, system, R.order, 10, 0.05)
    print("")

def leftcc_calculation_summary(L, omega, LR, is_converged, system):
    print("\n   Left CC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   omega = %.8f" % omega, "  LR = %.8f" % LR)
    print_ee_amplitudes(L, system, L.order, 10, 0.05)
    print("")

def print_cc_iteration_header():
    print("\n", CC_ITERATION_HEADER)
    print('    '+(len(CC_ITERATION_HEADER)) * "-")

def print_eomcc_iteration_header():
    print("\n", EOMCC_ITERATION_HEADER)
    print('    '+(len(EOMCC_ITERATION_HEADER)) * "-")

def print_cc_iteration(
    iteration_idx, residuum, delta_energy, correlation_energy, elapsed_time
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    print(
        ITERATION_FMT.format(
            iteration_idx, residuum, delta_energy, correlation_energy, time_str
        )
    )

def print_eomcc_iteration(
    iteration_idx, omega, residuum, delta_energy, elapsed_time
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    print(
        ITERATION_FMT.format(
            iteration_idx, residuum, omega, delta_energy, time_str
        )
    )

def print_amplitudes(R, system, order, nprint=10, thresh_print=1.0e-01):

    from ccpy.models.operators import ClusterOperator, FockOperator

    if isinstance(R, ClusterOperator):
        print_ee_amplitudes(R, system, order, nprint, thresh_print)
    if isinstance(R, FockOperator):
        delta_particles = R.num_particles - R.num_holes
        if delta_particles == 1:
            print_ea_amplitudes(R, system, order, nprint, thresh_print)
        elif delta_particles == -1:
            print_ip_amplitudes(R, system, order, nprint, thresh_print)

def print_ee_amplitudes(R, system, order, nprint, thresh_print):

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta
    n2a = system.noccupied_alpha ** 2 * system.nunoccupied_alpha ** 2
    n2b = system.noccupied_beta * system.noccupied_alpha * system.nunoccupied_beta * system.nunoccupied_alpha
    n2c = system.noccupied_beta ** 2 * system.nunoccupied_beta ** 2

    R1 = R.flatten()[: n1a + n1b]
    idx = np.flip(np.argsort(abs(R1)))
    print("     Largest Singly Excited Amplitudes:")
    for n in range(nprint):
        if idx[n] < n1a:
            a, i = np.unravel_index(idx[n], R.a.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}A     {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_alpha + 1,
                    R1[idx[n]],
                )
            )
        else:
            a, i = np.unravel_index(idx[n] - n1a, R.b.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  ->  {}B     {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    a + system.noccupied_beta + system.nfrozen + 1,
                    R1[idx[n]],
                )
            )
    if order < 2: return

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.aa[b, a, j, i] = 0.0
                    R.aa[a, b, j, i] = 0.0
                    R.aa[b, a, i, j] = 0.0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    R.bb[b, a, j, i] = 0.0
                    R.bb[a, b, j, i] = 0.0
                    R.bb[b, a, i, j] = 0.0

    R2 = R.flatten()[n1a + n1b : n1a + n1b + n2a + n2b + n2c]
    idx = np.flip(np.argsort(abs(R2)))
    print("     Largest Doubly Excited Amplitudes:")
    for n in range(nprint):
        if idx[n] < n2a:
            a, b, i, j = np.unravel_index(idx[n], R.aa.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  {}A  ->  {}A  {}A    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    a + system.noccupied_alpha + system.nfrozen + 1,
                    b + system.noccupied_alpha + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        elif idx[n] < n2a + n2b:
            a, b, i, j = np.unravel_index(idx[n] - n2a, R.ab.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  {}B  ->  {}A  {}B    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    a + system.noccupied_alpha + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        else:
            a, b, i, j = np.unravel_index(idx[n] - n2a - n2b, R.bb.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  {}B  ->  {}B  {}B    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    a + system.noccupied_beta + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )

    # Restore permutationally redundant amplitudes
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.aa[b, a, j, i] = R.aa[a, b, i, j]
                    R.aa[a, b, j, i] = -1.0 * R.aa[a, b, i, j]
                    R.aa[b, a, i, j] = -1.0 * R.aa[a, b, i, j]
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    R.bb[b, a, j, i] = R.bb[a, b, i, j]
                    R.bb[a, b, j, i] = -1.0 * R.bb[a, b, i, j]
                    R.bb[b, a, i, j] = -1.0 * R.bb[a, b, i, j]

    return


def print_ip_amplitudes(R, system, order, nprint, thresh_print):

    n1a = system.noccupied_alpha
    n1b = system.noccupied_beta
    n2a = system.noccupied_alpha ** 2 * system.nunoccupied_alpha
    n2b = system.noccupied_beta * system.noccupied_alpha * system.nunoccupied_alpha
    n2c = system.noccupied_beta * system.nunoccupied_beta * system.noccupied_alpha
    n2d = system.noccupied_beta ** 2 * system.nunoccupied_beta

    R1 = R.flatten()[: n1a + n1b]
    idx = np.flip(np.argsort(abs(R1)))
    print("     Largest 1h Amplitudes:")
    for n in range(nprint):
        if idx[n] < n1a:
            i, = np.unravel_index(idx[n], R.a.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  ->     {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    R1[idx[n]],
                )
            )
        else:
            i, = np.unravel_index(idx[n] - n1a, R.b.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  ->     {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    R1[idx[n]],
                )
            )
    if order < 2: return

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for b in range(system.nunoccupied_alpha):
        for j in range(system.noccupied_alpha):
            for i in range(j + 1, system.noccupied_alpha):
                R.aa[b, i, j] = 0.0
    for b in range(system.nunoccupied_beta):
        for j in range(system.noccupied_beta):
            for i in range(j + 1, system.noccupied_beta):
                R.bb[b, i, j] = 0.0

    R2 = R.flatten()[n1a + n1b: n1a + n1b + n2a + n2b + n2c + n2d]
    idx = np.flip(np.argsort(abs(R2)))
    print("     Largest 2h-1p Excited Amplitudes:")
    for n in range(nprint):
        if idx[n] < n2a:
            b, j, i = np.unravel_index(idx[n], R.aa.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  {}A  ->  {}A    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_alpha + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        elif idx[n] < n2a + n2b:
            b, j, i = np.unravel_index(idx[n] - n2a, R.ab.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  {}A  ->  {}A    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        elif idx[n] < n2a + n2b + n2c:
            b, j, i = np.unravel_index(idx[n] - n2a - n2b, R.ba.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  {}B  ->  {}B    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        else:
            b, j, i = np.unravel_index(idx[n] - n2a - n2b - n2c, R.bb.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  {}B  ->  {}B    {:.6f}".format(
                    n + 1,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )

    # Restore permutationally redundant terms
    for b in range(system.nunoccupied_alpha):
        for j in range(system.noccupied_alpha):
            for i in range(j + 1, system.noccupied_alpha):
                R.aa[b, i, j] = -1.0 * R.aa[b, j, i]
    for b in range(system.nunoccupied_beta):
        for j in range(system.noccupied_beta):
            for i in range(j + 1, system.noccupied_beta):
                R.bb[b, i, j] = -1.0 * R.bb[b, j, i]

    return

def print_ea_amplitudes(R, system, order, nprint, thresh_print):

    n1a = system.nunoccupied_alpha
    n1b = system.nunoccupied_beta
    n2a = system.noccupied_alpha * system.nunoccupied_alpha ** 2
    n2b = system.nunoccupied_beta * system.noccupied_alpha * system.nunoccupied_alpha
    n2c = system.noccupied_beta * system.nunoccupied_beta * system.nunoccupied_alpha
    n2d = system.noccupied_beta * system.nunoccupied_beta ** 2

    R1 = R.flatten()[: n1a + n1b]
    idx = np.flip(np.argsort(abs(R1)))
    print("     Largest 1p Amplitudes:")
    for n in range(nprint):
        if idx[n] < n1a:
            a, = np.unravel_index(idx[n], R.a.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     ->  {}A     {:.6f}".format(
                    n + 1,
                    a + system.noccupied_alpha + system.nfrozen + 1,
                    R1[idx[n]],
                )
            )
        else:
            a, = np.unravel_index(idx[n] - n1a, R.b.shape, order="C")
            if abs(R1[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     ->  {}B     {:.6f}".format(
                    n + 1,
                    a + system.noccupied_beta + system.nfrozen + 1,
                    R1[idx[n]],
                )
            )
    if order < 2: return

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for b in range(system.nunoccupied_alpha):
        for j in range(system.noccupied_alpha):
            for a in range(b + 1, system.nunoccupied_alpha):
                R.aa[a, j, b] = 0.0
    for b in range(system.nunoccupied_beta):
        for j in range(system.noccupied_beta):
            for a in range(b + 1, system.nunoccupied_beta):
                R.bb[a, j, b] = 0.0

    R2 = R.flatten()[n1a + n1b: n1a + n1b + n2a + n2b + n2c + n2d]
    idx = np.flip(np.argsort(abs(R2)))
    print("     Largest 2p-1h Excited Amplitudes:")
    for n in range(nprint):
        if idx[n] < n2a:
            b, j, a = np.unravel_index(idx[n], R.aa.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}A  {}A    {:.6f}".format(
                    n + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_alpha + system.nfrozen + 1,
                    a + system.noccupied_alpha + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        elif idx[n] < n2a + n2b:
            b, j, a = np.unravel_index(idx[n] - n2a, R.ab.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}A  {}B    {:.6f}".format(
                    n + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_alpha + system.nfrozen + 1,
                    a + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        elif idx[n] < n2a + n2b + n2c:
            b, j, a = np.unravel_index(idx[n] - n2a - n2b, R.ba.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  ->  {}B  {}A    {:.6f}".format(
                    n + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    a + system.noccupied_alpha + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )
        else:
            b, j, a = np.unravel_index(idx[n] - n2a - n2b - n2c, R.aa.shape, order="C")
            if abs(R2[idx[n]]) < thresh_print: continue
            print(
                "      [{}]     {}B  ->  {}B  {}B    {:.6f}".format(
                    n + 1,
                    j + system.nfrozen + 1,
                    b + system.noccupied_beta + system.nfrozen + 1,
                    a + system.noccupied_beta + system.nfrozen + 1,
                    R2[idx[n]],
                )
            )

    # Restore permutationally redundant terms
    for b in range(system.nunoccupied_alpha):
        for j in range(system.noccupied_alpha):
            for a in range(b + 1, system.nunoccupied_alpha):
                R.aa[a, j, b] = -1.0 * R.aa[b, j, a]
    for b in range(system.nunoccupied_beta):
        for j in range(system.noccupied_beta):
            for a in range(b + 1, system.nunoccupied_beta):
                R.bb[a, j, b] = -1.0 * R.bb[b, j, a]

    return


class SystemPrinter:
    def __init__(self, system):
        self.system = system

    def header(self):

        print(WHITESPACE, "System Information:")
        print(WHITESPACE, "----------------------------------------------------")
        print(WHITESPACE, "  Number of correlated electrons =", self.system.nelectrons)
        print(WHITESPACE, "  Number of correlated orbitals =", self.system.norbitals)
        print(WHITESPACE, "  Number of frozen orbitals =", self.system.nfrozen)
        print(
            WHITESPACE,
            "  Number of alpha occupied orbitals =",
            self.system.noccupied_alpha,
        )
        print(
            WHITESPACE,
            "  Number of alpha unoccupied orbitals =",
            self.system.nunoccupied_alpha,
        )
        print(
            WHITESPACE,
            "  Number of beta occupied orbitals =",
            self.system.noccupied_beta,
        )
        print(
            WHITESPACE,
            "  Number of beta unoccupied orbitals =",
            self.system.nunoccupied_beta,
        )
        print(WHITESPACE, "  Charge =", self.system.charge)
        print(WHITESPACE, "  Point group =", self.system.point_group)
        print(WHITESPACE, "  Symmetry of reference =", self.system.reference_symmetry)
        print(
            WHITESPACE, "  Spin multiplicity of reference =", self.system.multiplicity
        )
        print("")

        HEADER_FMT = "{:>10} {:>20} {:>13} {:>13}"
        MO_FMT = "{:>10} {:>20.6f} {:>13} {:>13.1f}"

        header = HEADER_FMT.format("MO #", "Energy (a.u.)", "Symmetry", "Occupation")
        print(header)
        print(len(header) * "-")
        for i in range(self.system.norbitals + self.system.nfrozen):
            print(
                MO_FMT.format(
                    i + 1,
                    self.system.mo_energies[i],
                    self.system.orbital_symmetries_all[i],
                    self.system.mo_occupation[i],
                )
            )
        print("")
        print(WHITESPACE, "Nuclear Repulsion Energy =", self.system.nuclear_repulsion)
        print(WHITESPACE, "Reference Energy =", self.system.reference_energy)
        print("")
        return
# def print_amplitudes(R, system, order, nprint=10, thresh_print=1.0e-01):
#
#     from ccpy.models.operators import ClusterOperator, FockOperator
#
#     n1a = system.noccupied_alpha * system.nunoccupied_alpha
#     n1b = system.noccupied_beta * system.nunoccupied_beta
#     n2a = system.noccupied_alpha ** 2 * system.nunoccupied_alpha ** 2
#     n2b = system.noccupied_beta * system.noccupied_alpha * system.nunoccupied_beta * system.nunoccupied_alpha
#     n2c = system.noccupied_beta ** 2 * system.nunoccupied_beta ** 2
#
#     R1 = R.flatten()[: n1a + n1b]
#     idx = np.flip(np.argsort(abs(R1)))
#     print("     Largest Singly Excited Amplitudes:")
#     for n in range(nprint):
#         if idx[n] < n1a:
#             a, i = np.unravel_index(idx[n], R.a.shape, order="C")
#             if abs(R1[idx[n]]) < thresh_print: continue
#             print(
#                 "      [{}]     {}A  ->  {}A     {:.6f}".format(
#                     n + 1,
#                     i + system.nfrozen + 1,
#                     a + system.nfrozen + system.noccupied_alpha + 1,
#                     R1[idx[n]],
#                 )
#             )
#         else:
#             a, i = np.unravel_index(idx[n] - n1a, R.b.shape, order="C")
#             if abs(R1[idx[n]]) < thresh_print: continue
#             print(
#                 "      [{}]     {}B  ->  {}B     {:.6f}".format(
#                     n + 1,
#                     i + system.nfrozen + 1,
#                     a + system.noccupied_beta + system.nfrozen + 1,
#                     R1[idx[n]],
#                 )
#             )
#     if order < 2: return
#
#     # Zero out the non-unique R amplitudes related by permutational symmetry
#     for a in range(system.nunoccupied_alpha):
#         for b in range(a + 1, system.nunoccupied_alpha):
#             for i in range(system.noccupied_alpha):
#                 for j in range(i + 1, system.noccupied_alpha):
#                     R.aa[b, a, j, i] = 0.0
#                     R.aa[a, b, j, i] = 0.0
#                     R.aa[b, a, i, j] = 0.0
#     for a in range(system.nunoccupied_beta):
#         for b in range(a + 1, system.nunoccupied_beta):
#             for i in range(system.noccupied_beta):
#                 for j in range(i + 1, system.noccupied_beta):
#                     R.bb[b, a, j, i] = 0.0
#                     R.bb[a, b, j, i] = 0.0
#                     R.bb[b, a, i, j] = 0.0
#
#     R2 = R.flatten()[n1a + n1b : n1a + n1b + n2a + n2b + n2c]
#     idx = np.flip(np.argsort(abs(R2)))
#     print("     Largest Doubly Excited Amplitudes:")
#     for n in range(nprint):
#         if idx[n] < n2a:
#             a, b, i, j = np.unravel_index(idx[n], R.aa.shape, order="C")
#             if abs(R2[idx[n]]) < thresh_print: continue
#             print(
#                 "      [{}]     {}A  {}A  ->  {}A  {}A    {:.6f}".format(
#                     n + 1,
#                     i + system.nfrozen + 1,
#                     j + system.nfrozen + 1,
#                     a + system.noccupied_alpha + system.nfrozen + 1,
#                     b + system.noccupied_alpha + system.nfrozen + 1,
#                     R2[idx[n]],
#                 )
#             )
#         elif idx[n] < n2a + n2b:
#             a, b, i, j = np.unravel_index(idx[n] - n2a, R.ab.shape, order="C")
#             if abs(R2[idx[n]]) < thresh_print: continue
#             print(
#                 "      [{}]     {}A  {}B  ->  {}A  {}B    {:.6f}".format(
#                     n + 1,
#                     i + system.nfrozen + 1,
#                     j + system.nfrozen + 1,
#                     a + system.noccupied_alpha + system.nfrozen + 1,
#                     b + system.noccupied_beta + system.nfrozen + 1,
#                     R2[idx[n]],
#                 )
#             )
#         else:
#             a, b, i, j = np.unravel_index(idx[n] - n2a - n2b, R.bb.shape, order="C")
#             if abs(R2[idx[n]]) < thresh_print: continue
#             print(
#                 "      [{}]     {}B  {}B  ->  {}B  {}B    {:.6f}".format(
#                     n + 1,
#                     i + system.nfrozen + 1,
#                     j + system.nfrozen + 1,
#                     a + system.noccupied_beta + system.nfrozen + 1,
#                     b + system.noccupied_beta + system.nfrozen + 1,
#                     R2[idx[n]],
#                 )
#             )
#     return
