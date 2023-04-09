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

def cc_calculation_summary(T, reference_energy, cc_energy, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    print("\n   CC Calculation Summary")
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Reference energy", reference_energy))
    print(DATA_FMT.format("   CC correlation energy", cc_energy))
    print(DATA_FMT.format("   Total CC energy", reference_energy + cc_energy))
    print_ee_amplitudes(T, system, T.order, print_thresh)
    print("")

def eomcc_calculation_summary(R, omega, r0, is_converged, system, print_thresh):
    print("\n   EOMCC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   ω = %.8f" % omega, "  r0 = %.8f" % r0)
    print_ee_amplitudes(R, system, R.order, print_thresh)
    print("")

def leftcc_calculation_summary(L, omega, LR, is_converged, system, print_thresh):
    print("\n   Left CC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   ω = %.8f" % omega, "  LR = %.8f" % LR)
    print_ee_amplitudes(L, system, L.order, print_thresh)
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

def print_ee_amplitudes(R, system, order, thresh_print):

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta
    n2a = system.noccupied_alpha ** 2 * system.nunoccupied_alpha ** 2
    n2b = system.noccupied_beta * system.noccupied_alpha * system.nunoccupied_beta * system.nunoccupied_alpha
    n2c = system.noccupied_beta ** 2 * system.nunoccupied_beta ** 2

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

    print("\n   Largest Singly and Doubly Excited Amplitudes:")
    n = 1
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            if abs(R.a[a, i]) <= thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}A   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_alpha + 1,
                    R.a[a, i],
                )
            )
            n += 1
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            if abs(R.b[a, i]) <= thresh_print: continue
            print(
                "      [{}]     {}B  ->  {}B   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_beta + 1,
                    R.b[a, i],
                )
            )
            n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    if abs(R.aa[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}A  {}A  ->  {}A  {}A  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_alpha + system.nfrozen + 1,
                            b + system.noccupied_alpha + system.nfrozen + 1,
                            R.aa[a, b, i, j],
                        )
                    )
                    n += 1
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    if abs(R.bb[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}B  {}B  ->  {}B  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_beta + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.bb[a, b, i, j],
                        )
                    )
                    n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    if abs(R.ab[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}A  {}B  ->  {}A  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_alpha + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.ab[a, b, i, j],
                        )
                    )
                    n += 1
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
