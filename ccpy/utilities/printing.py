import datetime
import numpy as np


WHITESPACE = "  "

# [TODO]: Make EOMCC printer
# [TODO]: Make left-CC printer


def ccpy_header():

    print("CCpy: Coupled-Cluster Package for Electronic Structure Calculations")
    print(WHITESPACE, "Authors:")
    print(WHITESPACE, " Karthik Gururangan (gururang@msu.edu)")
    print(WHITESPACE, " J. Emiliano Deustua (edeustua@caltech.edu)")
    print(WHITESPACE, " Affiliated with the Piecuch Group at Michigan State University")
    print("\n")


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

class CCPrinter:
    def __init__(self, calculation):
        self.calculation = calculation

    def cc_header(self):
        print(WHITESPACE, "--------------------------------------------------")
        print(
            WHITESPACE, "Calculation type = ", self.calculation.calculation_type.upper()
        )
        print(WHITESPACE, "Maximum iterations =", self.calculation.maximum_iterations)
        print(
            WHITESPACE,
            "Convergence tolerance =",
            self.calculation.convergence_tolerance,
        )
        print(WHITESPACE, "Energy shift =", self.calculation.energy_shift)
        print(WHITESPACE, "DIIS size = ", self.calculation.diis_size)
        print(WHITESPACE, "RHF symmetry =", self.calculation.RHF_symmetry)
        #if self.calculation.active_orders != [None]:
        #    print(WHITESPACE, "Number active occupied =", calculation.)

        print(WHITESPACE, "--------------------------------------------------")
        print("")
        print(
            WHITESPACE,
            "CC calculation started at",
            datetime.datetime.strptime(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M"
            ),
            "\n",
        )

    def eomcc_header(self):
        print(WHITESPACE, "--------------------------------------------------")
        print(
            WHITESPACE, "Calculation type = ", self.calculation.calculation_type.upper()
        )
        print(WHITESPACE, "Maximum iterations =", self.calculation.maximum_iterations)
        print(
            WHITESPACE,
            "Convergence tolerance =",
            self.calculation.convergence_tolerance,
        )
        print(WHITESPACE, "State target multiplicity =", self.calculation.multiplicity)
        print(WHITESPACE, "RHF symmetry =", self.calculation.RHF_symmetry)
        #if self.calculation.active_orders != [None]:
        #    print(WHITESPACE, "Number active occupied =", calculation.)

        print(WHITESPACE, "--------------------------------------------------")
        print("")
        print(
            WHITESPACE,
            "EOMCC calculation started at",
            datetime.datetime.strptime(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M"
            ),
            "\n",
        )

    @staticmethod
    def cc_calculation_summary(reference_energy, cc_energy):
        DATA_FMT = "{:<30} {:>20.8f}"
        print("\n   CC Calculation Summary")
        print("  --------------------------------------------------")
        print(DATA_FMT.format("   Reference energy", reference_energy))
        print(DATA_FMT.format("   CC correlation energy", cc_energy))
        print(DATA_FMT.format("   Total CC energy", reference_energy + cc_energy))
        print(
            "\n   CC calculation ended at",
            datetime.datetime.strptime(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "%Y-%m-%d %H:%M:%S",
            ),
        )
        print("")

    @staticmethod
    def eomcc_calculation_summary(omega, r0, is_converged):
        DATA_FMT = "{:>7} {:<6} {} {:.8f} {:>6} {:.8f}     {}"
        print("\n   EOMCC Calculation Summary")
        print("  --------------------------------------------------")
        for n in range(len(omega)):
            if is_converged[n]:
                convergence_label = 'converged'
            else:
                convergence_label = 'not converged'
            print(DATA_FMT.format("State", n + 1, "ω =", omega[n], "r0 =", r0[n], convergence_label))
        print(
            "\n   EOMCC calculation ended at",
            datetime.datetime.strptime(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "%Y-%m-%d %H:%M:%S",
            ),
        )
        print("")


ITERATION_HEADER_FMT = "{:>21} {:>16} {:>18} {:>19} {:>23}"
ITERATION_FMT = "{:>20} {:>20.10f} {:>20.10f} {:>20.10f} {:>20}"

CC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "δE", "ΔE", "Wall time"
)
EOMCC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "ω", "δω", "Wall time"
)


def print_cc_iteration_header():
    print("\n", CC_ITERATION_HEADER)
    print(len(CC_ITERATION_HEADER) * "-")

def print_eomcc_iteration_header(nroot):
    print("\n", EOMCC_ITERATION_HEADER)
    print(len(EOMCC_ITERATION_HEADER) * "-")


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

    R2 = R.flatten()[n1a + n1b :]
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
    return