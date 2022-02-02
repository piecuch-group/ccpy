import datetime

from art import tprint

CCPY_TITLE = """
   _____    _____     __ __     __  __  
  /\ __/\  /\ __/\  /_/\__/\  /\  /\  /\
  ) )__\/  ) )__\/  ) ) ) ) ) \ \ \/ / /
  / / /    / / /    /_/ /_/ /   \ \__/ / 
  \ \ \_   \ \ \_   \ \ \_\/     \__/ /  
   ) )__/\  ) )__/\  )_) )       / / /   
   \/___\/  \/___\/  \_\/        \/_/
"""

WHITESPACE = "  "


def ccpy_header():
    # twisted, sub-zero, swampland, starwars
    tprint("   ccpy", "twisted")
    # print(whitespace, ccpy_title)
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
                    self.system.orbital_symmetries[i],
                    self.system.mo_occupation[i],
                )
            )
        print("")
        print(WHITESPACE, "Nuclear Repulsion Energy =", self.system.nuclear_repulsion)
        print(WHITESPACE, "Reference Energy =", self.system.reference_energy)
        print("")


class CCPrinter:
    def __init__(self, calculation):
        self.calculation = calculation

    def header(self):
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

    @staticmethod
    def calculation_summary(reference_energy, cc_energy):
        DATA_FMT = "{:<30} {:>20.8f}"
        print("\nCC Calculation Summary")
        print("----------------------------------------")
        print(DATA_FMT.format("Reference energy", reference_energy))
        print(DATA_FMT.format("CC correlation energy", cc_energy))
        print(DATA_FMT.format("Total CC energy", reference_energy + cc_energy))
        print(
            "\nCC calculation ended at",
            datetime.datetime.strptime(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "%Y-%m-%d %H:%M:%S",
            ),
        )


ITERATION_HEADER_FMT = "{:>20} {:>20} {:>20} {:>20} {:>20}"
ITERATION_FMT = "{:>20} {:>20.8f} {:>20.8f} {:>20.8f} {:>20}"
ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "δE", "ΔE", "Wall time"
)


def print_iteration_header():
    print("\n", ITERATION_HEADER)
    print(len(ITERATION_HEADER) * "-")


def print_iteration(
    iteration_idx, residuum, delta_energy, correlation_energy, elapsed_time
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    print(
        ITERATION_FMT.format(
            iteration_idx, residuum, delta_energy, correlation_energy, time_str
        )
    )
