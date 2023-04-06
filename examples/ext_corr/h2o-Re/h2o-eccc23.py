from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_pspace_from_cipsi


if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "ndet_1000000/civecs.dat"

    mycc = Driver(system, H)
    mycc.run_eccc(method="eccc2", external_wavefunction=civecs)
    mycc.run_hbar(method="ccsd")
    mycc.run_leftcc(method="left_ccsd")

    pspace, excitations, excitation_count = get_pspace_from_cipsi(civecs, system, nexcit=3, ordered_index=False)

    mycc.run_ccp3(method="ccp3", pspace=pspace[0])


