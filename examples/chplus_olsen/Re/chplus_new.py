
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import Driver

if __name__ == "__main__":

    system, H = load_from_gamess(
            "/scratch/gururang/test_CCpy/chplus-1.0-olsen/chplus_re.log",
            "/scratch/gururang/test_CCpy/chplus-1.0-olsen/onebody.inp",
            "/scratch/gururang/test_CCpy/chplus-1.0-olsen/twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    mycc = Driver(system, H)
    mycc.options["maximum_iterations"] = 200

    mycc.run_cc(method="ccsd")
    mycc.run_hbar(method="ccsd")
    mycc.run_eomcc(method="eomccsd", state_index=[1, 2, 3])
    mycc.run_leftcc(method="left_ccsd", state_index=[0, 1, 2, 3])
    mycc.run_ccp3(method="crcc23", state_index=[0, 1, 2, 3])

