
from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver, lcc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt

from ccpy.eomcc.initial_guess import get_initial_guess

def print_amplitudes(operator, threshold=0.2):

    for key, value in operator.__dict__:

        if key not in ['a', 'b', 'aa', 'ab', 'bb', 'aaa', 'aab', 'abb', 'bbb']: continue

        print('')
        if len(value.shape) == 2:
            for a in range(value.shape[0]):
                for i in range(value.shape[1]):
                    if abs(value[a,i]) > threshold:
                        print('{}({},{}) = {}'.format(key,a+1,i+1,value[a,i]))
        if len(value.shape) == 4:
            for a in range(value.shape[0]):
                for b in range(value.shape[1]):
                    for i in range(value.shape[2]):
                        for j in range(value.shape[3]):
                            if abs(value[a,b,i,j]) > threshold:
                                print('{}({},{},{},{}) = {}'.format(key,a+1,b+1,i+1,j+1,value[a,b,i,j]))


if __name__ == "__main__":

    system, H = load_from_gamess(
            "hf_ccsd_6-31g_ae.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=3,
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H)
    print_amplitudes(T)


    Hbar = build_hbar_ccsdt(T, H)

    calculation = Calculation(
        order=3,
        calculation_type="left_ccsdt",
        convergence_tolerance=1.0e-08,
        maximum_iterations=200,
        RHF_symmetry=True,
    )

    L, _, _ = lcc_driver(calculation, system, T, Hbar)
    print_amplitudes(L)
