import numpy as np
from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.eomcc.initial_guess import get_initial_guess

def run_energy(r):


    mol = gto.Mole()

    mol.build(
        atom=[['Li', (0, 0, -0.5 * r)],
              ['H', (0, 0, 0.5 * r)]],
        basis="ccpcvtz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit='Bohr',
    )
    mf = scf.RHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=0)

    calculation = Calculation(
        calculation_type="ccsdt",
        RHF_symmetry=True,
        energy_shift=0.8
    )

    T, total_energy, converged_cc = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsdt(T, H)

    calculation = Calculation(
        calculation_type="eomccsdt",
        maximum_iterations=80,
        convergence_tolerance=1.0e-07,
        multiplicity=1,
        RHF_symmetry=True,
        low_memory=False,
    )

    R, omega = get_initial_guess(calculation, system, Hbar, 1, noact=0, nuact=0, guess_order=1)

    R, omega, r0, converged_eomcc = eomcc_driver(calculation, system, Hbar, T, R, omega)

    return T, R[0], total_energy, omega[0], converged_cc, converged_eomcc[0]

if __name__ == "__main__":

    R = np.arange(0.5, 15.0, 0.1)

    energy = np.zeros(len(R))
    omega = np.zeros(len(R))
    convg_ground = np.zeros(len(R))
    convg_excited = np.zeros(len(R))
    for i, r in enumerate(R):

        _, _, energy[i], omega[i], convg_ground[i], convg_excited[i] = run_energy(r)

    np.save("ccsdt_ccpcvtz.npy", energy)
    np.save("ccsdt_ccpcvtz_convg.npy", convg_ground)
    np.save("eomccsdt_ccpcvtz.npy", omega)
    np.save("eomccsdt_ccpcvtz_convg.npy", convg_excited)
