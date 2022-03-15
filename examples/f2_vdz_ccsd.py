from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.crcc23 import calc_crcc23

case = 'F2'

mol = gto.Mole()

if case == 'F2':
    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

if case == 'H2O':
    mol.build(
        atom="""H 0.0 -1.515263  -1.058898
                H 0.0 1.515263  -1.058898
                O 0.0 0.0 -0.0090""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit='Bohr',
    )
    mf = scf.RHF(mol)
    mf.kernel()

system, H = load_pyscf_integrals(mf, nfrozen=0,
                                 num_act_holes_alpha=5, num_act_particles_alpha=9,
                                 num_act_holes_beta=5, num_act_particles_beta=9,
                                 )
system.print_info()

calculation = Calculation(
    order=3,
    calculation_type="ccsdt",
    active_orders=[None],
    num_active=[None],
    convergence_tolerance=1.0e-08
)

T, total_energy, is_converged = cc_driver(calculation, system, H)

#calculation = Calculation(
#    order=2,
#    calculation_type="left_ccsd",
#    convergence_tolerance=1.0e-08
#)

#Hbar = build_hbar_ccsd(T, H)

#L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

#Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=False)








