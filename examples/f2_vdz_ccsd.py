from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.cc import driver

mol = gto.Mole()
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


system, H = load_pyscf_integrals(mf, nfrozen=2)
system.print_info()

calculation = Calculation(
    calculation_type="ccsd",
)

T, total_energy, is_converged = driver(calculation, system, H)






