from pyscf import gto, scf

from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

mol = gto.Mole()
mol.build(
    atom="""F 0.0 0.0 -2.66816
            F 0.0 0.0  2.66816""",
    basis="ccpvdz",
    charge=1,
    spin=1,
    symmetry="D2H",
    cart=True,
    unit='Bohr',
)
mf = scf.ROHF(mol)
mf.kernel()

nfrozen = 2
system, H = load_pyscf_integrals(mf, nfrozen)

print(system)
