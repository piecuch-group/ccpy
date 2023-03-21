
from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

if __name__ == "__main__":

    # singlet geometry
    tmm_singlet = """
            C  0.0000000000        0.0000000000       -2.5866403780
            C  0.0000000000        0.0000000000       -0.0574311160
            C  0.0000000000       -2.3554018654        1.3522478168
            C  0.0000000000        2.3554018654        1.3522478168
            H  0.0000000000        1.7421982357       -3.6398423076
            H  0.0000000000       -1.7421982357       -3.6398423076
            H  0.0000000000        2.3674499646        3.3825336644
            H  0.0000000000       -2.3674499646        3.3825336644
            H  0.0000000000        4.1366308985        0.3778613882
            H  0.0000000000       -4.1366308985        0.3778613882
    """
    tmm_singlet1 = """
            C  0.0000000000        0.0000000000       -2.5866403780
            C  0.0000000000        0.0000000000       -0.0574311160
            C  0.0000000000       -2.3554018654        1.3522478168
            C  0.0000000000        2.3554018654        1.3522478168
            H  0.0000000000        1.7421982357       -3.6398423076
            H  0.0000000000       -1.7421982357       -3.6398423076
            H  0.0000000000        2.3674499646        3.3825336644
            H  0.0000000000       -2.3674499646        3.3825336644
            H  0.0000000000        4.1366308985        0.3778613882
            H  0.0000000000       -4.2366308985        0.4778613882
    """
    mol = gto.M(
        atom=tmm_singlet1,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="Cs",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.chkfile = "tmm.chk"
    mf.diis_file = "tmm_diis.h5"
    mf.kernel()

    mol = gto.M(
        atom=tmm_singlet,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.verbose = 4

    #mf = scf.RHF(mol)
    #dm = mf.from_chk('tmm.chk')
    #mf.kernel(dm)

    mo_coeff = scf.chkfile.load('tmm.chk', 'scf/mo_coeff')
    mo_occ = scf.chkfile.load('tmm.chk', 'scf/mo_occ')
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    mf.kernel(dm)

    print("Expected result from GAMESS = ", -154.7517626266)
