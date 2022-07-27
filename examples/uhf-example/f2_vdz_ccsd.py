
def main(re):

    from pyscf import gto, scf, cc


    mol = gto.Mole()

    Re = 2.66816 # a.u.

    mol.build(
        atom=[['F', (0, 0, -0.5 * re * Re)], ['F', (0, 0, 0.5 * re * Re)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit='Bohr',
    )

    print("RHF reference")
    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf).run()
    print("CCSD total energy", mycc.e_tot)
    et = mycc.ccsd_t()
    print("CCSD(T) total energy", mycc.e_tot + et)

    print("UHF reference")
    mf = scf.UHF(mol)
    mf.init_guess_breaksym = True
    print(mf.init_guess_breaksym)
    mf.kernel()

    mycc = cc.UCCSD(mf).run()
    print("UCCSD total energy", mycc.e_tot)
    et = mycc.uccsd_t()
    print("UCCSD(T) total energy", mycc.e_tot + et)

if __name__ == "__main__":

    re = 3.0

    main(re)
