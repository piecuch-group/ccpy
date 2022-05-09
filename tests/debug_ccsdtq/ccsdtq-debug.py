import numpy as np

def get_spinorbital_quantities(mf, nelectrons, norbitals, file_dir=None, random=False):

    from integrals import get_integrals_from_pyscf, get_fock
    from hbar import get_ccsdt_intermediates

    o = slice(0, nelectrons)
    v = slice(nelectrons, 2 * norbitals)
    z, g, _, _ = get_integrals_from_pyscf(mf, 0)
    f = get_fock(z, g, o, v)

    if file_dir is not None:
        t1 = np.load(file_dir+"t1.npy")
        t2 = np.load(file_dir+"t2.npy")
        t3 = np.load(file_dir+"t3.npy")
        t4 = np.load(file_dir+"t4.npy")

    if not random:
        from make_t import get_random_t
        t1, t2, t3, t4 = get_random_t(nelectrons, 2 * norbitals - nelectrons)

    
    H1, H2 = get_ccsdt_intermediates(t1, t2, t3, z, g, o, v)

    return t1, t2, t3, t4, H1, H2, f, g, o, v

def get_spinint_quantities(t1, t2, t3, t4, system):

    from ccpy.models.operators import ClusterOperator

    # set up the spin-integrated part
    oa = slice(0, system.nelectrons, 2)
    ob = slice(1, system.nelectrons, 2)
    va = slice(0, 2 * system.norbitals - system.nelectrons, 2)
    vb = slice(1, 2 * system.norbitals - system.nelectrons, 2)

    T = ClusterOperator(system, order=4)
    dT = ClusterOperator(system, order=4)

    setattr(T, 'a', t1[va, oa])
    setattr(T, 'b', t1[vb, ob])
    setattr(T, 'aa', t2[va, va, oa, oa])
    setattr(T, 'ab', t2[va, vb, oa, ob])
    setattr(T, 'bb', t2[vb, vb, ob, ob])
    setattr(T, 'aaa', t3[va, va, va, oa, oa, oa])
    setattr(T, 'aab', t3[va, va, vb, oa, oa, ob])
    setattr(T, 'abb', t3[va, vb, vb, oa, ob, ob])
    setattr(T, 'bbb', t3[vb, vb, vb, ob, ob, ob])
    setattr(T, 'aaaa', t4[va, va, va, va, oa, oa, oa, oa])
    setattr(T, 'aaab', t4[va, va, va, vb, oa, oa, oa, ob])
    setattr(T, 'aabb', t4[va, va, vb, vb, oa, oa, ob, ob])
    setattr(T, 'abbb', t4[va, vb, vb, vb, oa, ob, ob, ob])
    setattr(T, 'bbbb', t4[vb, vb, vb, vb, ob, ob, ob, ob])

    return T, dT, oa, va, ob, vb

def main(file_dir):

    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
    from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates 

    from t4spinorb import calc_t4
    from t4aaaa import calc_t4_aaaa
    from t4aaab import calc_t4_aaab
    from t4aabb import calc_t4_aabb


    #geom = [['H', (-1.000, -1.000/2, 0.000)], 
    #        ['H', (-1.000,  1.000/2, 0.000)], 
    #        ['H', ( 1.000, -1.000/2, 0.000)], 
    #        ['H', ( 1.000,  1.000/2, 0.000)]]
    
    geom = [['H', (0, 1.515263, -1.058898)], 
            ['H', (0, -1.515263, -1.058898)], 
            ['O', (0.0, 0.0, -0.0090)]]

    mol = gto.Mole()

    mol.build(
        atom=geom,
        basis="dz",
        charge=0,
        spin=0,
        symmetry="C1",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H0 = load_pyscf_integrals(mf, nfrozen=0)
    system.print_info()

    t1, t2, t3, t4, H1, H2, f, g, o, v = get_spinorbital_quantities(mf, system.nelectrons, system.norbitals, file_dir=file_dir)

    T, dT, oa, va, ob, vb = get_spinint_quantities(t1, t2, t3, t4, system)
    H = get_ccsd_intermediates(T, H0)
    H = add_VT3_intermediates(T, H)

    print("Calculating spinorbital t4 update...")
    X4, I_vvooov = calc_t4(t1, t2, t3, t4, H1, H2, f, g, o, v)
    print("Calculating spin-integrated t4 update...")
    dT = calc_t4_aaaa(T, dT, H, H0)
    dT = calc_t4_aaab(T, dT, H, H0)
    dT, I3A_vvooov, I3B_vvooov, I3C_ovvvoo, I3D_vvooov, I3B_vovovo, I3C_vovovo, I3B_vovoov, I3C_ovvoov, I3B_vvovoo, I3C_vovvoo  = calc_t4_aabb(T, dT, H, H0)

    # Calculate the error in each projection
    error_aaaa = X4[va, va, va, va, oa, oa, oa, oa] - dT.aaaa
    error_aaab = X4[va, va, va, vb, oa, oa, oa, ob] - dT.aaab
    error_aabb = X4[va, va, vb, vb, oa, oa, ob, ob] - dT.aabb

    err_3a_vvooov = I_vvooov[va, va, oa, oa, oa, va] - I3A_vvooov
    err_3d_vvooov = I_vvooov[vb, vb, ob, ob, ob, vb] - I3D_vvooov

    err_3b_vvooov = I_vvooov[va, va, ob, oa, oa, vb] - I3B_vvooov
    err_3c_ovvvoo = np.transpose(I_vvooov, (2, 1, 0, 5, 4, 3))[oa, vb, vb, va, ob, ob] - I3C_ovvvoo

    err_3b_vovovo = np.transpose(I_vvooov, (0, 2, 1, 3, 5, 4))[va, oa, vb, oa, va, ob] - I3B_vovovo
    err_3c_vovovo = np.transpose(I_vvooov, (1, 2, 0, 4, 5, 3))[va, ob, vb, oa, vb, ob] - I3C_vovovo

    # !!! -> This one still seems to have an error
    err_3b_vovoov = np.transpose(I_vvooov, (0, 2, 1, 3, 4, 5))[va, oa, vb, oa, oa, vb] - I3B_vovoov
    err_3c_vovvoo = np.transpose(I_vvooov, (1, 2, 0, 5, 4, 3))[va, ob, vb, va, ob, ob] - I3C_vovvoo

    err_3b_vvovoo = np.transpose(I_vvooov, (0, 1, 2, 5, 3, 4))[va, va, ob, va, oa, ob] - I3B_vvovoo
    err_3c_ovvoov = np.transpose(I_vvooov, (2, 1, 0, 4, 3, 5))[oa, vb, vb, oa, ob, vb] - I3C_ovvoov

    print("Error in aaaa = ", np.linalg.norm(error_aaaa))
    print("Error in aaab = ", np.linalg.norm(error_aaab))
    print("Error in aabb = ", np.linalg.norm(error_aabb))
    print("   Intermediates:")
    print("   Error in I3A_vvooov = ", np.linalg.norm(err_3a_vvooov))
    print("   Error in I3D_vvooov = ", np.linalg.norm(err_3d_vvooov))
    print("")
    print("   Error in I3B_vvooov = ", np.linalg.norm(err_3b_vvooov))
    print("   Error in I3C_ovvvoo = ", np.linalg.norm(err_3c_ovvvoo))
    print("")
    print("   Error in I3B_vovovo = ", np.linalg.norm(err_3b_vovovo))
    print("   Error in I3C_vovovo = ", np.linalg.norm(err_3c_vovovo))
    print("")
    print("   Error in I3B_vovoov = ", np.linalg.norm(err_3b_vovoov))
    print("   Error in I3C_vovvoo = ", np.linalg.norm(err_3c_vovvoo))
    print("")
    print("   Error in I3B_vvovoo = ", np.linalg.norm(err_3b_vvovoo)) 
    print("   Error in I3C_ovvoov = ", np.linalg.norm(err_3c_ovvoov)) 

if __name__ == "__main__":

    file_dir = "/scratch/gururang/test_CCpy/"

    main(file_dir)



