"""Functions to calculate the ground-state CR-CC(2,4) quadruples correction to CCSD."""
import time
import numpy as np

from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import crcc24_opt_loops

def calc_crcc24(T, L, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CR-CC(2,4) correction to the CCSD energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    
    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaaa correction ####
    t1 = time.perf_counter()
    dA_aaaa, dB_aaaa, dC_aaaa, dD_aaaa = aaaa_correction(T, L, H, H0, d3aaa_o, d3aaa_v)
    # dA_aaaa, dB_aaaa, dC_aaaa, dD_aaaa = crcc24_opt_loops.crcc24a_opt(
    #     T.aa,
    #     L.aa,
    #     H0.a.oo,
    #     H0.a.vv,
    #     H.a.oo,
    #     H.a.vv,
    #     H.aa.voov,
    #     H.aa.oooo,
    #     H.aa.vvvv,
    #     H.aa.oovv,
    #     d3aaa_o,
    #     d3aaa_v,
    # )

    #### aaab correction ####
    t1 = time.perf_counter()
    dA_aaab, dB_aaab, dC_aaab, dD_aaab = aaab_correction(T, L, H, H0, d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v)
    # dA_aaab, dB_aaab, dC_aaab, dD_aaab = crcc24_loops.crcc24b(
    #     T.aa,
    #     T.ab,
    #     L.aa,
    #     L.ab,
    #     H0.a.oo,
    #     H0.a.vv,
    #     H0.b.oo,
    #     H0.b.vv,
    #     H.a.oo,
    #     H.a.vv,
    #     H.b.oo,
    #     H.b.vv,
    #     H.aa.voov,
    #     H.aa.oooo,
    #     H.aa.vvvv,
    #     H.aa.oovv,
    #     H.ab.voov,
    #     H.ab.ovov,
    #     H.ab.vovo,
    #     H.ab.ovvo,
    #     H.ab.oooo,
    #     H.ab.vvvv,
    #     H.ab.oovv,
    #     H.bb.voov,
    #     d3aaa_o,
    #     d3aaa_v,
    #     d3aab_o,
    #     d3aab_v,
    #     d3abb_o,
    #     d3abb_v,
    # )

    #### aabb correction ####
    t1 = time.perf_counter()
    dA_aabb, dB_aabb, dC_aabb, dD_aabb = aabb_correction(T, L, H, H0, d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v)
    # dA_aabb, dB_aabb, dC_aabb, dD_aabb = crcc24_opt_loops.crcc24c_opt(
    #     T.aa,
    #     T.ab,
    #     T.bb,
    #     L.aa,
    #     L.ab,
    #     L.bb,
    #     H0.a.oo,
    #     H0.a.vv,
    #     H0.b.oo,
    #     H0.b.vv,
    #     H.a.oo,
    #     H.a.vv,
    #     H.b.oo,
    #     H.b.vv,
    #     H.aa.voov,
    #     H.aa.oooo,
    #     H.aa.vvvv,
    #     H.aa.oovv,
    #     H.ab.voov,
    #     H.ab.ovov,
    #     H.ab.vovo,
    #     H.ab.ovvo,
    #     H.ab.oooo,
    #     H.ab.vvvv,
    #     H.ab.oovv,
    #     H.bb.voov,
    #     H.bb.oooo,
    #     H.bb.vvvv,
    #     H.bb.oovv,
    #     d3aaa_o,
    #     d3aaa_v,
    #     d3aab_o,
    #     d3aab_v,
    #     d3abb_o,
    #     d3abb_v,
    #     d3bbb_o,
    #     d3bbb_v,
    # )

    if use_RHF:
        correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
        correction_B = 2.0 * dB_aaaa + 2.0 * dB_aaab + dB_aabb
        correction_C = 2.0 * dC_aaaa + 2.0 * dC_aaab + dC_aabb
        correction_D = 2.0 * dD_aaaa + 2.0 * dD_aaab + dD_aabb

    else:
        #### abbb correction ####
        t1 = time.perf_counter()
        dA_abbb, dB_abbb, dC_abbb, dD_abbb = abbb_correction(T, L, H, H0, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v)
        #dA_abbb, dB_abbb, dC_abbb, dD_abbb = crcc24_opt_loops.crcc24d_opt(
        #    T.ab,
        #    T.bb,
        #    L.ab,
        #    L.bb,
        #    H0.a.oo,
        #    H0.a.vv,
        #    H0.b.oo,
        #    H0.b.vv,
        #    H.a.oo,
        #    H.a.vv,
        #    H.b.oo,
        #    H.b.vv,
        #    H.aa.voov,
        #    H.ab.voov,
        #    H.ab.ovov,
        #    H.ab.vovo,
        #    H.ab.ovvo,
        #    H.ab.oooo,
        #    H.ab.vvvv,
        #    H.ab.oovv,
        #    H.bb.voov,
        #    H.bb.oooo,
        #    H.bb.vvvv,
        #    H.bb.oovv,
        #    d3aab_o,
        #    d3aab_v,
        #    d3bbb_o,
        #    d3bbb_v,
        #    d3abb_o,
        #    d3abb_v,
        #)

        #### bbbb correction ####
        t1 = time.perf_counter()
        dA_bbbb, dB_bbbb, dC_bbbb, dD_bbbb = bbbb_correction(T, L, H, H0, d3bbb_o, d3bbb_v)
        # dA_bbbb, dB_bbbb, dC_bbbb, dD_bbbb = crcc24_opt_loops.crcc24e_opt(
        #     T.bb,
        #     L.bb,
        #     H0.b.oo,
        #     H0.b.vv,
        #     H.b.oo,
        #     H.b.vv,
        #     H.bb.voov,
        #     H.bb.oooo,
        #     H.bb.vvvv,
        #     H.bb.oovv,
        #     d3bbb_o,
        #     d3bbb_v,
        # )

        correction_A = dA_aaaa + dA_aaab + dA_aabb + dA_abbb + dA_bbbb
        correction_B = dB_aaaa + dB_aaab + dB_aabb + dB_abbb + dB_bbbb
        correction_C = dC_aaaa + dC_aaab + dC_aabb + dC_abbb + dC_bbbb
        correction_D = dD_aaaa + dD_aaab + dD_aabb + dD_abbb + dD_bbbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CR-CC(2,4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSD = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CR-CC(2,4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(2,4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(2,4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(2,4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc24 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta24 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc24, delta24

def moments_aaaa_ijkl(i, j, k, l, H, T):

    # # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl)
    # x4a = -0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.aa[:, :, j, l], optimize=True) # (1)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.aa[:, :, i, l], optimize=True) # (ij)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.aa[:, :, k, l], optimize=True) # (jk)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, k], T.aa[:, :, j, i], optimize=True) # (il)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, l], T.aa[:, :, j, k], optimize=True) # (kl)
    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, i, k], optimize=True) # (ij)(kl)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.aa[:, :, j, l], optimize=True) # (ik)
    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.aa[:, :, k, l], optimize=True) # (ij)(ik)
    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.aa[:, :, i, l], optimize=True) # (jk)(ik)
    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, i], T.aa[:, :, j, k], optimize=True) # (il)(ik)
    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, l], T.aa[:, :, j, i], optimize=True) # (kl)(ik)
    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, k, i], optimize=True) # (ij)(kl)(ik)
    # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl) -> A(i/jl)A(k/ijl)
    # (1)
    x4a = -0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.aa[:, :, j, l], optimize=True) # (1)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.aa[:, :, i, l], optimize=True) # (ij)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, k], T.aa[:, :, j, i], optimize=True) # (il)
    # (ik)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.aa[:, :, j, l], optimize=True) # (ik)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.aa[:, :, k, l], optimize=True) # (ij)(ik)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, i], T.aa[:, :, j, k], optimize=True) # (il)(ik)
    # (jk)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.aa[:, :, k, l], optimize=True) # (jk)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.aa[:, :, i, l], optimize=True) # (ij)(jk)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, j], T.aa[:, :, k, i], optimize=True) # (il)(jk)
    # (kl)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, l], T.aa[:, :, j, k], optimize=True) # (kl)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, i, k], optimize=True) # (ij)(kl)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, l], T.aa[:, :, j, i], optimize=True) # (il)(kl)

    # Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk)
    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, j], T.aa[:, :, :, l], T.aa[:, :, :, k], optimize=True) # (1)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, k, j], T.aa[:, :, :, l], T.aa[:, :, :, i], optimize=True) # (ik)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, l, j], T.aa[:, :, :, i], T.aa[:, :, :, k], optimize=True) # (il)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, k], T.aa[:, :, :, l], T.aa[:, :, :, j], optimize=True) # (jk)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, l], T.aa[:, :, :, j], T.aa[:, :, :, k], optimize=True) # (jl)
    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, k, l], T.aa[:, :, :, j], T.aa[:, :, :, i], optimize=True) # (ik)(jl)

    # Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil)
    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, k], T.aa[:, :, i, l], optimize=True) # (1)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, k], T.aa[:, :, j, l], optimize=True) # (ij)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, l, k], T.aa[:, :, i, j], optimize=True) # (jl)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, i], T.aa[:, :, k, l], optimize=True) # (ik)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, l], T.aa[:, :, i, k], optimize=True) # (kl)
    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, l], T.aa[:, :, j, k], optimize=True) # (ij)(kl)

    # antisymmetrize A(abcd)
    x4a -= np.transpose(x4a, (1, 0, 2, 3)) + np.transpose(x4a, (2, 0, 1, 3)) + np.transpose(x4a, (3, 1, 2, 0)) # (a/bcd)
    x4a -= np.transpose(x4a, (0, 2, 1, 3)) + np.transpose(x4a, (0, 3, 2, 1)) # (b/cd)
    x4a -= np.transpose(x4a, (0, 1, 3, 2)) # (cd)
    return x4a

def moments_aaab_ijkl(i, j, k, l, H, T):

    # Diagram 1:  -A(i/jk)A(c/ab) h2b(mdel) * t2a(abim) * t2a(ecjk)
    x4b = -0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, :], T.aa[:, :, j, k], optimize=True) # (1)
    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, j, :], T.aa[:, :, i, k], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, k, :], T.aa[:, :, j, i], optimize=True) # (ik)

    # Diagram 2:   A(k/ij)A(a/bc) h2a(mnij) * t2a(bcnk) * t2b(adml)
    x4b += 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, i, j], T.aa[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, k, j], T.aa[:, :, :, i], T.ab[:, :, :, l], optimize=True) # (ik)
    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, i, k], T.aa[:, :, :, j], T.ab[:, :, :, l], optimize=True) # (jk)
    
    # Diagram 3:  -A(ijk)A(c/ab) h2b(mdjf) * t2a(abim) * t2b(cfkl)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, i, :], T.ab[:, :, k, l], optimize=True) # (1)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, j, :], T.ab[:, :, k, l], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ik)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, k, :], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (jk)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (ij)(jk)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, k, :], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ik)(jk)

    # Diagram 4:  -A(ijk)A(abc) h2b(amie) * t2b(bejl) * t2b(cdkm)
    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, j, l], T.ab[:, :, k, :], optimize=True) # (1)
    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, i, l], T.ab[:, :, k, :], optimize=True) # (ij)
    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, k, :], T.ab[:, :, j, l], T.ab[:, :, i, :], optimize=True) # (ik)
    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, k, l], T.ab[:, :, j, :], optimize=True) # (jk)
    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, k, :], T.ab[:, :, i, l], T.ab[:, :, j, :], optimize=True) # (ij)(jk)
    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, k, l], T.ab[:, :, i, :], optimize=True) # (ik)(jk)
    
    # Diagram 5:   A(ijk)A(a/bc) h2b(mnjl) * t2a(bcmk) * t2b(adin)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, i], T.ab[:, :, k, :], optimize=True) # (ik)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, k, l], T.aa[:, :, :, j], T.ab[:, :, i, :], optimize=True) # (jk)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, j], T.ab[:, :, k, :], optimize=True) # (ij)(jk)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, k, l], T.aa[:, :, :, i], T.ab[:, :, j, :], optimize=True) # (ik)(jk)
    
    # Diagram 6:  -A(i/jk)A(abc) h2b(bmel) * t2a(ecjk) * t2b(adim)
    x4b -= np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, j, k], T.ab[:, :, i, :], optimize=True) # (1)
    x4b += np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, i, k], T.ab[:, :, j, :], optimize=True) # (ij)
    x4b += np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, j, i], T.ab[:, :, k, :], optimize=True) # (ik)
    
    # Diagram 7:  -A(i/jk)A(abc) h2a(amie) * t2a(ecjk) * t2b(bdml)
    x4b -= np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, j, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4b += np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, i, k], T.ab[:, :, :, l], optimize=True) # (ij)
    x4b += np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, j, i], T.ab[:, :, :, l], optimize=True) # (ik)
    
    # Diagram 8:   A(i/jk)A(c/ab) h2a(abef) * t2a(fcjk) * t2b(edil)
    x4b += 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, i], T.ab[:, :, k, l], optimize=True) # (ik)
    
    # Diagram 9:  -A(ijk)A(a/bc) h2a(amie) * t2a(bcmk) * t2b(edjl)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.ab[:, :, j, l], optimize=True) # (ik)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.ab[:, :, k, l], optimize=True) # (jk)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.ab[:, :, i, l], optimize=True) # (ij)(jk)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.ab[:, :, k, l], optimize=True) # (ik)(jk)
    
    # Diagram 10:  A(k/ij)A(abc) h2b(adef) * t2a(ebij) * t2b(cfkl)
    x4b += np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, i, j], T.ab[:, :, k, l], optimize=True) # (1)
    x4b -= np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, k, j], T.ab[:, :, i, l], optimize=True) # (ik)
    x4b -= np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (jk)

    # antisymmetrize A(abc)
    x4b -= np.transpose(x4b, (1, 0, 2, 3)) + np.transpose(x4b, (2, 1, 0, 3)) # (a/bc)
    x4b -= np.transpose(x4b, (0, 2, 1, 3)) # (bc)
    return x4b

def moments_aabb_ijkl(i, j, k, l, H, T):

    # Diagram 1:  -A(ij)A(kl)A(ab)A(cd) h2c(cmke) * t2b(adim) * t2b(bejl)
    x4c = -np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, k, :], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, k, :], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, l, :], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, l, :], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

    # Diagram 2:  -A(ij)A(kl)A(ab)A(cd) h2a(amie) * t2b(bcmk) * t2b(edjl)
    x4c -= np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

    # Diagram 3:  -A(kl)A(ab)A(cd) h2b(mcek) * t2a(aeij) * t2b(bdml)
    x4c -= np.einsum("mce,ae,bdm->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, i, j], T.ab[:, :, :, l], optimize=True) # (1)
    x4c += np.einsum("mce,ae,bdm->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, j], T.ab[:, :, :, k], optimize=True) # (2)

    # Diagram 4:  -A(ij)A(ab)A(cd) h2b(amie) * t2b(bdjm) * t2c(cekl)
    x4c -= np.einsum("ame,bdm,ce->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (1)
    x4c += np.einsum("ame,bdm,ce->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (ij)

    # Diagram 5:  -A(ij)A(kl)A(cd) h2b(mcek) * t2a(abim) * t2b(edjl)
    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

    # Diagram 6:  -A(ij)A(kl)A(ab) h2b(amie) * t2c(cdkm) * t2b(bejl)
    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, i, :], T.bb[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, j, :], T.bb[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, i, :], T.bb[:, :, l, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, j, :], T.bb[:, :, l, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

    # Diagram 7:  -A(ij)A(kl)A(ab)A(cd) h2b(bmel) * t2b(adim) * t2b(ecjk)
    x4c -= np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, l], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (1)
    x4c += np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, l], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c += np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, k], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c -= np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, k], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

    # Diagram 8:  -A(ij)A(kl)A(ab)A(cd) h2b(mdje) * t2b(bcmk) * t2b(aeil)
    x4c -= np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (1)
    x4c += np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (ij)
    x4c += np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (kl)
    x4c -= np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (ij)(kl)

    # Diagram 9:  -A(ij)A(cd) h2b(mdje) * t2a(abim) * t2c(cekl)
    x4c -= 0.5 * np.einsum("mde,abm,ce->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("mde,abm,ce->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (ij)

    # Diagram 10: -A(kl)A(ab) h2b(bmel) * t2a(aeij) * t2c(cdkm)
    x4c -= 0.5 * np.einsum("bme,ae,cdm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, i, j], T.bb[:, :, k, :], optimize=True) # (1)
    x4c += 0.5 * np.einsum("bme,ae,cdm->abcd", H.ab.vovo[:, :, :, k], T.aa[:, :, i, j], T.bb[:, :, l, :], optimize=True) # (kl)

    # Diagram 11:  A(kl)A(ab) h2a(mnij) * t2b(acmk) * t2b(bdnl)
    x4c += 0.5 * np.einsum("mn,acm,bdn->abcd", H.aa.oooo[:, :, i, j], T.ab[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4c -= 0.5 * np.einsum("mn,acm,bdn->abcd", H.aa.oooo[:, :, i, j], T.ab[:, :, :, l], T.ab[:, :, :, k], optimize=True) # (kl)

    # Diagram 12:  A(ij)A(kl) h2a(abef) * t2b(ecik) * t2b(fdjl)
    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

    # Diagram 13:  A(ij)A(kl) h2b(mnik) * t2a(abmj) * t2c(cdnl)
    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, i, k], T.aa[:, :, :, j], T.bb[:, :, :, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, j, k], T.aa[:, :, :, i], T.bb[:, :, :, l], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, j], T.bb[:, :, :, k], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, i], T.bb[:, :, :, k], optimize=True) # (ij)(kl)

    # Diagram 14:  A(ab)A(cd) h2b(acef) * t2a(ebij) * t2c(fdkl)
    x4c += np.einsum("acef,eb,fd->abcd", H.ab.vvvv, T.aa[:, :, i, j], T.bb[:, :, k, l], optimize=True) # (1)

    # Diagram 15:  A(ij)A(kl)A(ab)A(cd) h2b(mnik) * t2b(adml) * t2b(bcjn)
    x4c += np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, i, k], T.ab[:, :, :, l], T.ab[:, :, j, :], optimize=True) # (1)
    x4c -= np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, j, k], T.ab[:, :, :, l], T.ab[:, :, i, :], optimize=True) # (ij)
    x4c -= np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, i, l], T.ab[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (kl)
    x4c += np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, j, l], T.ab[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (ij)(kl)

    # Diagram 16:  A(ij)A(kl)A(ab)A(cd) h2b(acef) * t2b(edil) * t2b(bfjk)
    x4c += np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
    x4c -= np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c -= np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c += np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

    # Diagram 17:  A(ij)A(cd) h2c(mnkl) * t2b(adin) * t2b(bcjm)
    x4c += 0.5 * np.einsum("mn,adn,bcm->abcd", H.bb.oooo[:, :, k, l], T.ab[:, :, i, :], T.ab[:, :, j, :], optimize=True) # (1)
    x4c -= 0.5 * np.einsum("mn,adn,bcm->abcd", H.bb.oooo[:, :, k, l], T.ab[:, :, j, :], T.ab[:, :, i, :], optimize=True) # (ij)

    # Diagram 18:  A(ij)A(kl) h2c(cdef) * t2b(afil) * t2b(bejk)
    x4c += 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

    # antisymmetrize A(ab)A(cd)
    x4c -= np.transpose(x4c, (1, 0, 2, 3)) # (ab)
    x4c -= np.transpose(x4c, (0, 1, 3, 2)) # (cd)
    return x4c

def aaaa_correction(T, L, H, H0, d3aaa_o, d3aaa_v):

    nua, noa = T.a.shape

    dA_aaaa = 0.0
    dB_aaaa = 0.0
    dC_aaaa = 0.0
    dD_aaaa = 0.0

    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(j + 1, noa):
                for l in range(k + 1, noa):
                    
                    # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl)
                    # x4a = -0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.aa[:, :, j, l], optimize=True) # (1)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.aa[:, :, i, l], optimize=True) # (ij)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.aa[:, :, k, l], optimize=True) # (jk)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, k], T.aa[:, :, j, i], optimize=True) # (il)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, l], T.aa[:, :, j, k], optimize=True) # (kl)
                    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, i, k], optimize=True) # (ij)(kl)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.aa[:, :, j, l], optimize=True) # (ik)
                    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.aa[:, :, k, l], optimize=True) # (ij)(ik)
                    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.aa[:, :, i, l], optimize=True) # (jk)(ik)
                    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, i], T.aa[:, :, j, k], optimize=True) # (il)(ik)
                    # x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, l], T.aa[:, :, j, i], optimize=True) # (kl)(ik)
                    # x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, k, i], optimize=True) # (ij)(kl)(ik)
                    # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl) -> A(i/jl)A(k/ijl)
                    # (1)
                    x4a = -0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.aa[:, :, j, l], optimize=True) # (1)
                    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.aa[:, :, i, l], optimize=True) # (ij)
                    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, k], T.aa[:, :, j, i], optimize=True) # (il)
                    # (ik)
                    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.aa[:, :, j, l], optimize=True) # (ik)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.aa[:, :, k, l], optimize=True) # (ij)(ik)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, i], T.aa[:, :, j, k], optimize=True) # (il)(ik)
                    # (jk)
                    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.aa[:, :, k, l], optimize=True) # (jk)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.aa[:, :, i, l], optimize=True) # (ij)(jk)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, l, :], T.aa[:, :, :, j], T.aa[:, :, k, i], optimize=True) # (il)(jk)
                    # (kl)
                    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, l], T.aa[:, :, j, k], optimize=True) # (kl)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, i, k], optimize=True) # (ij)(kl)
                    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, l], T.aa[:, :, j, i], optimize=True) # (il)(kl)

                    # Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk)
                    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, j], T.aa[:, :, :, l], T.aa[:, :, :, k], optimize=True) # (1)
                    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, k, j], T.aa[:, :, :, l], T.aa[:, :, :, i], optimize=True) # (ik)
                    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, l, j], T.aa[:, :, :, i], T.aa[:, :, :, k], optimize=True) # (il)
                    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, k], T.aa[:, :, :, l], T.aa[:, :, :, j], optimize=True) # (jk)
                    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, i, l], T.aa[:, :, :, j], T.aa[:, :, :, k], optimize=True) # (jl)
                    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", H.aa.oooo[:, :, k, l], T.aa[:, :, :, j], T.aa[:, :, :, i], optimize=True) # (ik)(jl)

                    # Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil)
                    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, k], T.aa[:, :, i, l], optimize=True) # (1)
                    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, k], T.aa[:, :, j, l], optimize=True) # (ij)
                    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, l, k], T.aa[:, :, i, j], optimize=True) # (jl)
                    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, i], T.aa[:, :, k, l], optimize=True) # (ik)
                    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, l], T.aa[:, :, i, k], optimize=True) # (kl)
                    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, l], T.aa[:, :, j, k], optimize=True) # (ij)(kl)

                    dA, dB, dC, dD = crcc24_opt_loops.crcc24a_ijkl(
                            i + 1, j + 1, k + 1, l + 1,
                            x4a, T.aa, L.aa,
                            H0.a.oo, H0.a.vv,
                            H.a.oo, H.a.vv,
                            H.aa.voov, H.aa.oooo, H.aa.vvvv, H.aa.oovv,
                            d3aaa_o, d3aaa_v
                    )

                    dA_aaaa += dA
                    dB_aaaa += dB
                    dC_aaaa += dC
                    dD_aaaa += dD

    return dA_aaaa, dB_aaaa, dC_aaaa, dD_aaaa

                
def aaab_correction(T, L, H, H0, d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    dA_aaab = 0.0
    dB_aaab = 0.0
    dC_aaab = 0.0
    dD_aaab = 0.0

    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(j + 1, noa):
                for l in range(nob):

                    # Diagram 1:  -A(i/jk)A(c/ab) h2b(mdel) * t2a(abim) * t2a(ecjk)
                    x4b = -0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, :], T.aa[:, :, j, k], optimize=True) # (1)
                    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, j, :], T.aa[:, :, i, k], optimize=True) # (ij)
                    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, k, :], T.aa[:, :, j, i], optimize=True) # (ik)

                    # Diagram 2:   A(k/ij)A(a/bc) h2a(mnij) * t2a(bcnk) * t2b(adml)
                    x4b += 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, i, j], T.aa[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
                    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, k, j], T.aa[:, :, :, i], T.ab[:, :, :, l], optimize=True) # (ik)
                    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", H.aa.oooo[:, :, i, k], T.aa[:, :, :, j], T.ab[:, :, :, l], optimize=True) # (jk)
                    
                    # Diagram 3:  -A(ijk)A(c/ab) h2b(mdjf) * t2a(abim) * t2b(cfkl)
                    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, i, :], T.ab[:, :, k, l], optimize=True) # (1)
                    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, j, :], T.ab[:, :, k, l], optimize=True) # (ij)
                    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ik)
                    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, k, :], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (jk)
                    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (ij)(jk)
                    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", H.ab.ovov[:, :, k, :], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ik)(jk)

                    # Diagram 4:  -A(ijk)A(abc) h2b(amie) * t2b(bejl) * t2b(cdkm)
                    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, j, l], T.ab[:, :, k, :], optimize=True) # (1)
                    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, i, l], T.ab[:, :, k, :], optimize=True) # (ij)
                    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, k, :], T.ab[:, :, j, l], T.ab[:, :, i, :], optimize=True) # (ik)
                    x4b += np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, k, l], T.ab[:, :, j, :], optimize=True) # (jk)
                    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, k, :], T.ab[:, :, i, l], T.ab[:, :, j, :], optimize=True) # (ij)(jk)
                    x4b -= np.einsum("ame,be,cdm->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, k, l], T.ab[:, :, i, :], optimize=True) # (ik)(jk)
                    
                    # Diagram 5:   A(ijk)A(a/bc) h2b(mnjl) * t2a(bcmk) * t2b(adin)
                    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (1)
                    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (ij)
                    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, i], T.ab[:, :, k, :], optimize=True) # (ik)
                    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, k, l], T.aa[:, :, :, j], T.ab[:, :, i, :], optimize=True) # (jk)
                    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, j], T.ab[:, :, k, :], optimize=True) # (ij)(jk)
                    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", H.ab.oooo[:, :, k, l], T.aa[:, :, :, i], T.ab[:, :, j, :], optimize=True) # (ik)(jk)
                    
                    # Diagram 6:  -A(i/jk)A(abc) h2b(bmel) * t2a(ecjk) * t2b(adim)
                    x4b -= np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, j, k], T.ab[:, :, i, :], optimize=True) # (1)
                    x4b += np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, i, k], T.ab[:, :, j, :], optimize=True) # (ij)
                    x4b += np.einsum("bme,ec,adm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, j, i], T.ab[:, :, k, :], optimize=True) # (ik)
                    
                    # Diagram 7:  -A(i/jk)A(abc) h2a(amie) * t2a(ecjk) * t2b(bdml)
                    x4b -= np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, j, k], T.ab[:, :, :, l], optimize=True) # (1)
                    x4b += np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, i, k], T.ab[:, :, :, l], optimize=True) # (ij)
                    x4b += np.einsum("ame,ec,bdm->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, j, i], T.ab[:, :, :, l], optimize=True) # (ik)
                    
                    # Diagram 8:   A(i/jk)A(c/ab) h2a(abef) * t2a(fcjk) * t2b(edil)
                    x4b += 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (1)
                    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (ij)
                    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", H.aa.vvvv, T.aa[:, :, j, i], T.ab[:, :, k, l], optimize=True) # (ik)
                    
                    # Diagram 9:  -A(ijk)A(a/bc) h2a(amie) * t2a(bcmk) * t2b(edjl)
                    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
                    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, i], T.ab[:, :, j, l], optimize=True) # (ik)
                    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.aa[:, :, :, j], T.ab[:, :, k, l], optimize=True) # (jk)
                    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, k, :], T.aa[:, :, :, j], T.ab[:, :, i, l], optimize=True) # (ij)(jk)
                    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.aa[:, :, :, i], T.ab[:, :, k, l], optimize=True) # (ik)(jk)
                    
                    # Diagram 10:  A(k/ij)A(abc) h2b(adef) * t2a(ebij) * t2b(cfkl)
                    x4b += np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, i, j], T.ab[:, :, k, l], optimize=True) # (1)
                    x4b -= np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, k, j], T.ab[:, :, i, l], optimize=True) # (ik)
                    x4b -= np.einsum("adef,eb,cf->abcd", H.ab.vvvv, T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (jk)
                                    
                    dA, dB, dC, dD = crcc24_opt_loops.crcc24b_ijkl(
                        i + 1, j + 1, k + 1, l + 1,
                        x4b, T.aa, T.ab, L.aa, L.ab,
                        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.voov, H.aa.oooo, H.aa.vvvv, H.aa.oovv,
                        H.ab.voov, H.ab.ovov, H.ab.vovo, H.ab.ovvo, H.ab.oooo, H.ab.vvvv, H.ab.oovv,
                        H.bb.voov,
                        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                    )

                    dA_aaab += dA
                    dB_aaab += dB
                    dC_aaab += dC
                    dD_aaab += dD

    return dA_aaab, dB_aaab, dC_aaab, dD_aaab

def aabb_correction(T, L, H, H0, d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    dA_aabb = 0.0
    dB_aabb = 0.0
    dC_aabb = 0.0
    dD_aabb = 0.0

    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(nob):
                for l in range(k + 1, nob):

                    # Diagram 1:  -A(ij)A(kl)A(ab)A(cd) h2c(cmke) * t2b(adim) * t2b(bejl)
                    x4c = -np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, k, :], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
                    x4c += np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, k, :], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4c += np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, l, :], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
                    x4c -= np.einsum("cme,adm,be->abcd", H.bb.voov[:, :, l, :], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

                    # Diagram 2:  -A(ij)A(kl)A(ab)A(cd) h2a(amie) * t2b(bcmk) * t2b(edjl)
                    x4c -= np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
                    x4c += np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4c += np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (kl)
                    x4c -= np.einsum("ame,bcm,ed->abcd", H.aa.voov[:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

                    # Diagram 3:  -A(kl)A(ab)A(cd) h2b(mcek) * t2a(aeij) * t2b(bdml)
                    x4c -= np.einsum("mce,ae,bdm->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, i, j], T.ab[:, :, :, l], optimize=True) # (1)
                    x4c += np.einsum("mce,ae,bdm->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, j], T.ab[:, :, :, k], optimize=True) # (2)

                    # Diagram 4:  -A(ij)A(ab)A(cd) h2b(amie) * t2b(bdjm) * t2c(cekl)
                    x4c -= np.einsum("ame,bdm,ce->abcd", H.ab.voov[:, :, i, :], T.ab[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (1)
                    x4c += np.einsum("ame,bdm,ce->abcd", H.ab.voov[:, :, j, :], T.ab[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (ij)

                    # Diagram 5:  -A(ij)A(kl)A(cd) h2b(mcek) * t2a(abim) * t2b(edjl)
                    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
                    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, k], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
                    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", H.ab.ovvo[:, :, :, l], T.aa[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

                    # Diagram 6:  -A(ij)A(kl)A(ab) h2b(amie) * t2c(cdkm) * t2b(bejl)
                    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, i, :], T.bb[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (1)
                    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, j, :], T.bb[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, i, :], T.bb[:, :, l, :], T.ab[:, :, j, k], optimize=True) # (kl)
                    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", H.ab.voov[:, :, j, :], T.bb[:, :, l, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

                    # Diagram 7:  -A(ij)A(kl)A(ab)A(cd) h2b(bmel) * t2b(adim) * t2b(ecjk)
                    x4c -= np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, l], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (1)
                    x4c += np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, l], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)
                    x4c += np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, k], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (kl)
                    x4c -= np.einsum("bme,adm,ec->abcd", H.ab.vovo[:, :, :, k], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

                    # Diagram 8:  -A(ij)A(kl)A(ab)A(cd) h2b(mdje) * t2b(bcmk) * t2b(aeil)
                    x4c -= np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (1)
                    x4c += np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (ij)
                    x4c += np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (kl)
                    x4c -= np.einsum("mde,bcm,ae->abcd", H.ab.ovov[:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (ij)(kl)

                    # Diagram 9:  -A(ij)A(cd) h2b(mdje) * t2a(abim) * t2c(cekl)
                    x4c -= 0.5 * np.einsum("mde,abm,ce->abcd", H.ab.ovov[:, :, j, :], T.aa[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (1)
                    x4c += 0.5 * np.einsum("mde,abm,ce->abcd", H.ab.ovov[:, :, i, :], T.aa[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (ij)

                    # Diagram 10: -A(kl)A(ab) h2b(bmel) * t2a(aeij) * t2c(cdkm)
                    x4c -= 0.5 * np.einsum("bme,ae,cdm->abcd", H.ab.vovo[:, :, :, l], T.aa[:, :, i, j], T.bb[:, :, k, :], optimize=True) # (1)
                    x4c += 0.5 * np.einsum("bme,ae,cdm->abcd", H.ab.vovo[:, :, :, k], T.aa[:, :, i, j], T.bb[:, :, l, :], optimize=True) # (kl)

                    # Diagram 11:  A(kl)A(ab) h2a(mnij) * t2b(acmk) * t2b(bdnl)
                    x4c += 0.5 * np.einsum("mn,acm,bdn->abcd", H.aa.oooo[:, :, i, j], T.ab[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
                    x4c -= 0.5 * np.einsum("mn,acm,bdn->abcd", H.aa.oooo[:, :, i, j], T.ab[:, :, :, l], T.ab[:, :, :, k], optimize=True) # (kl)

                    # Diagram 12:  A(ij)A(kl) h2a(abef) * t2b(ecik) * t2b(fdjl)
                    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (1)
                    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)
                    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (kl)
                    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", H.aa.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)

                    # Diagram 13:  A(ij)A(kl) h2b(mnik) * t2a(abmj) * t2c(cdnl)
                    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, i, k], T.aa[:, :, :, j], T.bb[:, :, :, l], optimize=True) # (1)
                    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, j, k], T.aa[:, :, :, i], T.bb[:, :, :, l], optimize=True) # (ij)
                    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, i, l], T.aa[:, :, :, j], T.bb[:, :, :, k], optimize=True) # (kl)
                    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", H.ab.oooo[:, :, j, l], T.aa[:, :, :, i], T.bb[:, :, :, k], optimize=True) # (ij)(kl)

                    # Diagram 14:  A(ab)A(cd) h2b(acef) * t2a(ebij) * t2c(fdkl)
                    x4c += np.einsum("acef,eb,fd->abcd", H.ab.vvvv, T.aa[:, :, i, j], T.bb[:, :, k, l], optimize=True) # (1)

                    # Diagram 15:  A(ij)A(kl)A(ab)A(cd) h2b(mnik) * t2b(adml) * t2b(bcjn)
                    x4c += np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, i, k], T.ab[:, :, :, l], T.ab[:, :, j, :], optimize=True) # (1)
                    x4c -= np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, j, k], T.ab[:, :, :, l], T.ab[:, :, i, :], optimize=True) # (ij)
                    x4c -= np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, i, l], T.ab[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (kl)
                    x4c += np.einsum("mn,adm,bcn->abcd", H.ab.oooo[:, :, j, l], T.ab[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (ij)(kl)

                    # Diagram 16:  A(ij)A(kl)A(ab)A(cd) h2b(acef) * t2b(edil) * t2b(bfjk)
                    x4c += np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
                    x4c -= np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
                    x4c -= np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
                    x4c += np.einsum("acef,ed,bf->abcd", H.ab.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

                    # Diagram 17:  A(ij)A(cd) h2c(mnkl) * t2b(adin) * t2b(bcjm)
                    x4c += 0.5 * np.einsum("mn,adn,bcm->abcd", H.bb.oooo[:, :, k, l], T.ab[:, :, i, :], T.ab[:, :, j, :], optimize=True) # (1)
                    x4c -= 0.5 * np.einsum("mn,adn,bcm->abcd", H.bb.oooo[:, :, k, l], T.ab[:, :, j, :], T.ab[:, :, i, :], optimize=True) # (ij)

                    # Diagram 18:  A(ij)A(kl) h2c(cdef) * t2b(afil) * t2b(bejk)
                    x4c += 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
                    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
                    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
                    x4c += 0.25 * np.einsum("cdef,af,be->abcd", H.bb.vvvv, T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)

                    dA, dB, dC, dD = crcc24_opt_loops.crcc24c_ijkl(
                              i + 1, j + 1, k + 1, l + 1,
                              x4c, T.aa, T.ab, T.bb, L.aa, L.ab, L.bb,
                              H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                              H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                              H.aa.voov, H.aa.oooo, H.aa.vvvv, H.aa.oovv,
                              H.ab.voov, H.ab.ovov, H.ab.vovo, H.ab.ovvo, H.ab.oooo, H.ab.vvvv, H.ab.oovv,
                              H.bb.voov, H.bb.oooo, H.bb.vvvv, H.bb.oovv,
                              d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                    )

                    dA_aabb += dA
                    dB_aabb += dB
                    dC_aabb += dC
                    dD_aabb += dD

    return dA_aabb, dB_aabb, dC_aabb, dD_aabb


def abbb_correction(T, L, H, H0, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v):

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    dA_abbb = 0.0
    dB_abbb = 0.0
    dC_abbb = 0.0
    dD_abbb = 0.0

    for l in range(noa):
        for i in range(nob):
            for j in range(i + 1, nob):
                for k in range(j + 1, nob):

                    # Diagram 1:  -A(i/jk)A(c/ab) h2b(dmle) * t2c(abim) * t2c(ecjk)
                    x4d = -0.5 * np.einsum("dme,abm,ec->dabc", H.ab.voov[:, :, l, :], T.bb[:, :, i, :], T.bb[:, :, j, k], optimize=True) # (1)
                    x4d += 0.5 * np.einsum("dme,abm,ec->dabc", H.ab.voov[:, :, l, :], T.bb[:, :, j, :], T.bb[:, :, i, k], optimize=True) # (ij)
                    x4d += 0.5 * np.einsum("dme,abm,ec->dabc", H.ab.voov[:, :, l, :], T.bb[:, :, k, :], T.bb[:, :, j, i], optimize=True) # (ik)

                    # Diagram 2:   A(k/ij)A(a/bc) h2c(mnij) * t2c(bcnk) * t2b(dalm)
                    x4d += 0.5 * np.einsum("mn,bcn,dam->dabc", H.bb.oooo[:, :, i, j], T.bb[:, :, :, k], T.ab[:, :, l, :], optimize=True) # (1)
                    x4d -= 0.5 * np.einsum("mn,bcn,dam->dabc", H.bb.oooo[:, :, k, j], T.bb[:, :, :, i], T.ab[:, :, l, :], optimize=True) # (ik)
                    x4d -= 0.5 * np.einsum("mn,bcn,dam->dabc", H.bb.oooo[:, :, i, k], T.bb[:, :, :, j], T.ab[:, :, l, :], optimize=True) # (jk)

                    # Diagram 3:  -A(ijk)A(c/ab) h2b(dmfj) * t2c(abim) * t2b(fclk)
                    x4d -= 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, j], T.bb[:, :, i, :], T.ab[:, :, l, k], optimize=True) # (1)
                    x4d += 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, i], T.bb[:, :, j, :], T.ab[:, :, l, k], optimize=True) # (ij)
                    x4d += 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, j], T.bb[:, :, k, :], T.ab[:, :, l, i], optimize=True) # (ik)
                    x4d += 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, k], T.bb[:, :, i, :], T.ab[:, :, l, j], optimize=True) # (jk)
                    x4d -= 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, i], T.bb[:, :, k, :], T.ab[:, :, l, j], optimize=True) # (ij)(jk)
                    x4d -= 0.5 * np.einsum("dmf,abm,fc->dabc", H.ab.vovo[:, :, :, k], T.bb[:, :, j, :], T.ab[:, :, l, i], optimize=True) # (ik)(jk)

                    # Diagram 4:  -A(ijk)A(abc) h2b(maei) * t2b(eblj) * t2b(dcmk)
                    x4d -= np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, i], T.ab[:, :, l, j], T.ab[:, :, :, k], optimize=True)
                    x4d += np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, j], T.ab[:, :, l, i], T.ab[:, :, :, k], optimize=True)
                    x4d += np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, k], T.ab[:, :, l, j], T.ab[:, :, :, i], optimize=True)
                    x4d += np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, i], T.ab[:, :, l, k], T.ab[:, :, :, j], optimize=True)
                    x4d -= np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, k], T.ab[:, :, l, i], T.ab[:, :, :, j], optimize=True)
                    x4d -= np.einsum("mae,eb,dcm->dabc", H.ab.ovvo[:, :, :, j], T.ab[:, :, l, k], T.ab[:, :, :, i], optimize=True)

                    # Diagram 5:   A(ijk)A(a/bc) h2b(nmlj) * t2c(bcmk) * t2b(dani)
                    x4d += 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, j], T.bb[:, :, :, k], T.ab[:, :, :, i], optimize=True)
                    x4d -= 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, i], T.bb[:, :, :, k], T.ab[:, :, :, j], optimize=True)
                    x4d -= 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, j], T.bb[:, :, :, i], T.ab[:, :, :, k], optimize=True)
                    x4d -= 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, k], T.bb[:, :, :, j], T.ab[:, :, :, i], optimize=True)
                    x4d += 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, i], T.bb[:, :, :, j], T.ab[:, :, :, k], optimize=True)
                    x4d += 0.5 * np.einsum("nm,bcm,dan->dabc", H.ab.oooo[:, :, l, k], T.bb[:, :, :, i], T.ab[:, :, :, j], optimize=True)

                    # Diagram 6:  -A(i/jk)A(abc) h2b(mble) * t2c(ecjk) * t2b(dami)
                    x4d -= np.einsum("mbe,ec,dam->dabc", H.ab.ovov[:, :, l, :], T.bb[:, :, j, k], T.ab[:, :, :, i], optimize=True)
                    x4d += np.einsum("mbe,ec,dam->dabc", H.ab.ovov[:, :, l, :], T.bb[:, :, i, k], T.ab[:, :, :, j], optimize=True)
                    x4d += np.einsum("mbe,ec,dam->dabc", H.ab.ovov[:, :, l, :], T.bb[:, :, j, i], T.ab[:, :, :, k], optimize=True)

                    # Diagram 7:  -A(i/jk)A(abc) h2c(amie) * t2c(ecjk) * t2b(dblm)
                    x4d -= np.einsum("ame,ec,dbm->dabc", H.bb.voov[:, :, i, :], T.bb[:, :, j, k], T.ab[:, :, l, :], optimize=True)
                    x4d += np.einsum("ame,ec,dbm->dabc", H.bb.voov[:, :, j, :], T.bb[:, :, i, k], T.ab[:, :, l, :], optimize=True)
                    x4d += np.einsum("ame,ec,dbm->dabc", H.bb.voov[:, :, k, :], T.bb[:, :, j, i], T.ab[:, :, l, :], optimize=True)

                    # Diagram 8:   A(i/jk)A(c/ab) h2c(abef) * t2c(fcjk) * t2b(deli)
                    x4d += 0.5 * np.einsum("abef,fc,de->dabc", H.bb.vvvv, T.bb[:, :, j, k], T.ab[:, :, l, i], optimize=True)
                    x4d -= 0.5 * np.einsum("abef,fc,de->dabc", H.bb.vvvv, T.bb[:, :, i, k], T.ab[:, :, l, j], optimize=True)
                    x4d -= 0.5 * np.einsum("abef,fc,de->dabc", H.bb.vvvv, T.bb[:, :, j, i], T.ab[:, :, l, k], optimize=True)

                    # Diagram 9:  -A(ijk)A(a/bc) h2c(amie) * t2c(bcmk) * t2b(delj)
                    x4d -= 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, i, :], T.bb[:, :, :, k], T.ab[:, :, l, j], optimize=True)
                    x4d += 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, j, :], T.bb[:, :, :, k], T.ab[:, :, l, i], optimize=True)
                    x4d += 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, k, :], T.bb[:, :, :, i], T.ab[:, :, l, j], optimize=True)
                    x4d += 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, i, :], T.bb[:, :, :, j], T.ab[:, :, l, k], optimize=True)
                    x4d -= 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, k, :], T.bb[:, :, :, j], T.ab[:, :, l, i], optimize=True)
                    x4d -= 0.5 * np.einsum("ame,bcm,de->dabc", H.bb.voov[:, :, j, :], T.bb[:, :, :, i], T.ab[:, :, l, k], optimize=True)

                    # Diagram 10:  A(k/ij)A(abc) h2b(dafe) * t2c(ebij) * t2b(fclk)
                    x4d += np.einsum("dafe,eb,fc->dabc", H.ab.vvvv, T.bb[:, :, i, j], T.ab[:, :, l, k], optimize=True)
                    x4d -= np.einsum("dafe,eb,fc->dabc", H.ab.vvvv, T.bb[:, :, k, j], T.ab[:, :, l, i], optimize=True)
                    x4d -= np.einsum("dafe,eb,fc->dabc", H.ab.vvvv, T.bb[:, :, i, k], T.ab[:, :, l, j], optimize=True)

                    dA, dB, dC, dD = crcc24_opt_loops.crcc24d_ijkl(
                              i + 1, j + 1, k + 1, l + 1,
                              x4d, T.ab, T.bb, L.ab, L.bb,
                              H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                              H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                              H.aa.voov,
                              H.ab.voov, H.ab.ovov, H.ab.vovo, H.ab.ovvo, H.ab.oooo, H.ab.vvvv, H.ab.oovv,
                              H.bb.voov, H.bb.oooo, H.bb.vvvv, H.bb.oovv,
                              d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                    )

                    dA_abbb += dA
                    dB_abbb += dB
                    dC_abbb += dC
                    dD_abbb += dD

    return dA_abbb, dB_abbb, dC_abbb, dD_abbb


def bbbb_correction(T, L, H, H0, d3bbb_o, d3bbb_v):
    nub, nob = T.b.shape

    dA_bbbb = 0.0
    dB_bbbb = 0.0
    dC_bbbb = 0.0
    dD_bbbb = 0.0

    for i in range(nob):
        for j in range(i + 1, nob):
            for k in range(j + 1, nob):
                for l in range(k + 1, nob):
                    # Diagram 1: -A(jl/i/k)A(bc/a/d) h2c(amie) * t2c(bcmk) * t2c(edjl)
                    x4e = -0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, i, :], T.bb[:, :, :, k],
                                           T.bb[:, :, j, l], optimize=True)  # (1)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, j, :], T.bb[:, :, :, k],
                                           T.bb[:, :, i, l], optimize=True)  # (ij)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, i, :], T.bb[:, :, :, j],
                                           T.bb[:, :, k, l], optimize=True)  # (jk)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, l, :], T.bb[:, :, :, k],
                                           T.bb[:, :, j, i], optimize=True)  # (il)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, i, :], T.bb[:, :, :, l],
                                           T.bb[:, :, j, k], optimize=True)  # (kl)
                    x4e -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, j, :], T.bb[:, :, :, l],
                                           T.bb[:, :, i, k], optimize=True)  # (ij)(kl)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, k, :], T.bb[:, :, :, i],
                                           T.bb[:, :, j, l], optimize=True)  # (ik)
                    x4e -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, j, :], T.bb[:, :, :, i],
                                           T.bb[:, :, k, l], optimize=True)  # (ij)(ik)
                    x4e -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, k, :], T.bb[:, :, :, j],
                                           T.bb[:, :, i, l], optimize=True)  # (jk)(ik)
                    x4e -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, l, :], T.bb[:, :, :, i],
                                           T.bb[:, :, j, k], optimize=True)  # (il)(ik)
                    x4e -= 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, k, :], T.bb[:, :, :, l],
                                           T.bb[:, :, j, i], optimize=True)  # (kl)(ik)
                    x4e += 0.5 * np.einsum("ame,bcm,ed->abcd", H.bb.voov[:, :, j, :], T.bb[:, :, :, l],
                                           T.bb[:, :, k, i], optimize=True)  # (ij)(kl)(ik)

                    # Diagram 2: A(ij/kl)A(bc/ad) h2c(mnij) * t2c(adml) * t2c(bcnk)
                    x4e += 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, i, j], T.bb[:, :, :, l],
                                            T.bb[:, :, :, k], optimize=True)  # (1)
                    x4e -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, k, j], T.bb[:, :, :, l],
                                            T.bb[:, :, :, i], optimize=True)  # (ik)
                    x4e -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, l, j], T.bb[:, :, :, i],
                                            T.bb[:, :, :, k], optimize=True)  # (il)
                    x4e -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, i, k], T.bb[:, :, :, l],
                                            T.bb[:, :, :, j], optimize=True)  # (jk)
                    x4e -= 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, i, l], T.bb[:, :, :, j],
                                            T.bb[:, :, :, k], optimize=True)  # (jl)
                    x4e += 0.25 * np.einsum("mn,adm,bcn->abcd", H.bb.oooo[:, :, k, l], T.bb[:, :, :, j],
                                            T.bb[:, :, :, i], optimize=True)  # (ik)(jl)

                    # Diagram 3: A(jk/il)A(ab/cd) h2c(abef) * t2c(fcjk) * t2c(edil)
                    x4e += 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, j, k], T.bb[:, :, i, l],
                                            optimize=True)  # (1)
                    x4e -= 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, i, k], T.bb[:, :, j, l],
                                            optimize=True)  # (ij)
                    x4e -= 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, l, k], T.bb[:, :, i, j],
                                            optimize=True)  # (jl)
                    x4e -= 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, j, i], T.bb[:, :, k, l],
                                            optimize=True)  # (ik)
                    x4e -= 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, j, l], T.bb[:, :, i, k],
                                            optimize=True)  # (kl)
                    x4e += 0.25 * np.einsum("abef,fc,ed->abcd", H.bb.vvvv, T.bb[:, :, i, l], T.bb[:, :, j, k],
                                            optimize=True)  # (ij)(kl)

                    dA, dB, dC, dD = crcc24_opt_loops.crcc24e_ijkl(
                        i + 1, j + 1, k + 1, l + 1,
                        x4e, T.bb, L.bb,
                        H0.b.oo, H0.b.vv,
                        H.b.oo, H.b.vv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv, H.bb.oovv,
                        d3bbb_o, d3bbb_v
                    )

                    dA_bbbb += dA
                    dB_bbbb += dB
                    dC_bbbb += dC
                    dD_bbbb += dD

    return dA_bbbb, dB_bbbb, dC_bbbb, dD_bbbb
