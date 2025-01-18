import numpy as np
from ccpy.lib.core import deaeom4_p_loops
from ccpy.eomcc.deaeom4_intermediates import get_deaeom4_p_intermediates

def update(R, omega, H, RHF_symmetry, system, r3_excitations):
    R.ab, R.aba, R.abb, R.abaa, R.abab, R.abbb = deaeom4_p_loops.update_r(
        R.ab,
        R.aba,
        R.abb,
        R.abaa, 
        r3_excitations["abaa"],
        R.abab, 
        r3_excitations["abab"],
        R.abbb, 
        r3_excitations["abbb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system, t3_excitations, r3_excitations):

    # determine whether r3 updates should be done. Stupid compatibility with
    # empty sections r3_excitations
    do_r3 = {"abaa": True, "abab": True, "abbb": True}
    if np.array_equal(r3_excitations["abaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["abaa"] = False
    if np.array_equal(r3_excitations["abab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["abab"] = False
    if np.array_equal(r3_excitations["abbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3["abbb"] = False

    # Get intermediates
    X = get_deaeom4_p_intermediates(H, T, R, r3_excitations)
    # update R1
    dR = build_HR_2B(dR, R, r3_excitations, T, H)
    # update R2
    dR = build_HR_3B(dR, R, r3_excitations, T, X, H)
    dR = build_HR_3C(dR, R, r3_excitations, T, X, H)
    # update R3
    if do_r3["abaa"]:
        dR, R, r3_excitations = build_HR_4B(dR, R, r3_excitations, T, X, H)
    if do_r3["abab"]:
        dR, R, r3_excitations = build_HR_4C(dR, R, r3_excitations, T, X, H)
    if do_r3["abbb"]:
        dR, R, r3_excitations = build_HR_4D(dR, R, r3_excitations, T, X, H)
    return dR.flatten()

def build_HR_2B(dR, R, r3_excitations, T, H):
    """Calculate the projection <ab~|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.ab = np.einsum("ae,eb->ab", H.a.vv, R.ab, optimize=True)
    dR.ab += np.einsum("be,ae->ab", H.b.vv, R.ab, optimize=True)
    dR.ab += np.einsum("abef,ef->ab", H.ab.vvvv, R.ab, optimize=True)
    dR.ab += np.einsum("me,abem->ab", H.a.ov, R.aba, optimize=True)
    dR.ab += np.einsum("me,abem->ab", H.b.ov, R.abb, optimize=True)
    dR.ab += np.einsum("nbfe,aefn->ab", H.ab.ovvv, R.aba, optimize=True)
    dR.ab += 0.5 * np.einsum("anef,ebfn->ab", H.aa.vovv, R.aba, optimize=True)
    dR.ab += 0.5 * np.einsum("bnef,aefn->ab", H.bb.vovv, R.abb, optimize=True)
    dR.ab += np.einsum("anef,ebfn->ab", H.ab.vovv, R.abb, optimize=True)
    dR.ab = deaeom4_p_loops.build_hr_2b(
            dR.ab,
            R.abaa, r3_excitations["abaa"],
            R.abab, r3_excitations["abab"],
            R.abbb, r3_excitations["abbb"],
            H.aa.oovv, H.ab.oovv, H.bb.oovv,
    )
    return dR

def build_HR_3B(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~ck|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.aba = 0.5 * np.einsum("cake,eb->abck", H.aa.vvov, R.ab, optimize=True)
    dR.aba += np.einsum("cbke,ae->abck", H.ab.vvov, R.ab, optimize=True)
    dR.aba += np.einsum("ae,ebck->abck", H.a.vv, R.aba, optimize=True)
    dR.aba += 0.5 * np.einsum("be,aeck->abck", H.b.vv, R.aba, optimize=True)
    dR.aba += np.einsum("abef,efck->abck", H.ab.vvvv, R.aba, optimize=True)
    dR.aba += 0.25 * np.einsum("acef,ebfk->abck", H.aa.vvvv, R.aba, optimize=True)
    dR.aba += np.einsum("cmke,abem->abck", H.aa.voov, R.aba, optimize=True)
    dR.aba += np.einsum("cmke,abem->abck", H.ab.voov, R.abb, optimize=True)
    dR.aba -= 0.5 * np.einsum("mbke,aecm->abck", H.ab.ovov, R.aba, optimize=True)
    dR.aba -= 0.5 * np.einsum("mb,acmk->abck", X["ab"]["ov"], T.aa, optimize=True)
    dR.aba -= np.einsum("am,cbkm->abck", X["ab"]["vo"], T.ab, optimize=True)
    dR.aba -= 0.5 * np.einsum("mk,abcm->abck", H.a.oo, R.aba, optimize=True)
    dR.aba = deaeom4_p_loops.build_hr_3b(
             dR.aba,
             R.abaa, r3_excitations["abaa"],
             R.abab, r3_excitations["abab"],
             H.a.ov, H.b.ov,
             H.aa.ooov, H.aa.vovv, 
             H.ab.ooov, H.ab.vovv, H.ab.ovvv,
             H.bb.vovv,
    )
    return dR

def build_HR_3C(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~c~k~|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.abb = np.einsum("acek,eb->abck", H.ab.vvvo, R.ab, optimize=True)
    dR.abb += 0.5 * np.einsum("cbke,ae->abck", H.bb.vvov, R.ab, optimize=True)
    dR.abb += 0.5 * np.einsum("ae,ebck->abck", H.a.vv, R.abb, optimize=True)
    dR.abb += np.einsum("be,aeck->abck", H.b.vv, R.abb, optimize=True)
    dR.abb += np.einsum("abef,efck->abck", H.ab.vvvv, R.abb, optimize=True)
    dR.abb += 0.25 * np.einsum("bcef,aefk->abck", H.bb.vvvv, R.abb, optimize=True)
    dR.abb += np.einsum("mcek,abem->abck", H.ab.ovvo, R.aba, optimize=True)
    dR.abb += np.einsum("cmke,abem->abck", H.bb.voov, R.abb, optimize=True)
    dR.abb -= 0.5 * np.einsum("amek,ebcm->abck", H.ab.vovo, R.abb, optimize=True)
    dR.abb -= np.einsum("mb,acmk->abck", X["ab"]["ov"], T.ab, optimize=True)
    dR.abb -= 0.5 * np.einsum("am,bcmk->abck", X["ab"]["vo"], T.bb, optimize=True)
    dR.abb -= 0.5 * np.einsum("mk,abcm->abck", H.b.oo, R.abb, optimize=True)
    dR.abb = deaeom4_p_loops.build_hr_3c(
             dR.abb,
             R.abab, r3_excitations["abab"],
             R.abbb, r3_excitations["abbb"],
             H.a.ov, H.b.ov,
             H.aa.vovv,
             H.ab.oovo, H.ab.vovv, H.ab.ovvv,
             H.bb.ooov, H.bb.vovv, 
    )
    return dR

def build_HR_4B(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~cdkl|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.abaa, R.abaa, r3_excitations["abaa"] = deaeom4_p_loops.build_hr_4b(
            R.aba,
            R.abaa, r3_excitations["abaa"],
            R.abab, r3_excitations["abab"],
            T.aa, T.ab,
            H.a.oo, H.a.vv, H.b.vv,
            H.aa.vvvv, H.aa.oooo, H.aa.voov, H.aa.vooo, H.aa.vvov,
            H.ab.vvvv, H.ab.voov, H.ab.ovov, H.ab.vvov,
            X["aba"]["vvoo"], X["aba"]["vvvv"], X["aba"]["vovo"],
            X["ab"]["oo"],
    )
    return dR, R, r3_excitations

def build_HR_4C(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~cd~kl~|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.abab, R.abab, r3_excitations["abab"] = deaeom4_p_loops.build_hr_4c(
            R.aba, R.abb,
            R.abaa, r3_excitations["abaa"],
            R.abab, r3_excitations["abab"],
            R.abbb, r3_excitations["abbb"],
            T.aa, T.ab, T.bb,
            H.a.oo, H.b.oo, H.a.vv, H.b.vv,
            H.aa.vvvv, H.aa.voov, H.aa.vvov,
            H.ab.vvvv, H.ab.oooo, H.ab.voov, H.ab.ovvo, H.ab.ovov, H.ab.vovo,
            H.ab.vvov, H.ab.vvvo, H.ab.vooo, H.ab.ovoo,
            H.bb.vvvv, H.bb.voov, H.bb.vvov,
            X["aba"]["vvvv"], X["aba"]["vvoo"], X["aba"]["vovo"], 
            X["abb"]["vvvv"], X["abb"]["vvoo"], X["abb"]["ovvo"], 
            X["ab"]["oo"],
    )
    return dR, R, r3_excitations

def build_HR_4D(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ab~c~d~k~l~|[ (H_N e^(T1+T2))_C*(R2p+R3p1h+R4p2h) ]_C|0>."""
    dR.abbb, R.abbb, r3_excitations["abbb"] = deaeom4_p_loops.build_hr_4d(
            R.abb,
            R.abab, r3_excitations["abab"],
            R.abbb, r3_excitations["abbb"],
            T.ab, T.bb,
            H.b.oo, H.a.vv, H.b.vv,
            H.ab.vvvv, H.ab.ovvo, H.ab.vovo, H.ab.vvvo, 
            H.bb.vvvv, H.bb.oooo, H.bb.voov, H.bb.vvov, H.bb.vooo,
            X["abb"]["vvvv"], X["abb"]["ovvo"], X["abb"]["vvoo"],
            X["ab"]["oo"],
    )
    return dR, R, r3_excitations
