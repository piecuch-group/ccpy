"""CCSDT(P) diagrams of ccsdt_p_loops.f90 expressed as generator specs (the limiting case).

ROUTINES[name] lists (block name, spec) in the order of the hand-written routine."""
from codegen import Amp, Op, H1, LinearDiagram

T3A = Amp("t3a_excits", "t3a_amps", "n3aaa", "aaa", "aaa")
T3B = Amp("t3b_excits", "t3b_amps", "n3aab", "aab", "aab")
T3C = Amp("t3c_excits", "t3c_amps", "n3abb", "abb", "abb")
T3D = Amp("t3d_excits", "t3d_amps", "n3bbb", "bbb", "bbb")

# stored layouts (positions of the natural indices)
VOOV = (1, 3, 0, 2)   # h(c,m,k,e) -> h(m,e,c,k)
REV = (3, 2, 1, 0)    # h(p,q,r,s) -> h(s,r,q,p)
NAT = (0, 1, 2, 3)
OOOO = (1, 0, 3, 2)  # h(l,m,i,j) -> h(m,l,j,i)

NEW = "! allocate new sorting arrays"


def same_spin(T, s):
    """Diagrams 1-6 of update_t3a_p (s = "a") or update_t3d_p (s = "b")."""
    x, c = ("a", "a") if s == "a" else ("b", "c")
    X = x.upper()
    C = c.upper()
    other = T3B if s == "a" else T3C
    lad_o = LinearDiagram(
        header=[f"!!!! diagram 1: -A(i/jk) h1{x}(mi) * {T.excits[:3]}(abcmjk)",
                f"!!!! diagram 3: 1/2 A(i/jk) h2{c}(mnij) * {T.excits[:3]}(abcmnk)",
                f"! NOTE: WITHIN THESE LOOPS, H1{X}(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2"],
        target=T, tv="abc", to="ijk", source=T, sv="abc", so="mnk",
        op=Op(f"h2{c}_oooo", f"h2{c}(oooo)", "mnij", OOOO, f"H2{C}_oooo"),
        h1=H1(f"h1{x}_oo", f"H1{X}_oo", s, -1, free_first=False, combine="half"),
        loop_order=("ABCK", "ABCI", "ABCJ"), check_first="none", alloc_comment=NEW)
    lad_v = LinearDiagram(
        header=[f"!!!! diagram 2: A(a/bc) h1{x}(ae) * {T.excits[:3]}(ebcijk)",
                f"!!!! diagram 4: 1/2 A(c/ab) h2{c}(abef) * {T.excits[:3]}(ebcijk)",
                f"! NOTE: WITHIN THESE LOOPS, H1{X}(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2"],
        target=T, tv="abc", to="ijk", source=T, sv="aef", so="ijk",
        op=Op(f"h2{c}_vvvv", f"h2{c}(vvvv)", "bcef", REV, f"H2{C}_vvvv"),
        h1=H1(f"h1{x}_vv", f"H1{X}_vv", s, 1, free_first=True, combine="half"),
        key_order="ov", check_first="none", alloc_comment=NEW)
    d5 = LinearDiagram(
        header=[f"!!!! diagram 5: A(i/jk)A(a/bc) h2{c}(amie) * {T.excits[:3]}(ebcmjk)"],
        target=T, tv="abc", to="ijk", source=T, sv="abe", so="ijm",
        op=Op(f"h2{c}_voov", f"h2{c}(voov)", "cmke", VOOV, f"H2{C}_voov"))
    return [("lad_o", lad_o), ("lad_v", lad_v), ("d5", d5)]


T3A_D6 = LinearDiagram(
    header=["!!!! diagram 6: A(i/jk)A(a/bc) h2b(amie) * t3b(abeijm)"],
    target=T3A, tv="abc", to="ijk", source=T3B, sv="abe", so="ijm",
    op=Op("h2b_voov", "h2b(voov)", "cmke", VOOV, "H2B_voov"),
    outer="v", buffer=True, buffer_name="t3")

OVOV = (0, 3, 1, 2)   # h(m,c,i,e) -> h(m,e,c,i)
VOVO = (1, 2, 0, 3)   # h(a,m,e,k) -> h(m,e,a,k)
OVVO = (0, 2, 1, 3)   # h(m,c,e,k) -> h(m,e,c,k)


def t3b_blocks():
    T = T3B
    L = LinearDiagram
    return [
        ("d1_5", L(header=["!!!! diagram 1: -A(ij) h1a(mi)*t3b(abcmjk)", "!!!! diagram 5: A(ij) 1/2 h2a(mnij)*t3b(abcmnk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="abc", so="mnk",
                   op=Op("h2a_oooo", "h2a(oooo)", "mnij", OOOO, "H2A_oooo"),
                   h1=H1("h1a_oo", "H1A_oo", "a", -1, free_first=False), check_first="none", alloc_comment=NEW)),
        ("d2_6", L(header=["!!!! diagram 2: A(ab) h1a(ae)*t3b(ebcmjk)", "!!!! diagram 6: A(ab) 1/2 h2a(abef)*t3b(ebcmjk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="efc", so="ijk",
                   op=Op("h2a_vvvv", "h2a(vvvv)", "abef", REV, "H2A_vvvv"),
                   h1=H1("h1a_vv", "H1A_vv", "a", 1, free_first=True), key_order="ov", check_first="none", alloc_comment=NEW)),
        ("d3_7", L(header=["!!!! diagram 3: -h1b(mk)*t3b(abcijm)", "!!!! diagram 7: A(ij) h2b(mnjk)*t3b(abcimn)"],
                   target=T, tv="abc", to="ijk", source=T, sv="abc", so="imn",
                   op=Op("h2b_oooo", "h2b(oooo)", "mnjk", NAT, "H2B_oooo"),
                   h1=H1("h1b_oo", "H1B_oo", "b", -1, free_first=False, loops=("ABCI",)),
                   check_first="none", nest=True, alloc_comment=NEW)),
        ("d5_8", L(header=["!!!! diagram 5: h1b(ce)*t3b(abeijm)", "!!!! diagram 8: A(ab) h2b(bcef)*t3b(aefijk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="aef", so="ijk",
                   op=Op("h2b_vvvv", "h2b(vvvv)", "bcef", REV, "H2B_vvvv"),
                   h1=H1("h1b_vv", "H1B_vv", "b", 1, free_first=True, loops=("IJKA",)),
                   key_order="ov", check_first="none", nest=True, alloc_comment=NEW)),
        ("d9", L(header=["!!!! diagram 9: A(ij)A(ab) h2a(amie)*t3b(ebcmjk)"],
                 target=T, tv="abc", to="ijk", source=T, sv="ebc", so="mjk",
                 op=Op("h2a_voov", "h2a(voov)", "amie", VOOV, "H2A_voov"),
                 loop_order=("BCJK", "BCIK", "ACIK", "ACJK"), check_first="none", nest=True,
                 cnames="first", outer_by_loop={"BCIK": "v", "ACIK": "v", "ACJK": "v"})),
        ("d10", L(header=["!!!! diagram 10: h2c(cmke)*t3b(abeijm)"],
                  target=T, tv="abc", to="ijk", source=T, sv="abe", so="ijm",
                  op=Op("h2c_voov", "h2c(voov)", "cmke", VOOV, "H2C_voov"), check_first="none")),
        ("d11", L(header=["!!!! diagram 11: -A(ij) h2b(mcie)*t3b(abemjk)"],
                  target=T, tv="abc", to="ijk", source=T, sv="abe", so="mjk",
                  op=Op("h2b_ovov", "h2b(ovov)", "mcie", OVOV, "H2B_ovov"), sign=-1, check_first="none", nest=True)),
        ("d12", L(header=["!!!! diagram 12: -A(ab) h2b(amek)*t3b(ebcijm)"],
                  target=T, tv="abc", to="ijk", source=T, sv="ebc", so="ijm",
                  op=Op("h2b_vovo", "h2b(vovo)", "amek", VOVO, "H2B_vovo"), sign=-1, key_order="ov",
                  loop_order=("IJBC", "IJAC"), check_first="none", nest=True)),
        ("d13", L(header=["!!!! diagram 13: h2b(mcek)*t3a(abeijm) !!!!"],
                  target=T, tv="abc", to="ijk", source=T3A, sv="abe", so="ijm",
                  op=Op("h2b_ovvo", "h2b(ovvo)", "mcek", OVVO, "H2B_ovvo"),
                  buffer=True, buffer_name="t3", check_first="cycle", check_rest="cycle")),
        ("d14", L(header=["!!!! diagram 14: A(ab)A(ij) h2b(bmje)*t3c(aecimk)"],
                  target=T, tv="abc", to="ijk", source=T3C, sv="aec", so="imk",
                  op=Op("h2b_voov", "h2b(voov)", "bmje", VOOV, "H2B_voov"),
                  buffer=True, buffer_name="t3", loop_order=("ACIK", "ABIK", "ACIJ", "ABIJ"), nest=True)),
    ]


VVVV_B = (2, 3, 0, 1)  # h2b(a,c,d,f) -> h2b(d,f,a,c)
KC = "BCAIJK"   # t3c key priorities
KO = "JKIABC"


def t3c_blocks():
    T = T3C
    L = LinearDiagram
    nine = ("BCJK", "BCIK", "BCIJ", "ACJK", "ACIK", "ACIJ", "ABJK", "ABIK", "ABIJ")
    return [
        ("d1_5", L(header=["!!!! diagram 1: -A(jk) h1b(mk)*t3c(abcijm)", "!!!! diagram 5: A(jk) 1/2 h2c(mnjk)*t3c(abcimn)"],
                   target=T, tv="abc", to="ijk", source=T, sv="abc", so="imn",
                   op=Op("h2c_oooo", "h2c(oooo)", "mnjk", OOOO, "H2C_oooo"),
                   h1=H1("h1b_oo", "H1B_oo", "b", -1, free_first=False), key_order=KC, check_first="none", alloc_comment=NEW)),
        ("d2_6", L(header=["!!!! diagram 2: A(bc) h1b(ce)*t3c(abeijk)", "!!!! diagram 6: A(bc) 1/2 h2c(bcef)*t3c(aefijk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="aef", so="ijk",
                   op=Op("h2c_vvvv", "h2c(vvvv)", "bcef", REV, "H2C_vvvv"),
                   h1=H1("h1b_vv", "H1B_vv", "b", 1, free_first=True), key_order=KO, check_first="none", alloc_comment=NEW)),
        ("d3_7", L(header=["!!!! diagram 3: -h1a(mi)*t3c(abcmjk)", "!!!! diagram 7: A(jk) h2b(mnij)*t3c(abcmnk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="abc", so="mnk",
                   op=Op("h2b_oooo", "h2b(oooo)", "mnij", NAT, "H2B_oooo"),
                   h1=H1("h1a_oo", "H1A_oo", "a", -1, free_first=False, loops=("BCAK",)),
                   key_order=KC, loop_order=("BCAK", "BCAJ"), check_first="none", nest=True, alloc_comment=NEW)),
        ("d5_8", L(header=["!!!! diagram 5: h1a(ae)*t3c(ebcijk)", "!!!! diagram 8: A(bc) h2b(abef)*t3c(efcijk)"],
                   target=T, tv="abc", to="ijk", source=T, sv="efc", so="ijk",
                   op=Op("h2b_vvvv", "h2b(vvvv)", "abef", VVVV_B, "H2B_vvvv"),
                   h1=H1("h1a_vv", "H1A_vv", "a", 1, free_first=True, loops=("JKIB",)),
                   key_order=KO, check_first="none", nest=True, alloc_comment=NEW)),
        ("d9", L(header=["!!!! diagram 9: A(jk)A(bc) h2c(cmke)*t3c(abeijm)"],
                 target=T, tv="abc", to="ijk", source=T, sv="abe", so="ijm",
                 op=Op("h2c_voov", "h2c(voov)", "cmke", VOOV, "H2C_voov"), outer="v", loop_outer="o", check_first="none", nest=True)),
        ("d10", L(header=["!!!! diagram 10: h2a(amie)*t3c(ebcmjk)"],
                  target=T, tv="abc", to="ijk", source=T, sv="ebc", so="mjk",
                  op=Op("h2a_voov", "h2a(voov)", "amie", VOOV, "H2A_voov"), check_first="none")),
        ("d11", L(header=["!!!! diagram 11: -A(bc) h2b(mbie)*t3c(aecmjk)"],
                  target=T, tv="abc", to="ijk", source=T, sv="aec", so="mjk",
                  op=Op("h2b_ovov", "h2b(ovov)", "mbie", OVOV, "H2B_ovov"), sign=-1, key_order=KO,
                  loop_order=("JKAC", "JKAB"), check_first="none", nest=True)),
        ("d12", L(header=["!!!! diagram 12: -A(bc) h2b(amej)*t3c(ebcimk)"],
                  target=T, tv="abc", to="ijk", source=T, sv="ebc", so="imk",
                  op=Op("h2b_vovo", "h2b(vovo)", "amej", VOVO, "H2B_vovo"), sign=-1,
                  loop_order=("BCIK", "BCIJ"), check_first="none", nest=True)),
        ("d13", L(header=["!!!! diagram 13: h2b(amie)*t3d(ebcmjk)"],
                  target=T, tv="abc", to="ijk", source=T3D, sv="ebc", so="mjk",
                  op=Op("h2b_voov", "h2b(voov)", "amie", VOOV, "H2B_voov"),
                  buffer=True, buffer_name="t3", loop_order=nine, nest=True)),
        ("d14", L(header=["!!!! diagram 14: A(bc)A(jk) h2b(mbej)*t3b(aecimk)"],
                  target=T, tv="abc", to="ijk", source=T3B, sv="aec", so="imk",
                  op=Op("h2b_ovvo", "h2b(ovvo)", "mbej", OVVO, "H2B_ovvo"),
                  buffer=True, buffer_name="t3", nest=True)),
    ]


T3D_D6 = LinearDiagram(
    header=["!!!! diagram 6: A(i/jk)A(a/bc) h2b(maei) * t3c(ebcmjk)"],
    target=T3D, tv="abc", to="ijk", source=T3C, sv="ebc", so="mjk",
    op=Op("h2b_ovvo", "h2b(ovvo)", "maei", OVVO, "H2B_ovvo"),
    buffer=True, buffer_name="t3")


ROUTINES = {
    "update_t3a_p": same_spin(T3A, "a") + [("d6", T3A_D6)],
    "update_t3b_p": t3b_blocks(),
    "update_t3c_p": t3c_blocks(),
    "update_t3d_p": same_spin(T3D, "b") + [("d6", T3D_D6)],
}

T3A_D5 = dict(ROUTINES["update_t3a_p"])["d5"]
CCSDT_SPECS = {f"{r[7:10]}_{n}": s for r, blocks in ROUTINES.items() for n, s in blocks}
SPEC_ROUTINE = {f"{r[7:10]}_{n}": r for r, blocks in ROUTINES.items() for n, s in blocks}


# ---------------------------------------------------------------- interfaces
# Dummy-argument order (the Python call interface) and stored array shapes of
# update_t3?_p, as they are in ccsdt_p_loops.f90.
ARGS = {'a': ['resid', 't3a_amps', 't3a_excits', 't3b_amps', 't3b_excits', 't2a', 'H1A_oo', 'H1A_vv', 'H2A_oovv', 'H2A_vvov', 'H2A_vooo', 'H2A_oooo', 'H2A_voov', 'H2A_vvvv', 'H2B_oovv', 'H2B_voov', 'fA_oo', 'fA_vv', 'shift', 'n3aaa', 'n3aab', 'noa', 'nua', 'nob', 'nub'], 'b': ['resid', 't3a_amps', 't3a_excits', 't3b_amps', 't3b_excits', 't3c_amps', 't3c_excits', 't2a', 't2b', 'H1A_oo', 'H1A_vv', 'H1B_oo', 'H1B_vv', 'H2A_oovv', 'H2A_vvov', 'H2A_vooo', 'H2A_oooo', 'H2A_voov', 'H2A_vvvv', 'H2B_oovv', 'H2B_vvov', 'H2B_vvvo', 'H2B_vooo', 'H2B_ovoo', 'H2B_oooo', 'H2B_voov', 'H2B_vovo', 'H2B_ovov', 'H2B_ovvo', 'H2B_vvvv', 'H2C_oovv', 'H2C_voov', 'fA_oo', 'fA_vv', 'fB_oo', 'fB_vv', 'shift', 'n3aaa', 'n3aab', 'n3abb', 'noa', 'nua', 'nob', 'nub'], 'c': ['resid', 't3b_amps', 't3b_excits', 't3c_amps', 't3c_excits', 't3d_amps', 't3d_excits', 't2b', 't2c', 'H1A_oo', 'H1A_vv', 'H1B_oo', 'H1B_vv', 'H2A_oovv', 'H2A_voov', 'H2B_oovv', 'H2B_vooo', 'H2B_ovoo', 'H2B_vvov', 'H2B_vvvo', 'H2B_oooo', 'H2B_voov', 'H2B_vovo', 'H2B_ovov', 'H2B_ovvo', 'H2B_vvvv', 'H2C_oovv', 'H2C_vooo', 'H2C_vvov', 'H2C_oooo', 'H2C_voov', 'H2C_vvvv', 'fA_oo', 'fA_vv', 'fB_oo', 'fB_vv', 'shift', 'n3aab', 'n3abb', 'n3bbb', 'noa', 'nua', 'nob', 'nub'], 'd': ['resid', 't3c_amps', 't3c_excits', 't3d_amps', 't3d_excits', 't2c', 'H1B_oo', 'H1B_vv', 'H2C_oovv', 'H2C_vvov', 'H2C_vooo', 'H2C_oooo', 'H2C_voov', 'H2C_vvvv', 'H2B_oovv', 'H2B_ovvo', 'fB_oo', 'fB_vv', 'shift', 'n3abb', 'n3bbb', 'noa', 'nua', 'nob', 'nub']}

SHAPES = {'a': {'t3b_excits': 'n3aab,6', 't2a': 'nua,nua,noa,noa', 'H1A_oo': 'noa,noa', 'H1A_vv': 'nua,nua', 'H2A_oovv': 'noa,noa,nua,nua', 'H2B_oovv': 'noa,nob,nua,nub', 'H2A_vvov': 'nua,nua,nua,noa', 'H2A_vooo': 'noa,nua,noa,noa', 'H2A_oooo': 'noa,noa,noa,noa', 'H2A_voov': 'noa,nua,nua,noa', 'H2A_vvvv': 'nua,nua,nua,nua', 'H2B_voov': 'nob,nub,nua,noa', 'fA_vv': 'nua,nua', 'fA_oo': 'noa,noa', 't3b_amps': 'n3aab', 't3a_excits': 'n3aaa,6', 't3a_amps': 'n3aaa', 'I2A_vvov': 'nua,nua,nua,noa', 'I2A_vooo': 'noa,nua,noa,noa'}, 'b': {'t3a_excits': 'n3aaa,6', 't3c_excits': 'n3abb,6', 't2a': 'nua,nua,noa,noa', 't2b': 'nua,nub,noa,nob', 't3a_amps': 'n3aaa', 't3c_amps': 'n3abb', 'H1A_oo': 'noa,noa', 'H1A_vv': 'nua,nua', 'H1B_oo': 'nob,nob', 'H1B_vv': 'nub,nub', 'H2A_oovv': 'noa,noa,nua,nua', 'H2A_vvov': 'nua,nua,nua,noa', 'H2A_vooo': 'noa,nua,noa,noa', 'H2A_oooo': 'noa,noa,noa,noa', 'H2A_voov': 'noa,nua,nua,noa', 'H2A_vvvv': 'nua,nua,nua,nua', 'H2B_oovv': 'noa,nob,nua,nub', 'H2B_vooo': 'nob,nua,noa,nob', 'H2B_ovoo': 'noa,nub,noa,nob', 'H2B_vvov': 'nub,nua,nub,noa', 'H2B_vvvo': 'nua,nua,nub,nob', 'H2B_oooo': 'noa,nob,noa,nob', 'H2B_voov': 'nob,nub,nua,noa', 'H2B_vovo': 'nob,nua,nua,nob', 'H2B_ovov': 'noa,nub,nub,noa', 'H2B_ovvo': 'noa,nua,nub,nob', 'H2B_vvvv': 'nub,nua,nub,nua', 'H2C_oovv': 'nob,nob,nub,nub', 'H2C_voov': 'nob,nub,nub,nob', 'fA_oo': 'noa,noa', 'fA_vv': 'nua,nua', 'fB_oo': 'nob,nob', 'fB_vv': 'nub,nub', 't3b_excits': 'n3aab,6', 't3b_amps': 'n3aab', 'I2A_vooo': 'noa,nua,noa,noa', 'I2A_vvov': 'nua,nua,nua,noa', 'I2B_vooo': 'nob,nua,noa,nob', 'I2B_ovoo': 'noa,nub,noa,nob', 'I2B_vvov': 'nub,nua,nub,noa', 'I2B_vvvo': 'nua,nua,nub,nob'}, 'c': {'t3b_excits': 'n3aab,6', 't3d_excits': 'n3bbb,6', 't2b': 'nua,nub,noa,nob', 't2c': 'nub,nub,nob,nob', 't3b_amps': 'n3aab', 't3d_amps': 'n3bbb', 'H1A_oo': 'noa,noa', 'H1A_vv': 'nua,nua', 'H1B_oo': 'nob,nob', 'H1B_vv': 'nub,nub', 'H2A_oovv': 'noa,noa,nua,nua', 'H2A_voov': 'noa,nua,nua,noa', 'H2B_oovv': 'noa,nob,nua,nub', 'H2B_vooo': 'nob,nua,noa,nob', 'H2B_ovoo': 'noa,nub,noa,nob', 'H2B_vvov': 'nub,nua,nub,noa', 'H2B_vvvo': 'nua,nua,nub,nob', 'H2B_oooo': 'noa,nob,noa,nob', 'H2B_voov': 'nob,nub,nua,noa', 'H2B_vovo': 'nob,nua,nua,nob', 'H2B_ovov': 'noa,nub,nub,noa', 'H2B_ovvo': 'noa,nua,nub,nob', 'H2B_vvvv': 'nua,nub,nua,nub', 'H2C_oovv': 'nob,nob,nub,nub', 'H2C_vooo': 'nob,nub,nob,nob', 'H2C_vvov': 'nub,nub,nub,nob', 'H2C_oooo': 'nob,nob,nob,nob', 'H2C_voov': 'nob,nub,nub,nob', 'H2C_vvvv': 'nub,nub,nub,nub', 'fA_oo': 'noa,noa', 'fA_vv': 'nua,nua', 'fB_oo': 'nob,nob', 'fB_vv': 'nub,nub', 't3c_excits': 'n3abb,6', 't3c_amps': 'n3abb', 'I2C_vooo': 'nob,nub,nob,nob', 'I2C_vvov': 'nub,nub,nub,nob', 'I2B_vooo': 'nob,nua,noa,nob', 'I2B_ovoo': 'noa,nub,noa,nob', 'I2B_vvov': 'nub,nua,nub,noa', 'I2B_vvvo': 'nua,nua,nub,nob'}, 'd': {'t3c_excits': 'n3abb,6', 't2c': 'nub,nub,nob,nob', 't3c_amps': 'n3abb', 'H1B_oo': 'nob,nob', 'H1B_vv': 'nub,nub', 'H2B_oovv': 'noa,nob,nua,nub', 'H2B_ovvo': 'noa,nua,nub,nob', 'H2C_oovv': 'nob,nob,nub,nub', 'H2C_vooo': 'nob,nub,nob,nob', 'H2C_vvov': 'nub,nub,nub,nob', 'H2C_oooo': 'nob,nob,nob,nob', 'H2C_voov': 'nob,nub,nub,nob', 'H2C_vvvv': 'nub,nub,nub,nub', 'fB_vv': 'nub,nub', 'fB_oo': 'nob,nob', 't3d_excits': 'n3bbb,6', 't3d_amps': 'n3bbb', 'I2C_vooo': 'nob,nub,nob,nob', 'I2C_vvov': 'nub,nub,nub,nob'}}


# ---------------------------------------------------------------- moments
from codegen import Moment, Intermediate, Routine, xbuf_build  # noqa: E402

D = ["deallocate(xbuf)"]


def M(T, comment, left, right, loop, sign=1, outer="o", pre=(), post=(), ind="    "):
    return Moment(comment, T, "abc", "ijk", left, right, loop, sign, outer, list(pre), list(post), ind=ind)


def same_spin_moments(T, s):
    x = "a" if s == "a" else "c"
    X = x.upper()
    o, v = ("noa", "nua") if s == "a" else ("nob", "nub")
    pre = xbuf_build(f"{o},{o},{v},{v}", [("a", v), ("b", v), ("i", o), ("j", o)], "j,i,b,a", f"t2{x}(b,a,j,i)")
    return [M(T, "-A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)", f"I2{X}_vooo(m,a,i,j)", "xbuf(m,k,b,c)", ("m", o), -1, pre=pre, post=D),
            M(T, "A(i/jk)(c/ab) h2a(abie) * t2a(ecjk)", f"I2{X}_vvov(e,a,b,i)", f"t2{x}(e,c,j,k)", ("e", v))]


T3B_MOMENTS = [
    M(T3B, "A(ab) I2B(bcek) * t2a(aeij)", "I2B_vvvo(e,b,c,k)", "t2a(e,a,j,i)", ("e", "nua")),
    M(T3B, "A(ij) I2A(abie) * t2b(ecjk)", "I2A_vvov(e,a,b,i)", "t2b(e,c,j,k)", ("e", "nua")),
    M(T3B, "A(ij)A(ab) I2b(acie) * t2b(bejk)", "I2B_vvov(e,a,c,i)", "xbuf(e,b,k,j)", ("e", "nub"), outer="v",
      pre=xbuf_build("nub,nua,nob,noa", [("i", "noa"), ("j", "nob"), ("a", "nua"), ("b", "nub")], "b,a,j,i", "t2b(a,b,i,j)"), post=D),
    M(T3B, "-A(ij) h2b(mcjk) * t2a(abim)", "I2B_ovoo(m,c,j,k)", "xbuf(m,i,b,a)", ("m", "noa"), -1,
      pre=xbuf_build("noa,noa,nua,nua", [("a", "nua"), ("b", "nua"), ("i", "noa"), ("j", "noa")], "j,i,b,a", "t2a(b,a,j,i)"), post=D),
    M(T3B, "-A(ab) h2a(amij) * t2b(bcmk)", "I2A_vooo(m,a,i,j)", "xbuf(m,k,b,c)", ("m", "noa"), -1,
      pre=xbuf_build("noa,nob,nua,nub", [("b", "nub"), ("a", "nua"), ("j", "nob"), ("i", "noa")], "i,j,a,b", "t2b(a,b,i,j)"), post=D),
    M(T3B, "-A(ij)A(ab) h2b(amik) * t2b(bcjm)", "I2B_vooo(m,a,i,k)", "xbuf(m,j,c,b)", ("m", "nob"), -1,
      pre=xbuf_build("nob,noa,nub,nua", [("a", "nua"), ("b", "nub"), ("i", "noa"), ("j", "nob")], "j,i,b,a", "t2b(a,b,i,j)"), post=D),
]

I3 = "   "
T3C_MOMENTS = [
    M(T3C, "A(jk)A(bc) h2B(abej) * t2b(ecik)", "I2B_vvvo(e,a,b,j)", "t2b(e,c,i,k)", ("e", "nua"), outer="v", ind=I3),
    M(T3C, "A(bc) h2B(abie) * t2c(ecjk)", "I2B_vvov(e,a,b,i)", "t2c(e,c,j,k)", ("e", "nub"), ind=I3),
    M(T3C, "A(jk) h2C(cbke) * t2b(aeij)", "I2C_vvov(e,c,b,k)", "xbuf(e,a,j,i)", ("e", "nub"), ind=I3,
      pre=xbuf_build("nub,nua,nob,noa", [("i", "noa"), ("j", "nob"), ("a", "nua"), ("b", "nub")], "b,a,j,i", "t2b(a,b,i,j)"), post=D),
    M(T3C, "-A(kj)A(bc) h2b(mbij) * t2b(acmk)", "I2B_ovoo(m,b,i,j)", "xbuf(m,k,a,c)", ("m", "noa"), -1, ind=I3,
      pre=xbuf_build("noa,nob,nua,nub", [("b", "nub"), ("a", "nua"), ("j", "nob"), ("i", "noa")], "i,j,a,b", "t2b(a,b,i,j)"), post=D),
    M(T3C, "-A(jk) h2b(amij) * t2c(bcmk)", "I2B_vooo(m,a,i,j)", "xbuf(m,k,b,c)", ("m", "nob"), -1, ind=I3,
      pre=xbuf_build("nob,nob,nub,nub", [("b", "nub"), ("a", "nub"), ("j", "nob"), ("i", "nob")], "i,j,a,b", "t2c(a,b,i,j)"), post=D),
    M(T3C, "-A(bc) h2c(cmkj) * t2b(abim)", "I2C_vooo(m,c,k,j)", "xbuf(m,i,b,a)", ("m", "nob"), -1, ind=I3,
      pre=xbuf_build("nob,noa,nub,nua", [("a", "nua"), ("b", "nub"), ("i", "noa"), ("j", "nob")], "j,i,b,a", "t2b(a,b,i,j)"), post=D),
]

_A = ["H2A_oovv", "H2B_oovv"]
_C = ["H2B_oovv", "H2C_oovv"]
_n = ["noa", "nua", "nob", "nub"]
_ab = ["t3a_excits", "t3a_amps", "t3b_excits", "t3b_amps", "n3aaa", "n3aab"]
_bc = ["t3b_excits", "t3b_amps", "t3c_excits", "t3c_amps", "n3aab", "n3abb"]
_cd = ["t3c_excits", "t3c_amps", "t3d_excits", "t3d_amps", "n3abb", "n3bbb"]


def I2(name, half, hs, amps):
    return Intermediate(name, "0.5d0 * " if half else "", "H" + name[1:], hs + amps + _n)


INTERMEDIATES = {
    "update_t3a_p": [I2("I2A_vooo", 1, _A, _ab), I2("I2A_vvov", 1, _A, _ab)],
    "update_t3b_p": [I2("I2A_vooo", 1, _A, _ab), I2("I2A_vvov", 1, _A, _ab), I2("I2B_vooo", 0, _C, _bc),
                     I2("I2B_ovoo", 0, _A, _bc), I2("I2B_vvov", 0, _C, _bc), I2("I2B_vvvo", 0, _A, _bc)],
    "update_t3c_p": [I2("I2C_vooo", 1, _C, _cd), I2("I2C_vvov", 1, _C, _cd), I2("I2B_vooo", 0, _C, _bc),
                     I2("I2B_ovoo", 0, _A, _bc), I2("I2B_vvov", 0, _C, _bc), I2("I2B_vvvo", 0, _A, _bc)],
    "update_t3d_p": [I2("I2C_vooo", 1, _C, _cd), I2("I2C_vvov", 1, _C, _cd)],
}
MOMENTS = {"update_t3a_p": same_spin_moments(T3A, "a"), "update_t3b_p": T3B_MOMENTS,
           "update_t3c_p": T3C_MOMENTS, "update_t3d_p": same_spin_moments(T3D, "b")}
TARGET = {"update_t3a_p": T3A, "update_t3b_p": T3B, "update_t3c_p": T3C, "update_t3d_p": T3D}


def routine(name):
    r = name[9]
    return Routine(name, TARGET[name], "abc", "ijk", ARGS[r], SHAPES[r], INTERMEDIATES[name],
                   [s for _, s in ROUTINES[name]], MOMENTS[name],
                   update_comment="! Update t3" if r == "c" else "! Update t3 vector",
                   moment_ind="   " if r == "c" else "    ", init_grouped=(r == "a"))


CCSDT_ROUTINES = {name: routine(name) for name in ROUTINES}
