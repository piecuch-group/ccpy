"""Tests of the CC(P) loop generator.

  * the generated CCSDT(P) blocks match ccsdt_p_loops.f90
  * the generated code is numerically right against numpy einsum, with dense
    (idx_table) and sparse (sort_pspace) index tables

run: OMP_NUM_THREADS=4 ~/miniforge3/envs/ccpy/bin/python -m pytest -q devtools/ccp_codegen
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from build import build  # noqa: E402
from check_ccsdt import HAND, compare  # noqa: E402
from specs_ccsdt import CCSDT_SPECS, T3A_D5, T3A_D6  # noqa: E402
import dataclasses  # noqa: E402


def perm_sign(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def antisym(x, groups):
    """Sum over signed permutations within each group of axes."""
    out = np.zeros_like(x)
    for perms in itertools.product(*[itertools.permutations(g) for g in groups]):
        axes = list(range(x.ndim))
        sign = 1
        for g, p in zip(groups, perms):
            sign *= perm_sign([g.index(q) for q in p])
            for a, b in zip(g, p):
                axes[a] = b
        out += sign * x.transpose(axes)
    return out


def random_pspace(rng, nv, no, spins, frac):
    """Random selection of excitations (1-based columns) with the given spin layout."""
    def tuples(n, spin):
        out = []
        groups = [(s, spin.count(s)) for s in sorted(set(spin), key=spin.index)]
        parts = [list(itertools.combinations(range(1, n[s] + 1), c)) for s, c in groups]
        for combo in itertools.product(*parts):
            out.append(sum(combo, ()))
        return out
    rows = [v + o for v in tuples(nv, spins[0]) for o in tuples(no, spins[1])]
    keep = rng.random(len(rows)) < frac
    rows = np.array([r for r, k in zip(rows, keep) if k], dtype=np.int32)
    if len(rows) == 0:   # the empty-P sentinel of the Fortran code: one all-ones row, zero amplitude
        return np.ones((1, sum(len(x) for x in spins)), dtype=np.int32), np.zeros(1)
    return rows, rng.standard_normal(len(rows))


@pytest.mark.parametrize("name", list(CCSDT_SPECS))
def test_ccsdt_blocks_match_hand_code(name):
    res, _ = compare(name, CCSDT_SPECS[name], HAND.read_text().splitlines())
    assert not res.startswith("DIFF"), res


T3_WRAPPER = """module gen_t3
  use omp_lib
  use pspace_sort, only: get_index_table, sort4, sort_pspace, pspace_find, pspace_key
  implicit none
contains
{routines}
end module gen_t3
"""

T3_ROUTINE = """  subroutine {name}(resid, t3a_amps, t3a_excits, t3b_amps, t3b_excits, h2a_voov, h2b_voov, &
                    n3aaa, n3aab, noa, nua, nob, nub)
                  integer, intent(in) :: noa, nua, nob, nub, n3aaa, n3aab
                  integer, intent(inout) :: t3a_excits(n3aaa,6)
                  !f2py intent(in,out) :: t3a_excits(0:n3aaa-1,0:5)
                  real(kind=8), intent(inout) :: t3a_amps(n3aaa)
                  !f2py intent(in,out) :: t3a_amps(0:n3aaa-1)
                  integer, intent(in) :: t3b_excits(n3aab,6)
                  real(kind=8), intent(in) :: t3b_amps(n3aab)
                  real(kind=8), intent(in) :: h2a_voov(noa,nua,nua,noa), h2b_voov(nob,nub,nua,noa)
                  real(kind=8), intent(out) :: resid(n3aaa)
                  integer, allocatable :: idx_table(:,:,:,:), loc_arr(:,:), block_off(:)
                  integer(kind=8), allocatable :: block_key(:)
                  real(kind=8), allocatable :: t3_amps_buff(:)
                  integer, allocatable :: t3_excits_buff(:,:)
                  integer :: kdims(6), nblock
                  logical :: kchain(6)
                  real(kind=8) :: hmatel
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet, idx, nloc
                  resid = 0.0d0
{body}
  end subroutine {name}
"""


def t3_source():
    routines = []
    for mode in ("dense", "sparse"):
        body = "".join(dataclasses.replace(s, index=mode).emit() for s in (T3A_D5, T3A_D6))
        routines.append(T3_ROUTINE.format(name=f"voov_{mode}", body=body))
    return T3_WRAPPER.format(routines="\n".join(routines))


@pytest.fixture(scope="module")
def t3mod(tmp_path_factory):
    return build("gen_t3", t3_source(), ["voov_dense", "voov_sparse"], tmp_path_factory.mktemp("gen_t3"))


def dense_from_rows(rows, amps, shape, groups):
    x = np.zeros(shape)
    for r, t in zip(rows, amps):
        x[tuple(r - 1)] = t
    return antisym(x, groups)


@pytest.mark.parametrize("frac", [1.0, 0.4])
@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_t3_voov_numerics(t3mod, mode, frac):
    rng = np.random.default_rng(7)
    noa, nua, nob, nub = 4, 6, 3, 7
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    t3a_x, t3a = random_pspace(rng, nv, no, ("aaa", "aaa"), frac)
    t3b_x, t3b = random_pspace(rng, nv, no, ("aab", "aab"), frac)
    ha = rng.standard_normal((nua, noa, noa, nua))  # h2a(c,m,k,e)
    hb = rng.standard_normal((nua, nob, noa, nub))  # h2b(c,m~,k,e~)
    T3a = dense_from_rows(t3a_x, t3a, (nua,) * 3 + (noa,) * 3, [(0, 1, 2), (3, 4, 5)])
    T3b = dense_from_rows(t3b_x, t3b, (nua, nua, nub, noa, noa, nob), [(0, 1), (3, 4)])
    X = np.einsum("cmke,abeijm->abcijk", ha, T3a) + np.einsum("cmke,abeijm->abcijk", hb, T3b)
    R = antisym(X, [(0, 1, 2), (3, 4, 5)]) / 4.0
    fn = getattr(t3mod, f"voov_{mode}")
    resid, amps, ex = fn(t3a, np.asfortranarray(t3a_x), t3b, np.asfortranarray(t3b_x),
                         np.asfortranarray(ha.transpose(1, 3, 0, 2)), np.asfortranarray(hb.transpose(1, 3, 0, 2)))
    ref = np.array([R[tuple(r - 1)] for r in ex])
    assert np.allclose(resid, ref, atol=1e-11), np.abs(resid - ref).max()


# ------------------------------------------------------------------ T4 vvooov term
from specs_ccsdtq import T4A_VVOOOV, t4a_vvooov_routine  # noqa: E402



@pytest.fixture(scope="module")
def t4mod(tmp_path_factory):
    src = "module gen_t4\n  use omp_lib\n  use pspace_sort, only: get_index_table, sort4\n  implicit none\ncontains\n" \
        + t4a_vvooov_routine() + "end module gen_t4\n"
    return build("gen_t4", src, ["t4a_vvooov"], tmp_path_factory.mktemp("gen_t4"))


def ccsdtq_reference(H, T2a, T3a, T3b):
    """The I3A/I3B_vvooov lines of ccsdtq.update_t4a, with update_t4a's 576-term antisymmetrizer."""
    e = np.einsum
    I3A = (-0.5 * e("nmke,cdnl->cdmkle", H["aa_ooov"], T2a) + 0.5 * e("cmfe,fdkl->cdmkle", H["aa_vovv"], T2a)
           + 0.125 * e("mnef,cdfkln->cdmkle", H["aa_oovv"], T3a) + 0.25 * e("mnef,cdfkln->cdmkle", H["ab_oovv"], T3b))
    I3A = I3A - I3A.transpose(0, 1, 2, 4, 3, 5)
    I3A = I3A - I3A.transpose(1, 0, 2, 3, 4, 5)
    I3B = (-0.5 * e("nmke,cdnl->cdmkle", H["ab_ooov"], T2a) + 0.5 * e("cmfe,fdkl->cdmkle", H["ab_vovv"], T2a)
           + 0.125 * e("mnef,cdfkln->cdmkle", H["bb_oovv"], T3b))
    I3B = I3B - I3B.transpose(1, 0, 2, 3, 4, 5)
    I3B = I3B - I3B.transpose(0, 1, 2, 4, 3, 5)
    X = (36.0 / 576.0) * (e("cdmkle,abeijm->abcdijkl", I3A, T3a, optimize=True)
                          + e("cdmkle,abeijm->abcdijkl", I3B, T3b, optimize=True))
    return antisym(X, [(0, 1, 2, 3), (4, 5, 6, 7)])


@pytest.mark.parametrize("frac3,frac4", [(1.0, 1.0), (0.4, 0.5)])
def test_t4a_vvooov_numerics(t4mod, frac3, frac4):
    rng = np.random.default_rng(11)
    noa, nua, nob, nub = 5, 7, 4, 6
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    t4_x, t4 = random_pspace(rng, nv, no, ("aaaa", "aaaa"), frac4)
    t3a_x, t3a = random_pspace(rng, nv, no, ("aaa", "aaa"), frac3)
    t3b_x, t3b = random_pspace(rng, nv, no, ("aab", "aab"), frac3)
    T2a = antisym(rng.standard_normal((nua, nua, noa, noa)), [(0, 1), (2, 3)])
    T3a = dense_from_rows(t3a_x, t3a, (nua,) * 3 + (noa,) * 3, [(0, 1, 2), (3, 4, 5)])
    T3b = dense_from_rows(t3b_x, t3b, (nua, nua, nub, noa, noa, nob), [(0, 1), (3, 4)])
    r = rng.standard_normal
    H = {"aa_ooov": r((noa, noa, noa, nua)), "aa_vovv": r((nua, noa, nua, nua)), "aa_oovv": r((noa, noa, nua, nua)),
         "ab_oovv": r((noa, nob, nua, nub)), "ab_ooov": r((noa, nob, noa, nub)), "ab_vovv": r((nua, nob, nua, nub)),
         "bb_oovv": r((nob, nob, nub, nub))}
    R = ccsdtq_reference(H, T2a, T3a, T3b)
    F = np.asfortranarray
    args = [F(H["aa_ooov"].transpose(3, 1, 0, 2)),   # (e,m,n,k) <- (n,m,k,e)
            F(H["aa_vovv"].transpose(3, 1, 2, 0)),   # (e,m,f,c) <- (c,m,f,e)
            F(H["aa_oovv"].transpose(2, 0, 3, 1)),   # (e,m,f,n) <- (m,n,e,f)
            F(H["ab_ooov"].transpose(3, 1, 0, 2)),
            F(H["ab_vovv"].transpose(3, 1, 2, 0)),
            F(H["ab_oovv"].transpose(2, 0, 3, 1)),
            F(H["bb_oovv"].transpose(2, 0, 3, 1))]
    # argument order of the routine: aa_ooov, aa_vovv, aa_oovv, ab_ooov, ab_vovv, ab_oovv, bb_oovv
    resid, amps, ex = t4mod.t4a_vvooov(t4, F(t4_x), t3a, F(t3a_x), t3b, F(t3b_x), F(T2a), *args)
    ref = np.array([R[tuple(x - 1)] for x in ex])
    assert np.abs(ref).max() > 1e-3
    assert np.allclose(resid, ref, atol=1e-10), np.abs(resid - ref).max()


# ------------------------------------------- generated update_t3?_p vs hand code, bit for bit
from specs_ccsdt import CCSDT_ROUTINES, ARGS, SHAPES  # noqa: E402

GEN_CCSDT = """module gen_ccsdt_p
  use omp_lib
  use pspace_sort, only: get_index_table, sort4
  use ccsdt_p_loops, only: calc_I2A_vooo, calc_I2A_vvov, calc_I2B_vooo, calc_I2B_ovoo, &
                           calc_I2B_vvov, calc_I2B_vvvo, calc_I2C_vooo, calc_I2C_vvov
  implicit none
contains
{routines}
end module gen_ccsdt_p
"""


@pytest.fixture(scope="module")
def ccsdt_mods(tmp_path_factory):
    src = GEN_CCSDT.format(routines="\n".join(r.emit() for r in CCSDT_ROUTINES.values()))
    names = list(CCSDT_ROUTINES)
    mod = build("gen_ccsdt_p", src, names, tmp_path_factory.mktemp("gen_ccsdt"),
                extra=[Path(__file__).resolve().parents[2] / "ccpy/lib/fortran/ccsdt_p_loops.f90"])
    import importlib
    top = importlib.import_module("gen_ccsdt_p")
    return top.ccsdt_p_loops, top.gen_ccsdt_p


@pytest.mark.parametrize("frac", [1.0, 0.5])
@pytest.mark.parametrize("name", list(CCSDT_ROUTINES))
def test_generated_update_t3_bitwise(ccsdt_mods, name, frac):
    hand, gen = ccsdt_mods
    rng = np.random.default_rng(3)
    noa, nua, nob, nub = 4, 6, 3, 5
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    t3 = {x: random_pspace(rng, nv, no, (sp, sp), frac) for x, sp in zip("abcd", ("aaa", "aab", "abb", "bbb"))}
    r = name[9]
    kw = {}
    for a in ARGS[r]:
        if a in ("resid", "noa", "nua", "nob", "nub") or a.startswith("n3"):
            continue
        if a == "shift":
            kw[a] = 0.1
        elif a.endswith("_excits"):
            kw[a] = np.asfortranarray(t3[a[2]][0])
        elif a.endswith("_amps"):
            kw[a] = t3[a[2]][1].copy()
        else:
            kw[a] = np.asfortranarray(rng.standard_normal([dims[d] for d in SHAPES[r][a].split(",")]))
    out_h = getattr(hand, name)(**{k.lower(): (v.copy(order="F") if isinstance(v, np.ndarray) else v) for k, v in kw.items()})
    out_g = getattr(gen, name)(**{k.lower(): (v.copy(order="F") if isinstance(v, np.ndarray) else v) for k, v in kw.items()})
    for x, y in zip(out_h, out_g):
        assert np.array_equal(x, y)
    assert np.abs(out_h[0]).max() > 0


@pytest.mark.parametrize("name", list(CCSDT_ROUTINES))
def test_ccsdt_routines_match_hand_code(name):
    from check_ccsdt import compare_routine
    diffs, _ = compare_routine(name, HAND.read_text().splitlines())
    assert not diffs, diffs


# ------------------------------------------------ CCSDTQ(P) linear terms vs dense ccsdtq.py
from t4_routines import linear_routine  # noqa: E402
from dense_terms import parse  # noqa: E402

T4_FUNCS = ["update_t4a", "update_t4b", "update_t4c", "update_t4d", "update_t4e"]
PAIRS = {"oooo": [(0, 1), (2, 3)], "vvvv": [(0, 1), (2, 3)], "oovv": [(0, 1), (2, 3)], "vvov": [(0, 1)],
         "vooo": [(2, 3)], "ooov": [(0, 1)], "vovv": [(2, 3)]}


def random_h(rng, name, dims):
    """Random H with the antisymmetry of its same-spin block, e.g. H.aa.oooo."""
    _, sp, kinds = name.split(".")
    spins = sp * 2 if len(sp) == 1 else {"aa": "aaaa", "ab": "abab", "bb": "bbbb"}[sp]
    shape = [dims[("nu" if k == "v" else "no") + s] for k, s in zip(kinds, spins)]
    x = rng.standard_normal(shape)
    if sp in ("aa", "bb"):
        for p in PAIRS.get(kinds, []):
            x = x - x.swapaxes(*p)
    return x


T3_FUNCS = ["update_t3a", "update_t3b", "update_t3c", "update_t3d"]


@pytest.fixture(scope="module")
def t4lin(tmp_path_factory):
    texts, meta = [], {}
    for f in T4_FUNCS + T3_FUNCS:
        txt, args, ncount = linear_routine(f, f"{f[7:10]}_p_linear", source_rank=4 if f in T3_FUNCS else None)
        texts.append(txt)
        meta[f] = (args, ncount)
    src = "module gen_t4lin\n  use omp_lib\n  use pspace_sort\n  implicit none\ncontains\n" + "\n".join(texts) + "end module gen_t4lin\n"
    mod = build("gen_t4lin", src, [f"{f[7:10]}_p_linear" for f in T4_FUNCS + T3_FUNCS], tmp_path_factory.mktemp("gen_t4lin"))
    return mod, meta


SPIN_OF = {"t3a": "aaa", "t3b": "aab", "t3c": "abb", "t3d": "bbb",
           "t4a": "aaaa", "t4b": "aaab", "t4c": "aabb", "t4d": "abbb", "t4e": "bbbb"}


def spin_groups(spin):
    n = len(spin)
    groups = []
    for s in "ab":
        g = [p for p in range(n) if spin[p] == s]
        if len(g) > 1:
            groups += [tuple(g), tuple(p + n for p in g)]
    return groups


@pytest.mark.parametrize("frac", [1.0, 0.45])
@pytest.mark.parametrize("func", T4_FUNCS + T3_FUNCS)
def test_t4_linear_vs_ccsdtq(t4lin, func, frac):
    mod, meta = t4lin
    args, ncount = meta[func]
    rng = np.random.default_rng(5)
    noa, nua, nob, nub = 4, 5, 4, 5
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    sparse, dense = {}, {}
    for key, sp in SPIN_OF.items():
        x, t = random_pspace(rng, nv, no, (sp, sp), frac)
        sparse[key] = (x, t)
        shape = [nv[s] for s in sp] + [no[s] for s in sp]
        dense[key] = dense_from_rows(x, t, shape, spin_groups(sp))
    terms = [t for t in parse(func) if t.is_linear() and (func not in T3_FUNCS or len(t.ops[1].split(".")[1]) == 4)]
    H = {}
    for t in terms:
        if t.ops[0] not in H:
            H[t.ops[0]] = random_h(rng, t.ops[0], dims)
    tsp = terms[0].target
    dT = 0
    for t in terms:
        T = dense[{v: k for k, v in SPIN_OF.items()}[t.ops[1].split(".")[1]]]
        dT = dT + t.coef * np.einsum(t.spec, H[t.ops[0]], T, optimize=True)
    R = antisym(dT, spin_groups(tsp))
    from dense_terms import op_name
    hname = {op_name(k).upper()[:3] + op_name(k)[3:]: k for k in H}
    F = np.asfortranarray
    call = []
    for a in args:
        if a.kind == "resid":
            continue
        if a.kind in ("amps", "excits"):
            x, t = sparse[a.name[:3]]
            call.append(t.copy() if a.kind == "amps" else F(x))
        elif a.extra[1] == "cf":
            h = H[hname[a.name]]
            call.append(F(h if a.extra[0] == "oo" else h.T))
        else:
            call.append(F(np.transpose(H[hname[a.name]], a.extra[1])))
    resid, amps, ex = getattr(mod, f"{func[7:10]}_p_linear")(*call)
    ref = np.array([R[tuple(r - 1)] for r in ex])
    assert np.abs(ref).max() > 1e-6
    assert np.allclose(resid, ref, atol=1e-10), np.abs(resid - ref).max()


# ------------------------------------------- CCSDTQ(P) intermediate x T3 terms vs ccsdtq.py
import ast as _ast  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from block_terms import block_routine, parse_contractions, CCSDTQ as _CCSDTQ  # noqa: E402

T2_PATTERN = (r"einsum\('[a-z]+,[a-z]+->[a-z]+', I3\w+, T\.[ab]{2},|"
              r"einsum\('[a-z]+,[a-z]+,[a-z]+->[a-z]+', H\.\w+\.\w+, T\.[ab]{2}, T\.[ab]{2},")


def dense_reference_blocks(func, ns, only=None, pattern=r"einsum\('[a-z]+,[a-z]+->[a-z]+', I3\w+, T\.[ab]{3},"):
    """Execute the intermediate builds and the I x T3 contraction lines of ccsdtq.<func>."""
    tree = _ast.parse(_CCSDTQ.read_text())
    fn = next(n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == func)
    keep, nterm = [], -1
    for st in fn.body:
        src = _ast.unparse(st)
        tgt = st.targets[0] if isinstance(st, _ast.Assign) else getattr(st, "target", None)
        if isinstance(tgt, _ast.Name) and tgt.id.startswith("I"):
            keep.append(src)
        elif isinstance(tgt, _ast.Attribute) and src.startswith("dT."):
            if re.search(pattern, src):
                nterm += 1
                if only is not None and nterm not in only:
                    continue
                keep.append("out " + src.split(" ", 1)[1].replace("= ", "+= ", 1) if " = " in src.split("einsum")[0] else
                            "out " + src.split(" ", 1)[1])
    ns["out"] = 0.0
    exec("\n".join(keep), ns)
    return ns["out"]


@pytest.fixture(scope="module")
def t4blk(tmp_path_factory):
    texts, meta = [], {}
    for f in T4_FUNCS:
        txt, args, counts = block_routine(f, f"{f[7:10]}_p_t3blocks", kinds=("t3",))
        texts.append(txt)
        meta[f] = args
    src = "module gen_t4blk\n  use omp_lib\n  use pspace_sort\n  implicit none\ncontains\n" + "\n".join(texts) + "end module gen_t4blk\n"
    mod = build("gen_t4blk", src, [f"{f[7:10]}_p_t3blocks" for f in T4_FUNCS], tmp_path_factory.mktemp("gen_t4blk"))
    return mod, meta


@pytest.mark.parametrize("frac3,frac4", [(1.0, 1.0), (0.45, 0.5)])
@pytest.mark.parametrize("func", T4_FUNCS)
def test_t4_t3blocks_vs_ccsdtq(t4blk, func, frac3, frac4):
    check_blocks(t4blk, func, frac3, frac4, "t3blocks", None)


def check_blocks(fix, func, frac3, frac4, suffix, pattern):
    mod, args = fix[0], fix[1][func]
    rng = np.random.default_rng(9)
    noa, nua, nob, nub = 4, 5, 4, 5
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    sparse, dense = {}, {}
    for key, sp in SPIN_OF.items():
        x, t = random_pspace(rng, nv, no, (sp, sp), frac4 if len(sp) == 4 else frac3)
        sparse[key] = (x, t)
        dense[sp] = dense_from_rows(x, t, [nv[s] for s in sp] + [no[s] for s in sp], spin_groups(sp))
    for sp in ("aa", "ab", "bb"):
        x = rng.standard_normal([nv[sp[0]], nv[sp[1]], no[sp[0]], no[sp[1]]])
        dense[sp] = antisym(x, [(0, 1), (2, 3)]) if sp != "ab" else x
    Hd = {}

    def hget(name):
        if name not in Hd:
            Hd[name] = random_h(rng, name, dims)
        return Hd[name]

    class Lazy(SimpleNamespace):
        def __init__(self, prefix):
            super().__init__()
            self._p = prefix

        def __getattr__(self, sp):
            if sp.startswith("_"):
                raise AttributeError(sp)
            return SimpleNamespace(**{b: hget(f"{self._p}.{sp}.{b}") for b in
                                      ("oo", "vv", "ov", "vo") if len(sp) == 1} | _Blocks(self._p, sp))

    def _Blocks(p, sp):
        if len(sp) == 1:
            return {}
        return {k: hget(f"{p}.{sp}.{k}") for k in
                ["".join(x) for x in itertools.product("ov", repeat=4)]}

    H, H0 = Lazy("H"), Lazy("H0")
    T = SimpleNamespace(**dense)
    ns = {"np": np, "H": H, "H0": H0, "T": T}
    dT = dense_reference_blocks(func, ns) if pattern is None else dense_reference_blocks(func, ns, pattern=pattern)
    tsp = parse_contractions(func)[0].target if pattern is None else parse_contractions(func, kinds=("t2", "t2t2"))[0].target
    R = antisym(dT, spin_groups(tsp))
    F = np.asfortranarray
    call = []
    for nm, kind, dd, extra in args:
        if kind == "resid":
            call.append(np.zeros(len(sparse[nm[:3]][1]) if False else len(sparse[args[1][0][:3]][1])))
        elif kind in ("amps", "excits"):
            x, t = sparse[nm[:3]]
            call.append(t.copy() if kind == "amps" else F(x))
        elif kind == "t2":
            call.append(F(dense[extra]))
        else:
            opn, perm = extra
            call.append(F(np.transpose(hget(opn), perm)))
    resid, amps, ex = getattr(mod, f"{func[7:10]}_p_{suffix}")(*call, noa=noa, nua=nua, nob=nob, nub=nub)
    ref = np.array([R[tuple(r - 1)] for r in ex])
    assert np.abs(ref).max() > 1e-6
    assert np.allclose(resid, ref, atol=1e-9), np.abs(resid - ref).max()


@pytest.fixture(scope="module")
def t4blk2(tmp_path_factory):
    texts, meta = [], {}
    for f in T4_FUNCS:
        txt, args, counts = block_routine(f, f"{f[7:10]}_p_t2blocks", kinds=("t2", "t2t2"))
        texts.append(txt)
        meta[f] = args
    src = "module gen_t4blk2\n  use omp_lib\n  use pspace_sort\n  implicit none\ncontains\n" + "\n".join(texts) + "end module gen_t4blk2\n"
    mod = build("gen_t4blk2", src, [f"{f[7:10]}_p_t2blocks" for f in T4_FUNCS], tmp_path_factory.mktemp("gen_t4blk2"))
    return mod, meta


@pytest.mark.parametrize("frac3,frac4", [(1.0, 1.0), (0.45, 0.5)])
@pytest.mark.parametrize("func", T4_FUNCS)
def test_t4_t2blocks_vs_ccsdtq(t4blk2, func, frac3, frac4):
    check_blocks(t4blk2, func, frac3, frac4, "t2blocks", T2_PATTERN)


# ------------------------------------------------ T4 -> T2 scatter vs ccsdtq.py
from scatter_terms import scatter_routine, t4_terms  # noqa: E402

T2_FUNCS = ["update_t2a", "update_t2b", "update_t2c"]


@pytest.fixture(scope="module")
def t2sc(tmp_path_factory):
    texts, meta = [], {}
    for f in T2_FUNCS:
        txt, args, counts = scatter_routine(f, f"{f[7:10]}_p_t4")
        texts.append(txt)
        meta[f] = args
    src = "module gen_t2sc\n  use omp_lib\n  implicit none\ncontains\n" + "\n".join(texts) + "end module gen_t2sc\n"
    return build("gen_t2sc", src, [f"{f[7:10]}_p_t4" for f in T2_FUNCS], tmp_path_factory.mktemp("gen_t2sc")), meta


@pytest.mark.parametrize("frac", [1.0, 0.45])
@pytest.mark.parametrize("func", T2_FUNCS)
def test_t4_to_t2_vs_ccsdtq(t2sc, func, frac):
    mod, meta = t2sc
    rng = np.random.default_rng(13)
    noa, nua, nob, nub = 4, 5, 3, 6
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    sparse, dense = {}, {}
    for key, sp in SPIN_OF.items():
        if len(sp) != 4:
            continue
        x, t = random_pspace(rng, nv, no, (sp, sp), frac)
        sparse[key] = (x, t)
        dense[sp] = dense_from_rows(x, t, [nv[s] for s in sp] + [no[s] for s in sp], spin_groups(sp))
    ref, H = 0.0, {}
    for t in t4_terms(func):
        if t.ops[0] not in H:
            H[t.ops[0]] = random_h(rng, t.ops[0], dims)
        ref = ref + t.coef * np.einsum(t.spec, H[t.ops[0]], dense[t.ops[1].split(".")[1]], optimize=True)
    from block_terms import opname_f
    hn = {opname_f(k): k for k in H}
    F = np.asfortranarray
    call = []
    for nm, kind, dd, extra in meta[func]:
        if kind == "dense":
            call.append(F(np.zeros(ref.shape)))
        elif kind in ("amps", "excits"):
            x, t = sparse[nm[:3]]
            call.append(t.copy() if kind == "amps" else F(x))
        else:
            call.append(F(H[hn[nm]]))
    x2 = getattr(mod, f"{func[7:10]}_p_t4")(*call, noa=noa, nua=nua, nob=nob, nub=nub)
    assert np.abs(ref).max() > 1e-6
    assert np.allclose(x2, ref, atol=1e-11), np.abs(x2 - ref).max()


# ------------------------------------------------ VT3 dressing of H vs hbar_ccsdt.add_VT3_intermediates
from scatter_terms import vt3_terms  # noqa: E402


@pytest.mark.parametrize("frac", [1.0, 0.45])
def test_vt3_scatter(tmp_path_factory, frac):
    terms = vt3_terms()
    txt, args, counts = scatter_routine(None, "vt3", terms=terms)
    mod = build("gen_vt3", "module gen_vt3\n  use omp_lib\n  implicit none\ncontains\n" + txt + "end module gen_vt3\n",
                ["vt3"], tmp_path_factory.mktemp("gen_vt3"))
    rng = np.random.default_rng(17)
    noa, nua, nob, nub = 4, 5, 3, 6
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    sparse, dense = {}, {}
    for key, sp in SPIN_OF.items():
        if len(sp) == 3:
            x, t = random_pspace(rng, nv, no, (sp, sp), frac)
            sparse[key] = (x, t)
            dense[sp] = dense_from_rows(x, t, [nv[s] for s in sp] + [no[s] for s in sp], spin_groups(sp))
    H0 = {o: random_h(rng, o, dims) for o in ("H0.aa.oovv", "H0.ab.oovv", "H0.bb.oovv")}
    ref = {}
    for tg, t in terms:
        ref[tg] = ref.get(tg, 0) + t.coef * np.einsum(t.spec, H0[t.ops[0]], dense[t.ops[1].split(".")[1]], optimize=True)
    from block_terms import opname_f
    F = np.asfortranarray
    call = []
    for nm, kind, dd, extra in args:
        if kind == "dense":
            call.append(F(np.zeros(ref[extra].shape)))
        elif kind in ("amps", "excits"):
            x, t = sparse[nm[:3]]
            call.append(t.copy() if kind == "amps" else F(x))
        else:
            call.append(F(H0[extra[0]]))
    out = mod.vt3(*call, noa=noa, nua=nua, nob=nob, nub=nub)
    for (nm, kind, dd, extra), o in zip([a for a in args if a[1] == "dense"], out):
        assert np.allclose(o, ref[extra], atol=1e-11), (extra, np.abs(o - ref[extra]).max())


# ------------------------------------- ccpy/cc/ccsdtq_p.py passes what the generated module expects
def _fortran_signature(src, name):
    i = src.index(f"subroutine {name}(")
    j = src.index(")", i)
    return [a.strip() for a in src[i + len(f"subroutine {name}("):j].replace("&", "").split(",") if a.strip()]


def _python_call(name):
    import ast as a_
    tree = a_.parse((Path(__file__).resolve().parents[2] / "ccpy/cc/ccsdtq_p.py").read_text())
    for node in a_.walk(tree):
        if isinstance(node, a_.Call) and a_.unparse(node.func) == f"ccsdtq_p_loops.{name}":
            return [a_.unparse(x) for x in node.args]
    raise KeyError(name)


def _expected(fortran_arg):
    """The Python expression ccsdtq_p.py passes for a Fortran dummy argument."""
    import re as r_
    m = r_.match(r"t([34])([a-e])_(amps|excits)$", fortran_arg)
    if m:
        sp = {"3": {"a": "aaa", "b": "aab", "c": "abb", "d": "bbb"},
              "4": {"a": "aaaa", "b": "aaab", "c": "aabb", "d": "abbb", "e": "bbbb"}}[m.group(1)][m.group(2)]
        return f"T.{sp}" if m.group(3) == "amps" else f"t{m.group(1)}_excitations['{sp}']"
    if fortran_arg in ("t2a", "t2b", "t2c"):
        return {"t2a": "T.aa", "t2b": "T.ab", "t2c": "T.bb"}[fortran_arg]
    m = r_.match(r"H([12])([ABC])_(\w+)$", fortran_arg)
    if m:
        sp = {"1A": "a", "1B": "b", "2A": "aa", "2B": "ab", "2C": "bb"}[m.group(1) + m.group(2)]
        # the bare <mn||ef> is passed as H0, the rest is HBar
        return f"{'H0' if m.group(3) == 'oovv' else 'H'}.{sp}.{m.group(3)}"
    m = r_.match(r"f([AB])_(oo|vv)$", fortran_arg)
    if m:
        return f"H0.{m.group(1).lower()}.{m.group(2)}"
    return fortran_arg


@pytest.mark.parametrize("name", [f"update_t4{x}_p" for x in "abcde"])
def test_ccsdtq_p_calls_match_fortran(name):
    from write_ccsdtq import TARGET
    sig = _fortran_signature(TARGET.read_text(), name)
    sig = [a for a in sig if a != "resid" and not a.startswith(("n3", "n4")) and a not in ("noa", "nua", "nob", "nub")]
    assert _python_call(name) == [_expected(a) for a in sig]
