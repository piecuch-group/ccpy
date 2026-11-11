"""T4 contributions to the (dense) T2 equations: sparse T4 -> dense x2 scatter.

Each term of ccsdtq.update_t2? of the form  dT.<s> += coef * einsum("mnef,<t4>-><t2>", H0.xx.oovv, T.<t4>)
is evaluated exactly as the dense einsum, but over the stored T4 rows: every row stands for the
elements obtained by permuting its indices within each spin block (with the permutation sign), so
the routine adds coef * sign * t4 * v for all of them. The result is the same pre-antisymmetrized
x2 the dense code passes to the T2 update.
"""
from itertools import permutations, product

from codegen import IND, parity
from dense_terms import parse, T4
from block_terms import blocks_of, opname_f, SPINS, DIM

T2N = {"aa": "t2a", "ab": "t2b", "bb": "t2c"}


def t4_terms(func):
    return [t for t in parse(func) if len(t.ops) == 2 and t.ops[1].startswith("T.") and len(t.ops[1].split(".")[1]) == 4]


def vt3_terms(path=None):
    """The H0*T3 terms of hbar_ccsdt.add_VT3_intermediates (UHF branch) as (target, Term)."""
    import ast
    from pathlib import Path
    from block_terms import sym, _dotted
    from dense_terms import Term
    path = path or Path(__file__).resolve().parents[2] / "ccpy/hbar/hbar_ccsdt.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add_VT3_intermediates")
    out = []

    def walk(body):
        for st in body:
            if isinstance(st, ast.If):
                walk(st.orelse)       # flag_RHF False
            elif isinstance(st, ast.AugAssign):
                tgt = _dotted(st.target)
                s_ = 1.0 if isinstance(st.op, ast.Add) else -1.0
                for c, t in sym(st.value):
                    out.append((tgt, Term(tgt, s_ * c, t[1], t[2])))
    walk(fn.body)
    return out


def scatter_routine(func, name, terms=None, target=None):
    """Sparse T3/T4 -> dense scatter. Either the T4 terms of ccsdtq.<func> into x2, or the
    given (target name, Term) pairs, e.g. the VT3 terms into the H(vooo)/H(vvov) arrays."""
    if terms is None:
        terms = [("x2", t) for t in t4_terms(func)]
    tnames = []
    for tg, t in terms:
        if tg not in tnames:
            tnames.append(tg)
    dense_dims = {}
    for tg, t in terms:
        if tg == "x2":
            sps = SPINS[t.target]
            dense_dims[tg] = [DIM[("v", sps[0])], DIM[("v", sps[1])], DIM[("o", sps[0])], DIM[("o", sps[1])]]
        else:   # an H block, e.g. H.aa.vooo, natural layout
            parts = tg.split(".")
            dense_dims[tg] = [DIM[(k, s)] for k, s in zip(parts[2], SPINS[parts[1]])]
    fname = {tg: ("x2" if tg == "x2" else opname_f(tg)) for tg in tnames}
    sources, ops = [], []
    for _, t in terms:
        sp = t.ops[1].split(".")[1]
        if sp not in sources:
            sources.append(sp)
        if t.ops[0] not in ops:
            ops.append(t.ops[0])
    AMP = {**T4, "aaa": "t3a", "aab": "t3b", "abb": "t3c", "bbb": "t3d"}
    args = [(fname[tg], "dense", dense_dims[tg], tg) for tg in tnames]
    for sp in sources:
        args += [(f"{AMP[sp]}_amps", "amps", [f"n{len(sp)}{sp}"], sp),
                 (f"{AMP[sp]}_excits", "excits", [f"n{len(sp)}{sp}", str(2 * len(sp))], sp)]
    for o in ops:
        parts = o.split(".")
        args.append((opname_f(o), "op", [DIM[(k, s)] for k, s in zip(parts[2], SPINS[parts[1]])], (o, (0, 1, 2, 3))))
    counts = [f"n{len(sp)}{sp}" for sp in sources]
    allargs = [a[0] for a in args] + counts + ["noa", "nua", "nob", "nub"]
    L = []
    w = L.append
    w(" " * 14 + f"subroutine {name}({allargs[0]},&")
    rest = allargs[1:]
    for n_ in range(0, len(rest), 4):
        w(" " * 38 + ", ".join(rest[n_:n_ + 4]) + (")" if n_ + 4 >= len(rest) else ",&"))
    w(IND + f"integer, intent(in) :: {', '.join(counts + ['noa', 'nua', 'nob', 'nub'])}")
    for nm, kind, dims, extra in args:
        d = ",".join(dims)
        if kind == "dense":
            w(IND + f"real(kind=8), intent(inout) :: {nm}({d})")
            w(IND + f"!f2py intent(in,out) :: {nm}")
        elif kind == "excits":
            w(IND + f"integer, intent(in) :: {nm}({d})")
        else:
            w(IND + f"real(kind=8), intent(in) :: {nm}({d})")
    w(IND + "integer :: idet, r1, r2, r3, r4, r5, r6, r7, r8")
    w(IND + "integer :: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, s, u, v, w, y, z")
    w(IND + "real(kind=8) :: t_amp")
    for tg, t in terms:
        sp = t.ops[1].split(".")[1]
        n = 2 * len(sp)
        (oi, ai), tgt = t.spec.split("->")[0].split(","), t.spec.split("->")[1]
        v = opname_f(t.ops[0])
        x = fname[tg]
        w(IND + f"!!!! {tg} += {t.coef:+.6g} * {t.ops[0]}({oi}) * {t.ops[1]}({ai}) -> {tgt}")
        free = [y for y in oi if y not in ai]   # letters of H not on the amplitude: loop over them
        priv = "".join("," + y for y in free)
        w(IND + f"!$omp parallel do schedule(static) default(shared) private(idet,r1,r2,r3,r4,r5,r6,r7,r8,t_amp{priv}) reduction(+:{x})")
        w(IND + f"do idet = 1, n{len(sp)}{sp}")
        w(IND + "   " + " ".join(f"r{c + 1} = {AMP[sp]}_excits(idet,{c + 1});" for c in range(n)))
        w(IND + f"   t_amp = {AMP[sp]}_amps(idet)")
        blocks = [cols for _, _, cols in blocks_of(sp)]
        acc = {}
        for perms in product(*[permutations(cols) for cols in blocks]):
            pos = {}
            sign = 1
            for cols, pc in zip(blocks, perms):
                sign *= parity([cols.index(c) for c in pc])
                for axis, c in zip(cols, pc):
                    pos[axis] = c          # axis `axis` of the dense amplitude holds row value r(c)
            val = {ai[axis]: f"r{pos[axis] + 1}" for axis in range(n)}
            key = (",".join(val.get(y, y) for y in tgt), ",".join(val.get(y, y) for y in oi))
            acc[key] = acc.get(key, 0) + sign
        for y in free:
            k_ = dict(zip(oi, t.ops[0].split(".")[2]))[y]
            s_ = dict(zip(oi, SPINS[t.ops[0].split(".")[1]]))[y]
            w(IND + f"   do {y} = 1, {DIM[(k_, s_)]}")
        for (xi, vi), cnt in acc.items():
            if cnt:
                c = t.coef * cnt
                w(IND + f"   {x}({xi}) = {x}({xi}) + ({c:.16g}d0 * t_amp) * {v}({vi})")
        for y in free:
            w(IND + "   end do")
        w(IND + "end do")
        w(IND + "!$omp end parallel do")
    w(" " * 14 + f"end subroutine {name}")
    return "\n".join(L) + "\n", args, counts
