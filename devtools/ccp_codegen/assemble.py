"""Assemble ccpy/lib/fortran/ccsdtq_p_loops.f90.

The module is ccsdt_p_loops.f90 (update_t1*, calc_I2*) with
  * update_t2a/b/c taking the T4 lists and adding <ijab|(V_N T4)_C|0> to the residual,
  * update_t3?_p replaced by the generated CCSDTQ versions (CCSDT blocks + T4 -> T3 blocks),
  * add_vt3_p: the (V_N T3)_C dressing of Hbar used by the T4 equations,
  * update_t4?_p: one entry point per T4 spin case (natural-layout integrals in, the internal
    layouts are built inside) calling the generated t4?_p_lin / _b3 / _b2 and the T4 update.
Every entry point takes plain arrays in the order printed by python_call() (used to write
ccpy/cc/ccsdtq_p.py).
"""
import copy
from pathlib import Path

from codegen import IND
from dense_terms import linear_diagrams, amp as amp_of, T4 as T4NAME
from specs_ccsdt import CCSDT_ROUTINES
from t4_routines import linear_routine, op_dims
from block_terms import block_routine, SPINS, DIM
from scatter_terms import scatter_routine, vt3_terms

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / "ccpy/lib/fortran/ccsdt_p_loops.f90"
T4SPINS = ["aaaa", "aaab", "aabb", "abbb", "bbbb"]
T3NAME = {"aaa": "t3a", "aab": "t3b", "abb": "t3c", "bbb": "t3d"}
AMPNAME = {**T4NAME, **T3NAME}


def nat_name(source):
    """Fortran name of an integral in natural layout, as in ccsdt_p_loops: H.aa.voov -> H2A_voov,
    H.a.ov -> H1A_ov, and the bare H0.aa.oovv -> H2A_oovv."""
    _, sp, blk = source.split(".")
    if len(sp) == 1:
        return f"H1{sp.upper()}_{blk}"
    return f"H2{ {'aa': 'A', 'ab': 'B', 'bb': 'C'}[sp]}_{blk}"


def nat_dims(source):
    _, sp, blk = source.split(".")
    return [DIM[(k, s)] for k, s in zip(blk, SPINS[sp])]


def python_source(source):
    """How ccsdtq_p.py passes an integral: H -> Hbar, H0 -> bare Hamiltonian."""
    head, sp, blk = source.split(".")
    return f"{'H' if head == 'H' else 'H0'}.{sp}.{blk}"


def split_args(text):
    return [a.strip() for a in text.replace("&", "").replace("\n", "").split(",") if a.strip()]


# ------------------------------------------------------------------ T2
def t2_with_t4(src, x):
    """Add the T4 lists to update_t2<x> of ccsdt_p_loops and the T4 scatter after resid = x2."""
    f = f"update_t2{x}"
    txt, args, counts = scatter_routine(f, f"t2{x}_p_t4")
    t4 = [a for a in args if a[1] in ("amps", "excits")]
    ops = [a for a in args if a[1] == "op"]
    lines = src.splitlines(keepends=True)
    i = next(n for n, l in enumerate(lines) if f"subroutine {f}(" in l)
    j = next(n for n in range(i, len(lines)) if lines[n].strip() == f"end subroutine {f}")
    body = "".join(lines[i:j + 1])
    head, rest = body.split("shift,&", 1)
    pad = " " * (len(head.splitlines()[-1]))
    new_args = ", ".join(a[0] for a in t4) + ",&\n" + pad + ", ".join(nat_name(o[3][0]) for o in ops) + ",&\n" + pad
    body = head + new_args + "shift,&" + rest
    # declare the new arguments and the T4 counts
    decl = "\n" + IND + f"integer, intent(in) :: {', '.join(counts)}\n"
    for nm, kind, dims, extra in t4:
        decl += IND + f"{'integer' if kind == 'excits' else 'real(kind=8)'}, intent(in) :: {nm}({','.join(dims)})\n"
    for nm, kind, dims, extra in ops:
        decl += IND + f"real(kind=8), intent(in) :: {nat_name(extra[0])}({','.join(dims)})\n"
    k = body.index("\n", body.index("integer, intent(in) :: noa"))
    body = body[:k] + decl.rstrip("\n") + body[k:]
    body = body.replace("noa, nua, nob, nub)", f"{', '.join(counts)},&\n{pad}noa, nua, nob, nub)", 1)
    call = ", ".join(["resid"] + [a[0] for a in t4] + [nat_name(o[3][0]) for o in ops]) + ",&\n" + IND + " " * 19 + \
        ", ".join(counts + ["noa", "nua", "nob", "nub"])
    k = body.index("resid(:,:,:,:) = x2")
    k = body.index("\n", k)
    body = body[:k] + "\n" + IND + "! <ijab|(V_N T4)_C|0>\n" + IND + f"call t2{x}_p_t4({call})" + body[k:]
    out = "".join(lines[:i]) + body + "".join(lines[j + 1:])
    return out, txt, t4, ops


# ------------------------------------------------------------------ T3
def t3_ccsdtq(name):
    """update_t3?_p of CCSDTQ(P): the CCSDT routine plus its T4 -> T3 blocks (natural integrals)."""
    r = copy.deepcopy(CCSDT_ROUTINES[name])
    x = name[9]
    blocks = linear_diagrams(f"update_t3{x}", {}, source_rank=4, natural=True)
    extra, seen = [], set()
    for d in blocks:
        s = d.source
        for nm, typ, dims in ((s.amps, "real", [s.n]), (s.excits, "integer", [s.n, "8"])):
            if nm not in seen:
                seen.add(nm)
                extra.append((nm, typ, dims))
    for d in blocks:
        if d.op.shared not in seen:
            seen.add(d.op.shared)
            extra.append((d.op.shared, "real", op_dims(d.op.name, d.op.name.split("_")[1], d.op.stored)))
    counts = sorted({d.source.n for d in blocks})
    r.args = r.args + [c for c in counts if c not in r.args]
    r.extra_blocks = blocks
    r.extra_args = extra
    return r


# ------------------------------------------------------------------ T4
def t4_entry(sp):
    """update_t4?_p: natural integrals in; layouts built here; lin + b3 + b2 + update."""
    x = T4NAME[sp][2]
    f = f"update_t4{x}"
    lin_txt, lin_args, lin_counts = linear_routine(f, f"t4{x}_p_lin")
    b3_txt, b3_args, b3_counts = block_routine(f, f"t4{x}_p_b3", kinds=("t3",))
    b2_txt, b2_args, b2_counts = block_routine(f, f"t4{x}_p_b2", kinds=("t2", "t2t2"))
    t = amp_of(sp)
    # normalized argument specs of the three parts: (name, kind, info)
    parts = []
    for nm, args, counts in ((f"t4{x}_p_lin", lin_args, lin_counts), (f"t4{x}_p_b3", b3_args, b3_counts),
                             (f"t4{x}_p_b2", b2_args, b2_counts)):
        spec = []
        for a in args:
            if hasattr(a, "name"):   # linear routines
                if a.kind == "op":
                    kinds = a.extra[0]
                    src = "H." + {"1A": "a", "1B": "b", "2A": "aa", "2B": "ab", "2C": "bb"}[a.name[1:3]] + "." + a.name.split("_")[1]
                    perm = (0, 1) if a.extra[1] == "cf" and kinds == "oo" else (1, 0) if a.extra[1] == "cf" else tuple(a.extra[1])
                    spec.append((a.name, "op", (src, perm)))
                else:
                    spec.append((a.name, a.kind, a.extra))
            else:
                nm_, kind, dims, extra = a
                spec.append((nm_, kind, tuple(extra) if kind == "op" else extra))
        parts.append((nm, spec, counts))
    amps, t2s, ops = [], [], []
    for _, spec, _ in parts:
        for nm, kind, info in spec:
            if kind in ("amps", "excits") and nm not in [a[0] for a in amps]:
                amps.append((nm, kind, info))
            elif kind == "t2" and nm not in t2s:
                t2s.append(nm)
            elif kind == "op" and info[0] not in ops:
                ops.append(info[0])
    order = {"H.a": 0, "H.b": 1, "H.aa": 2, "H.ab": 3, "H.bb": 4, "H0.aa": 5, "H0.ab": 6, "H0.bb": 7}
    ops.sort(key=lambda o: (order[".".join(o.split(".")[:2])], o))
    names = [nat_name(o) for o in ops]
    assert len(set(names)) == len(names), names
    others = [a for a in amps if a[0] not in (t.amps, t.excits)]
    rank = {s_: n for n, s_ in enumerate(T4SPINS + list(T3NAME))}
    others.sort(key=lambda a: (rank[a[2]], a[1] == "excits"))
    t4o = [a for a in others if len(a[2]) == 4]
    t3o = [a for a in others if len(a[2]) == 3]
    t2s = [n for n in ("t2a", "t2b", "t2c") if n in t2s]
    counts = [t.n] + sorted({f"n{len(a[2])}{a[2]}" for a in others}, key=lambda c: (c[1], c))
    sig = ["resid", t.amps, t.excits] + [a[0] for a in t4o] + [a[0] for a in t3o] + t2s + names + \
        ["fA_oo", "fA_vv", "fB_oo", "fB_vv", "shift"] + counts + ["noa", "nua", "nob", "nub"]
    L = []
    w = L.append
    nm = f"update_t4{x}_p"
    w(" " * 14 + f"subroutine {nm}({sig[0]},&")
    for n_ in range(1, len(sig), 4):
        w(" " * 38 + ", ".join(sig[n_:n_ + 4]) + (")" if n_ + 4 >= len(sig) else ",&"))
    w("")
    w(IND + f"integer, intent(in) :: {', '.join(counts + ['noa', 'nua', 'nob', 'nub'])}")
    for a, kind, info in t4o + t3o:
        n = f"n{len(info)}{info}"
        w(IND + (f"integer, intent(in) :: {a}({n},{2 * len(info)})" if kind == "excits" else f"real(kind=8), intent(in) :: {a}({n})"))
    t2dims = {"t2a": "nua,nua,noa,noa", "t2b": "nua,nub,noa,nob", "t2c": "nub,nub,nob,nob"}
    for a in t2s:
        w(IND + f"real(kind=8), intent(in) :: {a}({t2dims[a]})")
    for o, n in zip(ops, names):
        w(IND + f"real(kind=8), intent(in) :: {n}({','.join(nat_dims(o))})")
    w(IND + "real(kind=8), intent(in) :: fA_oo(noa,noa), fA_vv(nua,nua), fB_oo(nob,nob), fB_vv(nub,nub), shift")
    w("")
    w(IND + f"integer, intent(inout) :: {t.excits}({t.n},8)")
    w(IND + f"!f2py intent(in,out) :: {t.excits}(0:{t.n}-1,0:7)")
    w(IND + f"real(kind=8), intent(inout) :: {t.amps}({t.n})")
    w(IND + f"!f2py intent(in,out) :: {t.amps}(0:{t.n}-1)")
    w(IND + f"real(kind=8), intent(out) :: resid({t.n})")
    w("")
    # internal layouts
    local = {}
    for _, spec, _ in parts:
        for nm_, kind, info in spec:
            if kind == "op" and tuple(info[1]) != tuple(range(len(info[1]))):
                local.setdefault((info[0], tuple(info[1])), f"{nat_name(info[0])}_{''.join(map(str, info[1]))}".lower())
    for (o, perm), ln in local.items():
        d = nat_dims(o)
        w(IND + f"real(kind=8), allocatable :: {ln}({','.join(':' for _ in d)})")
    w(IND + "integer :: idet, p1, p2, p3, p4, a, b, c, d, i, j, k, l")
    w(IND + "real(kind=8) :: denom")
    w("")
    w(IND + "! integrals in the layouts of the sparse kernels (summed or block indices first)")
    for (o, perm), ln in local.items():
        d = nat_dims(o)
        sd = [d[p] for p in perm]
        idx = ["p1", "p2", "p3", "p4"][:len(d)]
        nat_idx = [None] * len(d)
        for k_, p in enumerate(perm):
            nat_idx[p] = idx[k_]
        w(IND + f"allocate({ln}({','.join(sd)}))")
        for k_ in reversed(range(len(d))):
            w(IND + "   " * (len(d) - 1 - k_) + f"do {idx[k_]} = 1, {sd[k_]}")
        w(IND + "   " * len(d) + f"{ln}({','.join(idx)}) = {nat_name(o)}({','.join(nat_idx)})")
        for k_ in range(len(d)):
            w(IND + "   " * (len(d) - 1 - k_) + "end do")
    w("")

    def arg_of(nm_, kind, info):
        if kind == "op":
            return local.get((info[0], tuple(info[1])), nat_name(info[0]))
        return nm_

    for (pn, spec, cnt), label in zip(parts, ("(H(2) * T3 + H(2) * T4)_C linear terms",
                                               "(I(T2, T3) * T3)_C terms", "(I(T2, T4) * T2)_C and (H * T2 * T2)_C terms")):
        w(IND + f"! {label}")
        call = [arg_of(*s) for s in spec] + cnt + ["noa", "nua", "nob", "nub"]
        rows = [", ".join(call[n_:n_ + 6]) for n_ in range(0, len(call), 6)]
        w(IND + f"call {pn}(" + (",&\n" + IND + " " * (6 + len(pn))).join(rows) + ")")
    for (o, perm), ln in local.items():
        w(IND + f"deallocate({ln})")
    w("")
    w(IND + "! Update t4 vector")
    occ = " + ".join(f"f{s.upper()}_oo({y},{y})" for y, s in zip("ijkl", sp))
    vir = " - ".join(f"f{s.upper()}_vv({y},{y})" for y, s in zip("abcd", sp))
    w(IND + "!$omp parallel shared(resid," + t.excits + "," + t.amps + ",fA_oo,fA_vv,fB_oo,fB_vv," + t.n + ",shift),&")
    w(IND + "!$omp private(idet,a,b,c,d,i,j,k,l,denom)")
    w(IND + "!$omp do schedule(static)")
    w(IND + f"do idet = 1, {t.n}")
    w(IND + "    " + " ".join(f"{y} = {t.excits}(idet,{p + 1});" for p, y in enumerate("abcd")))
    w(IND + "    " + " ".join(f"{y} = {t.excits}(idet,{p + 5});" for p, y in enumerate("ijkl")))
    w(IND + f"    denom = {occ} - {vir}")
    w(IND + "    resid(idet) = resid(idet)/(denom - shift)")
    w(IND + f"    {t.amps}(idet) = {t.amps}(idet) + resid(idet)")
    w(IND + "end do")
    w(IND + "!$omp end do")
    w(IND + "!$omp end parallel")
    w("")
    w(" " * 14 + f"end subroutine {nm}")
    text = "\n".join(L) + "\n"
    py = dict(t4o=t4o, t3o=t3o, t2s=t2s, ops=ops)
    kernels = {f"t4{x}_p_lin": lin_txt, f"t4{x}_p_b3": b3_txt, f"t4{x}_p_b2": b2_txt}
    return text, py, kernels


# ------------------------------------------------------------------ module
def kernel_module(name, body):
    """Fortran module ccsdtq_p_<name> holding one T4 kernel (built apart from the f2py module)."""
    return (f"module ccsdtq_p_{name}\n      use pspace_sort\n      implicit none\n      contains\n\n{body}\nend module ccsdtq_p_{name}\n")


def assemble():
    """(Fortran source of module ccsdtq_p_loops, data for writing the Python calls, {kernel name: source of
    its own module}). The T4 kernels t4?_p_lin/_b3/_b2 live in their own modules so they compile in
    parallel and stay out of the f2py wrapper."""
    src = HAND.read_text().replace("module ccsdt_p_loops", "module ccsdtq_p_loops")
    src = src.replace("use pspace_sort, only: get_index_table, sort4\n",
                      "use pspace_sort, only: get_index_table, sort4, sort_pspace, pspace_find, pspace_key, release_sort_workspace\n")
    calls = {}
    extra_routines = []
    for x in "abc":
        src, txt, t4, ops = t2_with_t4(src, x)
        extra_routines.append(txt)
        calls[f"update_t2{x}"] = dict(t4=t4, ops=[o[3][0] for o in ops])
    for name in CCSDT_ROUTINES:
        r = t3_ccsdtq(name)
        lines = src.splitlines(keepends=True)
        i = next(n for n, l in enumerate(lines) if f"subroutine {name}(" in l)
        j = next(n for n in range(i, len(lines)) if lines[n].strip() == f"end subroutine {name}")
        src = "".join(lines[:i]) + r.emit() + "".join(lines[j + 1:])
        calls[name] = r.extra_args
    txt, args, counts = scatter_routine(None, "add_vt3_p", terms=vt3_terms())
    extra_routines.append(txt)
    calls["add_vt3_p"] = args
    kernels = {}
    for sp in T4SPINS:
        text, py, kern = t4_entry(sp)
        extra_routines.append(text)
        kernels.update({k: kernel_module(k, v) for k, v in kern.items()})
        calls[f"update_t4{T4NAME[sp][2]}_p"] = py
    i = src.rindex("end module ccsdtq_p_loops")
    src = src[:i] + "\n".join(extra_routines) + "\n" + src[i:]
    j = src.index("      implicit none\n")
    src = src[:j] + "".join(f"      use ccsdtq_p_{k}\n" for k in kernels) + "\n" + src[j:]
    return src, calls, kernels


if __name__ == "__main__":
    s, c, _ = assemble()
    print(len(s.splitlines()), "lines")
