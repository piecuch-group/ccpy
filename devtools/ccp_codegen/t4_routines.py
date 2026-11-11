"""Assemble generated CCSDTQ(P) routines (Fortran text + their argument interface)."""
from dataclasses import dataclass

from codegen import IND
from dense_terms import linear_diagrams, amp

INT_DIMS = ("noa", "nua", "nob", "nub")


def op_dims(name, natural_kinds, stored):
    """Fortran dims of a stored integral array, e.g. h2b_voov with layout (1,3,0,2)."""
    sp = name[2]
    spins = {"a": "aaaa", "b": "abab", "c": "bbbb"}[sp] if name[1] == "2" else sp * 2
    if name[1] == "1":
        spins = {"a": "aa", "b": "bb"}[sp]
    d = [{("v", "a"): "nua", ("o", "a"): "noa", ("v", "b"): "nub", ("o", "b"): "nob"}[(k, s)]
         for k, s in zip(natural_kinds, spins)]
    return [d[p] for p in stored]


@dataclass
class Arg:
    name: str
    kind: str      # "excits", "amps", "op", "resid"
    dims: list
    extra: object = None   # (spin) for amplitudes, (natural kinds, stored) for operators


def linear_routine(func, name=None, source_rank=None):
    """Subroutine with every linear (H T3/T4) block of ccsdtq.<func>, and its arguments."""
    blocks = linear_diagrams(func, source_rank=source_rank)
    tgt = blocks[0].target
    sources, ops = {}, {}
    for d in blocks:
        if d.source.excits != tgt.excits:
            sources[d.source.excits] = d.source
        ops[d.op.name] = d.op
        if d.h1:
            ops[d.h1.name] = d.h1
    args = [Arg("resid", "resid", [tgt.n]),
            Arg(tgt.amps, "amps", [tgt.n], tgt.spin_v), Arg(tgt.excits, "excits", [tgt.n, str(2 * len(tgt.spin_v))], tgt.spin_v)]
    nexc = str(2 * len(tgt.spin_v))
    for s in sources.values():
        args += [Arg(s.amps, "amps", [s.n], s.spin_v), Arg(s.excits, "excits", [s.n, str(2 * len(s.spin_v))], s.spin_v)]
    for nm, o in ops.items():
        if hasattr(o, "natural"):
            kinds = nm.split("_")[1]
            args.append(Arg(o.shared, "op", op_dims(nm, kinds, o.stored), (kinds, o.stored, o.natural)))
        else:  # h1 delta terms: stored (contracted, free)
            kinds = nm.split("_")[1]
            args.append(Arg(o.shared, "op", op_dims(nm, kinds, (0, 1)), (kinds, "cf")))
    name = name or f"{tgt.excits[:3]}_p_linear"
    ncount = [tgt.n] + [s.n for s in sources.values()]
    L = []
    w = L.append
    allargs = [a.name for a in args] + ncount + list(INT_DIMS)
    w(" " * 14 + f"subroutine {name}({allargs[0]},&")
    rest = allargs[1:]
    for n_ in range(0, len(rest), 4):
        w(" " * 38 + ", ".join(rest[n_:n_ + 4]) + (")" if n_ + 4 >= len(rest) else ",&"))
    w(IND + f"integer, intent(in) :: {', '.join(ncount + list(INT_DIMS))}")
    for a in args:
        dims = ",".join(a.dims)
        if a.name == tgt.excits:
            w(IND + f"integer, intent(inout) :: {a.name}({dims})")
            w(IND + f"!f2py intent(in,out) :: {a.name}(0:{tgt.n}-1,0:{nexc}-1)")
        elif a.name == tgt.amps:
            w(IND + f"real(kind=8), intent(inout) :: {a.name}({dims})")
            w(IND + f"!f2py intent(in,out) :: {a.name}(0:{tgt.n}-1)")
        elif a.kind == "resid":
            w(IND + f"real(kind=8), intent(out) :: resid({dims})")
        elif a.kind == "excits":
            w(IND + f"integer, intent(in) :: {a.name}({dims})")
        else:
            w(IND + f"real(kind=8), intent(in) :: {a.name}({dims})")
    w(IND + "integer, allocatable :: idx_table(:,:,:,:), loc_arr(:,:), block_off(:)")
    w(IND + "integer(kind=8), allocatable :: block_key(:)")
    w(IND + "integer :: kdims(6), nblock")
    w(IND + "logical :: kchain(6)")
    w(IND + "real(kind=8), allocatable :: t3_amps_buff(:), t4_amps_buff(:)")
    w(IND + "integer, allocatable :: t3_excits_buff(:,:), t4_excits_buff(:,:)")
    w(IND + "real(kind=8) :: hmatel, hmatel1, hmatel2, hmatel3, hmatel4")
    w(IND + "integer :: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, idet, jdet, idx, nloc")
    w(IND + "resid = 0.0d0")
    body = "\n".join(L) + "\n\n" + "".join(d.emit() + "\n" for d in blocks)
    body += IND + "call release_sort_workspace()\n"
    body += " " * 14 + f"end subroutine {name}\n"
    return body, args, ncount
