"""Read the dense CCSDTQ equations (ccpy/cc/ccsdtq.py) and turn their terms into sparse specs.

Every `dT.<spin> (+|-)= coef * np.einsum("op,amp->tgt", H.xx.yyyy, T.zzz)` line of an
update function is a Term. Linear terms (one H, one T3/T4) become LinearDiagrams; the
one-body h1(oo)/h1(vv) terms are merged into the matching h2(oooo)/h2(vvvv) ladder as
delta terms, as in the hand-written CCSDT(P) loops.

The same Terms evaluate the dense reference (numpy) for the tests.
"""
import ast
import re
from dataclasses import dataclass
from math import factorial
from pathlib import Path

from codegen import Amp, H1, LinearDiagram, Op

CCSDTQ = Path(__file__).resolve().parents[2] / "ccpy/cc/ccsdtq.py"

T4 = {"aaaa": "t4a", "aaab": "t4b", "aabb": "t4c", "abbb": "t4d", "bbbb": "t4e"}
T3 = {"aaa": "t3a", "aab": "t3b", "abb": "t3c", "bbb": "t3d"}
T2 = {"aa": "t2a", "ab": "t2b", "bb": "t2c"}


def amp(spin):
    """Amp for a T3/T4 spin case, e.g. "aaab" -> t4b_excits, t4b_amps, n4aaab."""
    name = (T4 if len(spin) == 4 else T3)[spin]
    return Amp(f"{name}_excits", f"{name}_amps", f"n{len(spin)}{spin}", spin, spin)


def op_name(h):
    """H.aa.voov -> h2a_voov, H.a.oo -> h1a_oo"""
    _, sp, blk = h.split(".")
    if len(sp) == 1:
        return f"h1{sp}_{blk}"
    return f"h2{ {'aa': 'a', 'ab': 'b', 'bb': 'c'}[sp]}_{blk}"


@dataclass
class Term:
    target: str      # spin of the target, e.g. "aaab"
    coef: float
    spec: str        # einsum spec
    ops: list        # operand names, e.g. ["H.aa.voov", "T.aaab"]

    @property
    def parts(self):
        lhs, out = self.spec.split("->")
        return lhs.split(","), out

    def is_linear(self):
        """One H and one T3/T4 operand."""
        return (len(self.ops) == 2 and self.ops[0].startswith("H.") and self.ops[1].startswith("T.")
                and len(self.ops[1].split(".")[1]) in (3, 4))


def parse(func, path=CCSDTQ):
    """Terms of `def func(...)` in ccsdtq.py, in source order (intermediates excluded)."""
    src = path.read_text()
    body = src[src.index(f"def {func}("):]
    if "\ndef " in body[1:]:
        body = body[:body.index("\ndef ", 1)]
    out = []
    pat = re.compile(r"dT\.(\w+)\s*(\+=|-=|=)\s*([^\n]*?)np\.einsum\(\"([^\"]+)\",\s*([^)]*?),\s*optimize=True\)")
    for m in pat.finditer(body):
        spin, opr, pre, spec, args = m.groups()
        pre = pre.strip().rstrip("*").strip()
        coef = {"": 1.0, "-": -1.0}.get(pre.replace(" ", "")) or float(eval(pre))
        if opr == "-=":
            coef = -coef
        ops = [a.strip() for a in args.split(",")]
        out.append(Term(spin, coef, spec, ops))
    return out


def group_size(spin):
    """|S| of the full antisymmetrizer over same-spin virtuals and occupieds."""
    n = 1
    for s in set(spin):
        n *= factorial(spin.count(s)) ** 2
    return n


def canonical(term):
    """Rename the einsum letters: target columns -> a,b,c,d / i,j,k,l (so that letter order is
    value order within a block), contracted virtuals -> e,f,.. and occupieds -> m,n,.."""
    (oi, ai), tgt = term.parts
    nt, ns = len(term.target), len(term.ops[1].split(".")[1])
    ren = {x: "abcd"[p] for p, x in enumerate(tgt[:nt])}
    ren.update({x: "ijkl"[p] for p, x in enumerate(tgt[nt:])})
    cv = [x for x in ai[:ns] if x not in ren]
    co = [x for x in ai[ns:] if x not in ren]
    ren.update({x: "efgh"[len(tgt[:nt]) - 4 + p] if nt == 4 else "defg"[p] for p, x in enumerate(cv)})
    ren.update({x: "mnop"[p] if nt == 4 else "lmno"[p] for p, x in enumerate(co)})
    r = lambda s_: "".join(ren[x] for x in s_)
    return r(oi), r(ai), r(tgt)


def diagram(term, layouts, buffer_names=("t4", "t3"), natural=False):
    """LinearDiagram of a linear term: base sign/scale, contracted-first integral layout."""
    oi, ai, tgt = canonical(term)
    h, t = term.ops
    tsp = term.target
    nt = len(tsp)
    ssp = t.split(".")[1]
    ns = len(ssp)
    target, source = amp(tsp), amp(ssp)
    tv, to = tgt[:nt], tgt[nt:]
    sv, so = ai[:ns], ai[ns:]
    cont = [x for x in oi if x not in tgt]
    name = op_name(h)
    if natural:
        layouts[name] = tuple(range(len(oi)))
    elif name not in layouts:
        layouts[name] = tuple([oi.index(x) for x in oi if x in cont] + [oi.index(x) for x in oi if x not in cont])
    op = Op(name, f"{name[:3]}({name[4:]})", oi, layouts[name], name.upper()[:3] + name[3:])
    same = source.excits == target.excits
    d = LinearDiagram(header=[f"!!!! {term.coef:+.6g} * {h}({term.parts[0][0]}) * {t}({term.parts[0][1]}) -> {term.parts[1]}"],
                      target=target, tv=tv, to=to, source=source, sv=sv, so=so, op=op,
                      buffer=not same, buffer_name=buffer_names[0] if ns == 4 else buffer_names[1])
    # base scale: coef * |S| / #cosets * prod(k!) over same-spin contracted groups of the source
    ncos = len(d.cosets("v")) * len(d.cosets("o"))
    kfac = 1
    for kd in ("v", "o"):
        _, src_l, _, sspin = d.letters(kd)
        for s in "ab":
            kfac *= factorial(sum(1 for x in d.contracted(kd) if sspin[src_l.index(x)] == s))
    val = term.coef * group_size(tsp) / ncos * kfac
    d.sign = 1 if val > 0 else -1
    d.scale = abs(val)
    return d


def ladder_key(d):
    """(kind, spins of the contracted pair) of an h2 ladder diagram, else None."""
    for kd in ("o", "v"):
        c = d.contracted(kd)
        other = d.contracted("v" if kd == "o" else "o")
        if len(c) == 2 and not other:
            return kd, "".join(sorted(d.cspin(kd, x) for x in c))
    return None


def attach_h1(ladder, h1d):
    """Merge the h1 diagram into the h2 ladder as delta terms.

    Same-spin pair in a block of n letters: every single-orbital-difference ket is reached
    by n-1 (loop, coset) pairs, so the h1 terms carry 1/(n-1). Mixed-spin pair: the ket
    is reached once per position of the other-spin contracted letter, so the h1 terms go
    only in the loops where that letter sits first in its block."""
    kd = "o" if h1d.contracted("o") else "v"
    s = h1d.op.name[2]
    pair = ladder.contracted(kd)
    spins = [ladder.cspin(kd, x) for x in pair]
    h1sign = h1d.sign * ladder.sign  # relative to the h2 term
    shared = h1d.op.name.upper()[:3] + h1d.op.name[3:]
    if spins[0] == spins[1]:
        n = sum(1 for x in (ladder.to if kd == "o" else ladder.tv) if ladder.tspin(kd, x) == s)
        comb = "sum" if n == 2 else ("half" if n == 3 else "scaled")
        h = H1(h1d.op.name, shared, s, h1sign, free_first=(kd == "v"), combine=comb,
               scale=f"(1.0d0/{n - 1}.0d0)")
    else:
        other = [x for x in pair if ladder.cspin(kd, x) != s][0]
        blk = [b for b in ladder.source.blocks(kd) if b[0] == ladder.cspin(kd, other)][0]
        loops = [ladder.loop_name(lv, lo) for lv, lo in ladder.order()
                 if ladder.ccol(kd, lo if kd == "o" else lv, other) == blk[1]]
        h = H1(h1d.op.name, shared, s, h1sign, free_first=(kd == "v"), loops=tuple(loops))
    ladder.h1 = h
    return ladder


def linear_diagrams(func, layouts=None, source_rank=None, natural=False):
    """LinearDiagrams of all linear terms of an update function, h1(oo/vv) merged into ladders.
    source_rank=4 keeps only the terms with a T4 source (e.g. the T4 part of update_t3a)."""
    layouts = {} if layouts is None else layouts
    terms = [t for t in parse(func) if t.is_linear()
             and (source_rank is None or len(t.ops[1].split(".")[1]) == source_rank)]
    diags = [(t, diagram(t, layouts, natural=natural)) for t in terms]
    h1s = [(t, d) for t, d in diags if len(t.ops[0].split(".")[1]) == 1 and t.ops[0].split(".")[2] in ("oo", "vv")]
    rest = [(t, d) for t, d in diags if (t, d) not in h1s]
    for t, d in h1s:
        kd = "o" if d.contracted("o") else "v"
        s = d.op.name[2]
        cands = [x for _, x in rest if x.source.excits == d.source.excits and ladder_key(x)
                 and ladder_key(x)[0] == kd and s in ladder_key(x)[1]]
        cands.sort(key=lambda x: ladder_key(x)[1] != s + s)  # same-spin pair first
        if not cands:
            raise ValueError(f"no ladder for {t}")
        attach_h1(cands[0], d)
    return [d for _, d in rest]
