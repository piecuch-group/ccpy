"""Intermediate x T3 terms of the dense T4 equations (ccsdtq.py) as q-blocked sparse code.

A term of ccsdtq.update_t4?
    dT.<s> += coef * np.einsum("<I>,<T3>-><tgt>", I3X_..., T.<t3>)
contracts an intermediate I (built in the source from H*T2 and H0*T3 einsums, then
antisymmetrized with np.transpose) with a sparse T3 over two internal indices (x,y).

Sparse algorithm (keeps the dense N^8 build + N^10 contraction scaling):
  for each coset class (which target letters go to I):
      sort T4 by the q columns (the target letters of I)
      for each q bucket:  build the dense block I_q(x,y) once
          dense pieces:   I_q(:,:) += c * t2(..s..) * H(:,:,..s..)        (loop over s)
          sparse pieces:  I_q(:,:) += c * t3 * H0(:,:,s1,s2)               (T3 view bucket q)
      for each T4 row in the bucket:  resid += sign * sum_rows I_q(x,y) * t3 (T3 view bucket p)
A "view" lists every T3 row once per placement of the key letters inside each spin block,
key columns first, with the antisymmetry sign folded into the amplitude.
"""
import ast
import re
from dataclasses import dataclass, field
from itertools import combinations, permutations
from math import comb

from codegen import IND, parity
from dense_terms import CCSDTQ, group_size

SPINS = {"aa": "aaaa", "ab": "abab", "bb": "bbbb", "a": "aa", "b": "bb"}
ANTI = {"oooo": [(0, 1), (2, 3)], "vvvv": [(0, 1), (2, 3)], "oovv": [(0, 1), (2, 3)], "vvov": [(0, 1)],
        "vooo": [(2, 3)], "ooov": [(0, 1)], "vovv": [(2, 3)]}
DIM = {("v", "a"): "nua", ("o", "a"): "noa", ("v", "b"): "nub", ("o", "b"): "nob"}


# ------------------------------------------------------------------ parsing
def _num(node):
    return float(eval(compile(ast.Expression(node), "<c>", "eval")))


def _is_const(node):
    return all(isinstance(n, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Load))
               for n in ast.walk(node))


def _dotted(node):
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    out.append(node.id)
    return ".".join(reversed(out))


def sym(node):
    """Expression -> list of (coef, ("einsum", spec, ops) | ("transpose", name, perm))."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        s = 1.0 if isinstance(node.op, ast.Add) else -1.0
        return sym(node.left) + [(s * c, t) for c, t in sym(node.right)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [(-c, t) for c, t in sym(node.operand)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if _is_const(node.left):
            return [(_num(node.left) * c, t) for c, t in sym(node.right)]
        return [(_num(node.right) * c, t) for c, t in sym(node.left)]
    if isinstance(node, ast.Call):
        f = _dotted(node.func)
        if f == "np.einsum":
            return [(1.0, ("einsum", node.args[0].value, [_dotted(a) for a in node.args[1:]]))]
        if f == "np.transpose":
            return [(1.0, ("transpose", node.args[0].id, tuple(_num(e) for e in node.args[1].elts)))]
    raise ValueError(ast.dump(node))


@dataclass
class Piece:
    coef: float
    spec: str      # einsum producing the intermediate; its output letters are the I axes
    ops: list


def apply_transposes(pieces, trans):
    """I <- I + sum_k c_k transpose(I, perm_k)."""
    out = list(pieces)
    for c, (_, _, perm) in trans:
        for p in pieces:
            lhs, o = p.spec.split("->")
            out.append(Piece(p.coef * c, f"{lhs}->{''.join(o[int(k)] for k in perm)}", p.ops))
    return out


@dataclass
class Contraction:
    target: str          # T4 spin, e.g. "aaab"
    coef: float
    spec: str            # "<I>,<partner>-><tgt>"
    iname: str
    t3: str              # spin of the partner amplitude: T3 ("aab", sparse) or T2 ("ab", dense)
    pieces: list


def _t2t2(target, coef, spec, ops):
    """H*T2*T2 -> (I = H*T2) x T2, choosing the pairing whose partner shares one index with I."""
    (h, ta, tb), tgt = spec.split("->")[0].split(","), spec.split("->")[1]
    for part, other, po, oo in ((ta, tb, ops[1], ops[2]), (tb, ta, ops[2], ops[1])):
        summed = [x for x in h if x in other]
        iout = "".join(x for x in h + other if x not in summed)
        shared = [x for x in iout if x in part]
        if len(shared) == 1:
            return Contraction(target, coef, f"{iout},{part}->{tgt}", "W", po.split(".")[1],
                               [Piece(1.0, f"{h},{other}->{iout}", [ops[0], oo])])
    raise ValueError(spec)


def parse_contractions(func, path=CCSDTQ, kinds=("t3", "t2", "t2t2")):
    """Terms of ccsdtq.<func> that factor as intermediate x amplitude:
       "t3"  : I x T3 (I from H*T2 and H0*T3);  "t2": I x T2 (I from H0*T4);
       "t2t2": H*T2*T2 as (H*T2) x T2."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    inter, out = {}, []
    for st in fn.body:
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id.startswith("I"):
            inter[st.targets[0].id] = [Piece(c, t[1], t[2]) for c, t in sym(st.value)]
        elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id.startswith("I"):
            nm = st.target.id
            s = 1.0 if isinstance(st.op, ast.Add) else -1.0
            terms = [(s * c, t) for c, t in sym(st.value)]
            if all(t[0] == "transpose" for _, t in terms):
                assert all(t[1] == nm for _, t in terms)
                inter[nm] = apply_transposes(inter[nm], terms)
            else:
                inter[nm] = inter[nm] + [Piece(c, t[1], t[2]) for c, t in terms]
        elif isinstance(st, (ast.AugAssign, ast.Assign)):
            tgt = st.target if isinstance(st, ast.AugAssign) else st.targets[0]
            if not (isinstance(tgt, ast.Attribute) and _dotted(tgt).startswith("dT.")):
                continue
            s = -1.0 if isinstance(st, ast.AugAssign) and isinstance(st.op, ast.Sub) else 1.0
            ts = _dotted(tgt)[3:]
            for c, t in sym(st.value):
                if t[0] != "einsum":
                    continue
                ops = t[2]
                if ops[0].startswith("I") and len(ops) == 2:
                    n = len(ops[1].split(".")[1])
                    if (n == 3 and "t3" in kinds) or (n == 2 and "t2" in kinds):
                        out.append(Contraction(ts, s * c, t[1], ops[0], ops[1].split(".")[1], list(inter[ops[0]])))
                elif (len(ops) == 3 and ops[0].startswith("H") and "t2t2" in kinds
                      and all(len(o.split(".")[1]) == 2 for o in ops[1:])):
                    out.append(_t2t2(ts, s * c, t[1], ops))
    return out


# ------------------------------------------------------------------ algebra
def letter_spins(opname, letters):
    """{letter: (kind, spin)} of an operand, e.g. H.ab.voov or T.aab or T.ab."""
    parts = opname.split(".")
    if parts[0] == "T":
        sp = parts[1]
        n = len(sp)
        return {x: ("v" if p < n else "o", sp[p % n]) for p, x in enumerate(letters)}
    sp, kinds = parts[1], parts[2]
    return {x: (k, s) for x, k, s in zip(letters, kinds, SPINS[sp])}


def antisym_pairs(opname):
    parts = opname.split(".")
    if parts[0] == "T":
        sp = parts[1]
        if len(sp) == 2 and sp[0] == sp[1]:
            return [(0, 1), (2, 3)]
        return []
    if parts[1] in ("aa", "bb"):
        return ANTI.get(parts[2], [])
    return []



FRESH = "uvwzrstgh"
T4N = {"aaaa": "t4a", "aaab": "t4b", "aabb": "t4c", "abbb": "t4d", "bbbb": "t4e"}
T3N = {"aaa": "t3a", "aab": "t3b", "abb": "t3c", "bbb": "t3d"}
T2N = {"aa": "t2a", "ab": "t2b", "bb": "t2c"}


def blocks_of(sp):
    """[(kind, spin, [columns 0-based])] of a T3/T4 in column order."""
    n = len(sp)
    out = []
    for kind, off in (("v", 0), ("o", n)):
        for s in "ab":
            cols = [off + p for p in range(n) if sp[p] == s]
            if cols:
                out.append((kind, s, cols))
    return out


def canonical(facs):
    """Sort antisymmetric index pairs of H/T2 and the same-spin blocks of T3/T4 factors.
    Returns (sign, key)."""
    sign, out = 1, []
    for opn, lets in facs:
        lets = list(lets)
        parts = opn.split(".")
        if parts[0] == "T" and len(parts[1]) in (3, 4):
            for _, _, cols in blocks_of(parts[1]):
                sub = [lets[p] for p in cols]
                srt = sorted(sub)
                sign *= parity([srt.index(x) for x in sub])
                for p, x in zip(cols, srt):
                    lets[p] = x
        else:
            for a, b in antisym_pairs(opn):
                if lets[a] > lets[b]:
                    lets[a], lets[b] = lets[b], lets[a]
                    sign = -sign
        out.append((opn, "".join(lets)))
    return sign, tuple(out)


@dataclass
class View:
    """T3 list with every row once per placement of the key letters inside each spin block,
    rows laid out (key columns..., other columns...), the antisymmetry sign in the amplitude."""
    sp: str
    nkey: tuple   # number of key letters of each block (blocks_of order)

    @property
    def name(self):
        return "v" + self.sp + "".join(str(n) for n in self.nkey)


def sparse_lookup(sp, letters, key, layout):
    """Lookup of a sparse T factor t(letters) by its key letters.

    layout[b] = (key positions, read positions) inside block b of the stored row, for
    the key letters (ascending) and the summed letters (read in ascending column order).
    Returns (key letters in lookup order, summed letters in read order,
             [(sign, assignment of summed letters to the read letters)])."""
    keys, reads, blocks = [], [], []
    for (kind, s, cols), (kp, rp) in zip(blocks_of(sp), layout):
        lets = [letters[c] for c in cols]
        k = sorted(x for x in lets if x in key)
        o = [x for x in lets if x not in key]
        assert len(k) == len(kp) and len(o) == len(rp)
        keys += k
        reads += o
        blocks.append((lets, k, o, kp, rp))
    terms = [(1, {})]
    for lets, k, o, kp, rp in blocks:
        new = []
        for perm in permutations(o):
            arr = [None] * len(lets)
            for x, p in zip(k, kp):
                arr[p] = x
            for x, p in zip(perm, rp):
                arr[p] = x
            # the stored row holds o[t] (read) where perm[t] stands in the factor
            sg = parity([arr.index(x) for x in lets])
            sub = dict(zip(perm, o))
            new += [(s0 * sg, {**m0, **sub}) for s0, m0 in terms]
        terms = new
    return keys, reads, terms


@dataclass
class BlockTerm:
    """dT(tgt) += coef * sum_int I(q, int) * P(p, int), P a sparse T3 or a dense T2."""
    c: Contraction
    layouts: dict = field(default_factory=dict)   # stored integral arrays: name -> (opname, perm)
    views: dict = field(default_factory=dict)
    sources: set = field(default_factory=set)     # T4 spins read by sparse pieces

    def __post_init__(self):
        c = self.c
        (isub, tsub), tgt = c.spec.split("->")[0].split(","), c.spec.split("->")[1]
        nt = len(c.target)
        ren = {x: "abcd"[p] for p, x in enumerate(tgt[:nt])}
        ren.update({x: "ijkl"[p] for p, x in enumerate(tgt[nt:])})
        internal = [x for x in isub if x not in tgt]
        ren.update({x: "xy"[p] for p, x in enumerate(internal)})
        self.isub = "".join(ren[x] for x in isub)
        self.tsub = "".join(ren[x] for x in tsub)
        self.internal = ["x", "y"][:len(internal)]
        ls = letter_spins(f"T.{c.t3}", self.tsub)
        self.ispin = [ls[x] for x in self.internal]
        self.tspin = letter_spins(f"T.{c.target}", "abcd"[:nt] + "ijkl"[:nt])
        self.qbase = [x for x in self.isub if x not in "xy"]
        self.instances = self.expand()
        for _, facs in self.instances:
            for o, _ in facs:
                if o.startswith("T.") and len(o.split(".")[1]) == 4:
                    self.sources.add(o.split(".")[1])

    def expand(self):
        """Pieces in the contraction's letters, canonicalized and merged."""
        acc = {}
        for p in self.c.pieces:
            lhs, out = p.spec.split("->")
            ren = {o: i for o, i in zip(out, self.isub)}
            fresh = iter(FRESH)
            for x in lhs.replace(",", ""):
                if x not in ren:
                    ren[x] = next(fresh)
            facs = [(opn, "".join(ren[x] for x in lets)) for opn, lets in zip(p.ops, lhs.split(","))]
            sign, key = canonical(facs)
            acc[key] = acc.get(key, 0.0) + sign * p.coef
        return [(cf, list(k)) for k, cf in acc.items() if abs(cf) > 1e-14]

    def antisymmetric(self, a, b):
        """Is the intermediate antisymmetric under the swap of letters a and b? (exact on the
        canonical pieces; a False is always safe, it only adds coset classes)"""
        sw = {a: b, b: a}
        acc = {}
        for cf, facs in self.instances:
            sg, key = canonical([(o, "".join(sw.get(x, x) for x in l)) for o, l in facs])
            acc[key] = acc.get(key, 0.0) + sg * cf
        mine = {tuple(f): cf for cf, f in self.instances}
        return all(abs(acc.get(k, 0.0) + mine.get(k, 0.0)) < 1e-12 for k in set(acc) | set(mine))

    def parts(self, pool):
        """q letters of one target block grouped by the antisymmetry of the intermediate."""
        q = [x for x in self.qbase if x in pool]
        groups = [[x] for x in q]
        for a, b in combinations(q, 2):
            if self.antisymmetric(a, b):
                ga = next(g for g in groups if a in g)
                gb = next(g for g in groups if b in g)
                if ga is not gb:
                    ga += gb
                    groups.remove(gb)
        return [sorted(g) for g in groups]

    def classes(self):
        """Coset representatives: which target letters fill each antisymmetric group of q letters.

        Within a group (and within p) letters go in ascending order; the intermediate need not
        be antisymmetric between groups, so groups are filled as ordered choices."""
        nt = len(self.c.target)
        per_kind = []
        for kind, letters in (("v", "abcd"[:nt]), ("o", "ijkl"[:nt])):
            opts = [({}, [])]
            for s in "ab":
                pool = [x for x in letters if self.tspin[x] == (kind, s)]
                if not pool:
                    continue
                pb = [x for x in pool if x not in self.qbase]
                choices = [({}, [], pool)]
                for g in self.parts(pool):
                    nxt = []
                    for m, q, left in choices:
                        for cmb in combinations(left, len(g)):
                            m2 = dict(m)
                            m2.update(zip(g, cmb))
                            nxt.append((m2, q + list(cmb), [x for x in left if x not in cmb]))
                    choices = nxt
                blk = []
                for m, q, left in choices:
                    m = dict(m)
                    m.update(zip(pb, left))
                    blk.append((m, q))
                opts = [({**m0, **m1}, q0 + q1) for m0, q0 in opts for m1, q1 in blk]
            per_kind.append((letters, opts))
        res = []
        (lv, ov), (lo, oo) = per_kind
        for mo, qo in oo:
            for mv, qv in ov:
                perm = {**mv, **mo}
                sign = parity([lv.index(perm[x]) for x in lv]) * parity([lo.index(perm[x]) for x in lo])
                res.append((qv + qo, perm, sign))
        return res

    def scale(self):
        return self.c.coef * group_size(self.c.target) / len(self.classes())

    # ---- views and stored layouts
    def view(self, sp, letters, key):
        nk = tuple(sum(1 for c in cols if letters[c] in key) for _, _, cols in blocks_of(sp))
        v = View(sp, nk)
        self.views.setdefault(v.name, v)
        return v

    def stored(self, opn, letters, first):
        """Name of `opn` stored with the letters `first` leading (block axes), rest natural."""
        perm = [letters.index(x) for x in first] + [p for p, x in enumerate(letters) if x not in first]
        base = opname_f(opn)
        name = base if perm == sorted(perm) else f"{base}_{''.join(map(str, perm))}"
        self.layouts[name] = (opn, tuple(perm))
        return name, [letters[p] for p in perm[len(first):]]

    def placements(self, sp, letters, key):
        """Key positions of a T4 factor: every choice of positions per block."""
        per = [[]]
        for _, _, cols in blocks_of(sp):
            nk = sum(1 for c in cols if letters[c] in key)
            n = len(cols)
            per = [o + [(list(kp), [p for p in range(n) if p not in kp])] for o in per
                   for kp in combinations(range(n), nk)]
        return per

    # ---- emission
    def passes(self):
        """(q columns, pass) for every coset class: the base pass (dense and T3 pieces) and one
        pass per placement of each T4 piece."""
        nt = len(self.c.target)
        base = [(cf, f) for cf, f in self.instances if not any(o.startswith("T.") and len(o) == 6 for o, _ in f)]
        t4 = [(cf, f) for cf, f in self.instances if (cf, f) not in base]
        out = []
        for q, perm, sign in self.classes():
            qcols = ["abcd".index(x) + 1 if x in "abcd" else "ijkl".index(x) + 1 + nt for x in q]
            qcols, q = (list(t) for t in zip(*sorted(zip(qcols, q))))
            if base:
                out.append((tuple(qcols), dict(term=self, q=q, perm=perm, sign=sign, insts=base, pl=None)))
            for cf, f in t4:
                (to, tl), = [(o, l) for o, l in f if o.startswith("T.") and len(o) == 6]
                tl = "".join(perm.get(x, x) for x in tl)
                for pl in self.placements(to.split(".")[1], tl, set(q)):
                    out.append((tuple(qcols), dict(term=self, q=q, perm=perm, sign=sign, insts=[(cf, f)], pl=pl)))
        return out

    def source_key(self, facs, pl):
        """(buffer, n4, cols, dims, chain) of the index of the T4 piece's buffer for one placement."""
        (to, tl), = [(o, l) for o, l in facs if o.startswith("T.") and len(o) == 6]
        sp = to.split(".")[1]
        cols = []
        dims, chain = [], []
        for (kind, s, bcols), (kp, rp) in zip(blocks_of(sp), pl):
            for t_, p in enumerate(kp):
                cols.append(bcols[p] + 1)
                dims.append(DIM[(kind, s)])
                chain.append(".true." if t_ > 0 else ".false.")
        return f"b{T4N[sp]}", f"n4{sp}", tuple(cols), tuple(dims), tuple(chain)

    def ix_dims(self):
        return ",".join(DIM[s] for s in self.ispin)

    def emit_body(self, w, q, qcols, perm, sign, insts, pl, sk, first, last):
        """Body of one pass for block kdx: IX from the instances, then IX x partner into resid."""
        col = ":" if len(self.internal) == 1 else ":,:"
        nt = len(self.c.target)
        ex = f"{T4N[self.c.target]}_excits"
        I3, I6 = IND + "   ", IND + "      "
        w(I3 + f"idet = {first}")
        w(I3 + " ".join(f"{x} = {ex}(idet,{cc});" for x, cc in zip(q, qcols)))
        w(I3 + "IX = 0.0d0")
        for cf, facs in insts:
            self.emit_instance(w, I3, cf, facs, perm, pl, sk, col)
        w(I3 + f"do idet = {first}, {last}")
        p = [x for x in "abcd"[:nt] + "ijkl"[:nt] if x not in q]
        pcols = ["abcd".index(x) + 1 if x in "abcd" else "ijkl".index(x) + 1 + nt for x in p]
        w(I6 + " ".join(f"{x} = {ex}(idet,{cc});" for x, cc in zip(p, pcols)))
        w(I6 + "val = 0.0d0")
        tl = "".join(perm.get(x, x) for x in self.tsub)
        sg = 1
        if len(self.c.t3) == 3:   # sparse T3 partner through a view
            v = self.view(self.c.t3, tl, set(p))
            layout = [(list(range(nk)), list(range(nk, len(cols))))
                      for (_, _, cols), nk in zip(blocks_of(self.c.t3), v.nkey)]
            keys, reads, terms = sparse_lookup(self.c.t3, tl, set(p), layout)
            w(I6 + f"idx = idx_{v.name}({','.join(keys)})")
            w(I6 + "if (idx/=0) then")
            w(I6 + f"do jdet = loc_{v.name}(1,idx), loc_{v.name}(2,idx)")
            w(I6 + "   " + "; ".join(f"{x} = {v.name}_excits(jdet,{5 + t_})" for t_, x in enumerate(reads)))
            acc = {}
            for s_, mp in terms:
                args = ",".join(mp.get(x, x) for x in self.internal)
                acc[args] = acc.get(args, 0) + s_
            expr = " + ".join(f"({cnt:+d})*IX({a_})" for a_, cnt in acc.items() if cnt)
            w(I6 + f"   val = val + {v.name}_amps(jdet) * ({expr})")
            w(I6 + "end do")
            w(I6 + "end if")
        else:                     # dense T2 partner
            (x,) = self.internal
            w(I6 + f"do x = 1, {DIM[self.ispin[0]]}")
            w(I6 + f"   val = val + IX(x) * {T2N[self.c.t3]}({','.join(tl)})")
            w(I6 + "end do")
        tot = sign * sg * self.scale()
        w(I6 + f"resid(idet) = resid(idet) {'+' if tot > 0 else '-'} {abs(tot):.16g}d0 * val")
        w(I3 + "end do")

    def emit_instance(self, w, I3, cf, facs, perm, pl, sk, col):
        sub = lambda s_: "".join(perm.get(x, x) for x in s_)
        sp_f = [(o, sub(l)) for o, l in facs if o.startswith("T.") and len(o.split(".")[1]) in (3, 4)]
        if sp_f:   # sparse piece: H0 * T3 (view) or H0 * T4 (sorted buffer, placement pl)
            (to, tl), = sp_f
            (ho, hl), = [(o, sub(l)) for o, l in facs if not o.startswith("T.")]
            sp = to.split(".")[1]
            q = [x for x in tl if x in "abcdijkl"]
            if len(sp) == 3:
                v = self.view(sp, tl, set(q))
                layout = [(list(range(nk)), list(range(nk, len(cols))))
                          for (_, _, cols), nk in zip(blocks_of(sp), v.nkey)]
                keys, reads, terms = sparse_lookup(sp, tl, set(q), layout)
                look = f"idx = idx_{v.name}({','.join(keys)})"
                rng = f"do jdet = loc_{v.name}(1,idx), loc_{v.name}(2,idx)"
                pre = []
                rcols = [f"{v.name}_excits(jdet,{5 + t_})" for t_ in range(len(reads))]
                amp = f"{v.name}_amps(jdet)"
            else:
                keys, reads, terms = sparse_lookup(sp, tl, set(q), pl)
                b = f"b{T4N[sp]}"
                ix = f"sidx({sk})"
                look = f"idx = pspace_find({ix}%key, {ix}%nblock, pspace_key((/{','.join(keys)}/), {len(keys)}, {ix}%dims, {ix}%chain))"
                rng = f"do jj = {ix}%off(idx), {ix}%off(idx+1)-1"
                pre = [f"jdet = {ix}%perm(jj)"]
                rc = []
                for (kind, s, bcols), (kp, rp) in zip(blocks_of(sp), pl):
                    rc += [bcols[p] + 1 for p in rp]
                rcols = [f"{b}_excits(jdet,{c_})" for c_ in rc]
                amp = f"{b}_amps(jdet)"
            assert set(hl) == set(self.internal) | set(reads), (hl, reads)
            name, rest = self.stored(ho, hl, self.internal)
            acc = {}
            for s_, mp in terms:
                args = [mp.get(x, x) for x in rest]
                # antisymmetric pairs of the integral among its summed letters
                for a_, b_ in antisym_pairs(ho):
                    la, lb = hl[a_], hl[b_]
                    if la in rest and lb in rest:
                        pa, pb = rest.index(la), rest.index(lb)
                        if reads.index(args[pa]) > reads.index(args[pb]):
                            args[pa], args[pb] = args[pb], args[pa]
                            s_ = -s_
                key = ",".join(args)
                acc[key] = acc.get(key, 0) + s_
            w(I3 + look)
            w(I3 + "if (idx/=0) then")
            w(I3 + rng)
            w(I3 + "   " + "; ".join(pre + [f"{x} = {rc_}" for x, rc_ in zip(reads, rcols)]))
            for args, cnt in acc.items():
                if cnt:
                    w(I3 + f"   IX({col}) = IX({col}) + ({cf * cnt:.16g}d0 * {amp}) * {name}({col},{args})")
            w(I3 + "end do")
            w(I3 + "end if")
            return
        # dense piece: H(int, .., s, ..) * T2(.., s, ..), loop over the summed letter s
        (ho, hl), = [(o, sub(l)) for o, l in facs if o.startswith("H")]
        (to_, tl), = [(o, sub(l)) for o, l in facs if o.startswith("T.")]
        assert all(x in hl for x in self.internal), (facs,)
        s = [x for x in hl if x in tl]
        assert len(s) == 1, facs
        name, rest = self.stored(ho, hl, self.internal)
        kind = letter_spins(ho, hl)[s[0]]
        w(I3 + f"do {s[0]} = 1, {DIM[kind]}")
        w(I3 + f"   IX({col}) = IX({col}) + ({cf:.16g}d0 * {T2N[to_.split('.')[1]]}({','.join(tl)})) * {name}({col},{','.join(rest)})")
        w(I3 + "end do")


def emit_fused(w, run, qcols, nb, first, last):
    """Passes of one target sort (same IX shape) in one parallel region over the q blocks."""
    w(IND + "!$omp parallel default(shared) private(IX,val,a,b,c,d,i,j,k,l,x,y,u,v,w,z,r,s,t,g,h,idet,jdet,jj,kdx,idx)")
    w(IND + f"allocate(IX({run[0]['term'].ix_dims()}))")
    w(IND + "!$omp do schedule(dynamic,8)")
    w(IND + f"do kdx = 1, {nb}")
    w(IND + "   " + f"if ({last} < {first}) cycle")
    for p in run:
        t = p["term"]
        w(IND + "   " + f"! {t.c.coef:+.6g} * {t.c.iname}({t.isub}) * t{t.c.t3}({t.tsub}), q = ({','.join(p['q'])}), sign {p['sign']:+d}")
        t.emit_body(w, p["q"], qcols, p["perm"], p["sign"], p["insts"], p["pl"], p.get("sidx"), first, last)
    w(IND + "end do")
    w(IND + "!$omp end do")
    w(IND + "deallocate(IX)")
    w(IND + "!$omp end parallel")


def emit_grouped(terms, tsp):
    """All passes of all terms, one T4 sort per distinct set of q columns."""
    from codegen import LinearDiagram, Amp
    nt = len(tsp)
    t4 = T4N[tsp]
    ex, am, n4 = f"{t4}_excits", f"{t4}_amps", f"n4{tsp}"
    groups = {}
    for t in terms:
        for qc, ps in t.passes():
            groups.setdefault(qc, []).append(ps)
    L = []
    w = L.append
    a4 = Amp(ex, am, n4, tsp, tsp)
    dummy = LinearDiagram([], a4, "abcd"[:nt], "ijkl"[:nt], a4, "abcd"[:nt], "ijkl"[:nt], None)
    allocated = None
    srcs = {}
    for qcols, ps in groups.items():
        keycols = [("v", c_ - 1) if c_ <= nt else ("o", c_ - 1 - nt) for c_ in qcols]
        dims = [DIM[(kd, tsp[c_])] for kd, c_ in keycols]
        w(IND + f"!!!! {len(ps)} passes on T4 sorted by columns ({','.join(map(str, qcols))})")
        if len(qcols) <= 4:
            if allocated != dims:
                if allocated:
                    w(IND + "deallocate(loc_arr,idx_table)")
                w(IND + f"nloc = {'*'.join(dims)}")
                w(IND + f"allocate(loc_arr(2,nloc),idx_table({','.join(dims)}))")
                allocated = dims
            w(IND + f"call get_index_table(idx_table, {', '.join(dummy.ranges(keycols))}, {', '.join(dims)})")
            w(IND + f"call sort4({ex}, {am}, loc_arr, idx_table, (/{','.join(map(str, qcols))}/), {', '.join(dims)}, nloc, {n4}, resid)")
            nb, first, last = "nloc", "loc_arr(1,kdx)", "loc_arr(2,kdx)"
        else:
            chain = [".true." if p > 0 and keycols[p][0] == keycols[p - 1][0] and dims[p] == dims[p - 1] else ".false."
                     for p in range(len(qcols))]
            w(IND + f"tdims(1:{len(qcols)}) = (/{','.join(dims)}/)")
            w(IND + f"tchain(1:{len(qcols)}) = (/{','.join(chain)}/)")
            w(IND + f"call sort_pspace({ex}, {am}, {n4}, {len(qcols)}, (/{','.join(map(str, qcols))}/), tdims, tchain, tnblk, tkey, toff, resid)")
            nb, first, last = "tnblk", "toff(kdx)", "toff(kdx+1)-1"
        # Passes with the same IX shape run in one parallel region (one block loop). Each distinct
        # source key is indexed once per routine (sidx), not re-sorted per pass.
        runs = {}
        for p in ps:
            if p["pl"] is not None:
                p["sidx"] = srcs.setdefault(p["term"].source_key(p["insts"][0][1], p["pl"]), len(srcs) + 1)
            runs.setdefault(p["term"].ix_dims(), []).append(p)
        for run in runs.values():
            emit_fused(w, run, list(qcols), nb, first, last)
    if allocated:
        w(IND + "deallocate(loc_arr,idx_table)")
    if srcs:
        H = [IND + f"! index of each T4 source buffer by the key columns of its placements ({len(srcs)} distinct)",
             IND + f"allocate(sidx({len(srcs)}))"]
        for (b, n, cols, dims, chain), k in srcs.items():
            H.append(IND + f"call index_pspace({b}_excits, {n}, {len(cols)}, (/{','.join(map(str, cols))}/), "
                           f"(/{','.join(dims)}/), (/{','.join(chain)}/), sidx({k}))")
        L[:0] = H
        w(IND + "deallocate(sidx)")
    return "\n".join(L) + "\n"


def emit_view_builds(views):
    """Build and sort every T3 view (one buffer per view)."""
    L = []
    w = L.append
    for v in views.values():
        t = T3N[v.sp]
        blocks = blocks_of(v.sp)
        combos = [[]]
        for (kind, s, cols), nk in zip(blocks, v.nkey):
            combos = [cb + [(cols, k)] for cb in combos for k in combinations(cols, nk)]
        nrep = len(combos)
        nm = v.name
        w(IND + f"! view {nm}: every {t} row once per key placement, key columns first")
        w(IND + f"allocate({nm}_amps({nrep}*n3{v.sp}),{nm}_excits({nrep}*n3{v.sp},6))")
        w(IND + f"do idet = 1, n3{v.sp}")
        w(IND + "   " + " ".join(f"r{c + 1} = {t}_excits(idet,{c + 1});" for c in range(6)))
        w(IND + f"   jdet = {nrep}*(idet-1)")
        for r, cb in enumerate(combos, 1):
            keys, conts, sign = [], [], 1
            for cols, k in cb:
                kk = list(k)
                cc = [c for c in cols if c not in k]
                sign *= parity([(kk + cc).index(c) for c in cols])
                keys += kk
                conts += cc
            row = ",".join(f"r{c + 1}" for c in keys + conts)
            w(IND + f"   {nm}_excits(jdet+{r},:) = (/{row}/); {nm}_amps(jdet+{r}) = {'-' if sign < 0 else ''}{t}_amps(idet)")
        w(IND + "end do")
        kdims, rng = [], []
        for (kind, s, cols), nk in zip(blocks, v.nkey):
            for t_ in range(nk):
                kdims.append(DIM[(kind, s)])
                rng.append(f"(/1,{DIM[(kind, s)]}/)" if t_ == 0 else f"(/-1,{DIM[(kind, s)]}/)")
        w(IND + f"nloc_{nm} = {'*'.join(kdims)}")
        w(IND + f"allocate(loc_{nm}(2,nloc_{nm}),idx_{nm}({','.join(kdims)}))")
        w(IND + f"call get_index_table(idx_{nm}, {', '.join(rng)}, {', '.join(kdims)})")
        w(IND + f"call sort4({nm}_excits, {nm}_amps, loc_{nm}, idx_{nm}, (/1,2,3,4/), {', '.join(kdims)}, nloc_{nm}, {nrep}*n3{v.sp})")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ routine
def block_routine(func, name=None, only=None, kinds=("t3", "t2", "t2t2")):
    """Subroutine with the intermediate x amplitude terms of ccsdtq.<func>. Returns (text, args,
    counts); args are (name, kind, dims, extra) with extra = (opname, stored perm) for integrals."""
    cons = parse_contractions(func, kinds=kinds)
    if only is not None:
        cons = [cons[k] for k in only]
    layouts, views = {}, {}
    terms = [BlockTerm(c, layouts, views) for c in cons]
    bodies = [emit_grouped(terms, cons[0].target)]
    tsp = cons[0].target
    t4 = T4N[tsp]
    name = name or f"{t4}_p_blocks"
    sources = sorted(set().union(*[t.sources for t in terms]), key=list(T4N).index)
    t3used = sorted({v.sp for v in views.values()}, key=list(T3N).index)
    t2used = sorted({o.split(".")[1] for t in terms for _, f in t.instances for o, _ in f
                     if o.startswith("T.") and len(o) == 4} | {t.c.t3 for t in terms if len(t.c.t3) == 2})
    args = [("resid", "resid", [f"n4{tsp}"], None), (f"{t4}_amps", "amps", [f"n4{tsp}"], tsp),
            (f"{t4}_excits", "excits", [f"n4{tsp}", "8"], tsp)]
    for sp in sources:
        if sp != tsp:
            args += [(f"{T4N[sp]}_amps", "amps", [f"n4{sp}"], sp), (f"{T4N[sp]}_excits", "excits", [f"n4{sp}", "8"], sp)]
    for sp in t3used:
        args += [(f"{T3N[sp]}_amps", "amps", [f"n3{sp}"], sp), (f"{T3N[sp]}_excits", "excits", [f"n3{sp}", "6"], sp)]
    for sp in t2used:
        sps = SPINS[sp]
        args.append((T2N[sp], "t2", [DIM[("v", sps[0])], DIM[("v", sps[1])], DIM[("o", sps[0])], DIM[("o", sps[1])]], sp))
    for nm, (opn, perm) in layouts.items():
        parts = opn.split(".")
        nat = [DIM[(k, s)] for k, s in zip(parts[2], SPINS[parts[1]])]
        args.append((nm, "op", [nat[p] for p in perm], (opn, perm)))
    counts = [f"n4{tsp}"] + [f"n4{sp}" for sp in sources if sp != tsp] + [f"n3{sp}" for sp in t3used]
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
        if nm == f"{t4}_excits":
            w(IND + f"integer, intent(inout) :: {nm}({d})")
            w(IND + f"!f2py intent(in,out) :: {nm}(0:n4{tsp}-1,0:7)")
        elif nm == f"{t4}_amps":
            w(IND + f"real(kind=8), intent(inout) :: {nm}({d})")
            w(IND + f"!f2py intent(in,out) :: {nm}(0:n4{tsp}-1)")
        elif kind == "resid":
            w(IND + f"real(kind=8), intent(inout) :: resid({d})")
            w(IND + f"!f2py intent(in,out) :: resid(0:n4{tsp}-1)")
        elif kind == "excits":
            w(IND + f"integer, intent(in) :: {nm}({d})")
        else:
            w(IND + f"real(kind=8), intent(in) :: {nm}({d})")
    w(IND + "integer, allocatable :: idx_table(:,:,:,:), loc_arr(:,:), toff(:)")
    w(IND + "integer(kind=8), allocatable :: tkey(:)")
    w(IND + "integer :: tdims(8), tnblk")
    w(IND + "logical :: tchain(8)")
    for sp in sources:
        w(IND + f"real(kind=8), allocatable :: b{T4N[sp]}_amps(:)")
        w(IND + f"integer, allocatable :: b{T4N[sp]}_excits(:,:)")
    for v in views:
        w(IND + f"real(kind=8), allocatable :: {v}_amps(:)")
        w(IND + f"integer, allocatable :: {v}_excits(:,:), loc_{v}(:,:), idx_{v}(:,:,:,:)")
        w(IND + f"integer :: nloc_{v}")
    w(IND + "real(kind=8), allocatable :: IX(:,:)" if any(len(t.internal) == 2 for t in terms) else
      IND + "real(kind=8), allocatable :: IX(:)")
    if any(len(t.internal) == 2 for t in terms) and any(len(t.internal) == 1 for t in terms):
        raise ValueError("mixed block ranks in one routine")
    w(IND + "real(kind=8) :: val")
    w(IND + "integer :: a, b, c, d, i, j, k, l, x, y, u, v, w, z, r, s, t, g, h")
    w(IND + "integer :: r1, r2, r3, r4, r5, r6, idet, jdet, jj, kdx, idx, nloc")
    w(IND + "type(pspace_index), allocatable :: sidx(:)")
    out = "\n".join(L) + "\n"
    for sp in sources:   # private copies: the target list itself is re-sorted by q
        b = f"b{T4N[sp]}"
        w2 = f"{T4N[sp]}"
        out += IND + f"allocate({b}_amps(n4{sp}),{b}_excits(n4{sp},8))\n"
        out += IND + f"{b}_amps(:) = {w2}_amps(:)\n" + IND + f"{b}_excits(:,:) = {w2}_excits(:,:)\n"
    out += emit_view_builds(views) + "\n" + "\n".join(bodies)
    for v in views:
        out += IND + f"deallocate({v}_amps,{v}_excits,loc_{v},idx_{v})\n"
    for sp in sources:
        out += IND + f"deallocate(b{T4N[sp]}_amps,b{T4N[sp]}_excits)\n"
    out += IND + "call release_sort_workspace()\n"
    out += " " * 14 + f"end subroutine {name}\n"
    return out, args, counts


def opname_f(opn):
    """H.aa.ooov -> H2A_ooov, H0.ab.oovv -> V2B_oovv, H.a.oo -> H1A_oo"""
    parts = opn.split(".")
    head = "H" if parts[0] == "H" else "V"
    if len(parts[1]) == 1:
        return f"{head}1{parts[1].upper()}_{parts[2]}"
    return f"{head}2{ {'aa': 'A', 'ab': 'B', 'bb': 'C'}[parts[1]]}_{parts[2]}"


