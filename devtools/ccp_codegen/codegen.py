"""Generator for the sparse CC(P) update loops (ccsdt_p_loops.f90 style).

A diagram is given by its base term, e.g. for diagram 5 of update_t3a_p

    A(c/ab)A(k/ij) h2a_voov(c,m,k,e) * t3a(a,b,e,i,j,m)

The generator enumerates the antisymmetrizer cosets and the source sorts,
and works out every sign, bucket key and integral index. The output follows
the hand-written ccsdt_p_loops.f90 line for line.
"""
from dataclasses import dataclass, field
from itertools import combinations
import re

IND = " " * 18  # routine-level indentation of the hand-written code


def parity(seq):
    """+1/-1 parity of the permutation that sorts seq (distinct items)."""
    seq = list(seq)
    inv = sum(1 for x in range(len(seq)) for y in range(x + 1, len(seq)) if seq[x] > seq[y])
    return -1 if inv % 2 else 1


def dim(kind, spin):
    return {("v", "a"): "nua", ("o", "a"): "noa", ("v", "b"): "nub", ("o", "b"): "nob"}[(kind, spin)]


def ncomb(n, k, m):
    """Fortran text for C(n-k, m) written the way the hand code does."""
    terms = [n if k + t == 0 else f"({n}-{k + t})" for t in range(m)]
    if k == 0 and m > 1:
        terms[0] = n
    txt = "*".join(terms)
    return txt + {1: "", 2: "/2", 3: "/6", 4: "/24"}[m]


@dataclass
class Amp:
    """A sparse amplitude array: column letters and spins, Fortran names."""
    excits: str
    amps: str
    n: str
    spin_v: str  # e.g. "aab"
    spin_o: str

    @property
    def nv(self):
        return len(self.spin_v)

    def blocks(self, kind):
        """Column ranges (0-based, within the virt or occ part) of each spin block."""
        spins = self.spin_v if kind == "v" else self.spin_o
        out, start = [], 0
        for s in sorted(set(spins), key=spins.index):
            cnt = spins.count(s)
            out.append((s, start, cnt))
            start += cnt
        return out


@dataclass
class Op:
    name: str           # Fortran array name as written in the loops, e.g. h2a_voov
    label: str          # comment label, e.g. h2a(voov)
    natural: str        # base-term indices in natural order, e.g. "cmke"
    stored: tuple       # positions of natural indices in the stored (reordered) array
    shared: str         # name in the omp shared clause, e.g. H2A_voov


@dataclass
class H1:
    """One-body (h1 oo or vv) delta terms added to a ladder (h2 oooo/vvvv) diagram.

    Slater rules on bra = target and ket = source: h1(C1,F1) * delta(C2,F2), antisymmetrized
    over the contracted (C) and free (F) pairs; only spin-allowed pairings are kept."""
    name: str           # e.g. h1a_oo
    shared: str         # e.g. H1A_oo
    spin: str           # spin of the h1 operator
    sign: int           # -1 for oo, +1 for vv (relative to the h2 term)
    free_first: bool    # condition written (free==contracted) instead of (contracted==free)
    stored_cf: bool = True  # stored array indexed (contracted, free)
    loops: tuple = ()   # loops that carry the h1 terms (default: all)
    combine: str = "sum"    # "half": hmatel + 0.5d0*(...);  "sum": (hmatel + hmatel1 + ...)*amp
    scale: str = ""     # with combine="scaled": hmatel = hmatel + <scale>*(...)


@dataclass
class LinearDiagram:
    """resid(target) += A A' op(free..., contracted...) * source(key..., contracted...)."""
    header: list
    target: Amp
    tv: str             # target virtual letters, e.g. "abc"
    to: str             # target occupied letters, e.g. "ijk"
    source: Amp
    sv: str             # source virtual letters of the base term, e.g. "abe"
    so: str             # source occupied letters of the base term, e.g. "ijm"
    op: Op
    outer: str = "o"    # which antisymmetrizer is the outer loop ("o" or "v")
    buffer: bool = False
    buffer_name: str = ""
    resid: str = "resid"
    index: str = "auto"  # "dense" (idx_table + sort4), "sparse" (sort_pspace + pspace_find), "auto"
    alloc_comment: str = "! allocate sorting arrays (can be reused for each permutation)"
    h1: H1 = None
    loop_order: tuple = ()   # explicit loop names, e.g. ("ABCK","ABCI","ABCJ")
    key_order: str = "vo"    # key columns: virtuals first ("vo") or occupieds first ("ov")
    check_first: str = "if"  # idx guard of a loop's first term: "if", "none", "cycle"
    check_rest: str = "if"   # idx guard of the other terms: "if", "cycle"
    nest: bool = False       # indent the jdet loop inside the idx guard
    sign: int = 1            # sign of the base term
    scale: float = 1.0       # magnitude of the base term when it is not 1
    cnames: str = "column"   # contracted letters named by source column ("column") or in order ("first")
    outer_by_loop: dict = None  # per-loop override of `outer`
    loop_outer: str = None   # nesting of the source sorts if it differs from `outer`

    # ---------------------------------------------------------------- algebra
    def letters(self, kind):
        return (self.tv, self.sv, self.target.spin_v, self.source.spin_v) if kind == "v" else \
               (self.to, self.so, self.target.spin_o, self.source.spin_o)

    def free(self, kind):
        tgt, src, _, _ = self.letters(kind)
        return [x for x in tgt if x not in src]

    def contracted(self, kind):
        tgt, src, _, _ = self.letters(kind)
        return [x for x in src if x not in tgt]

    def tspin(self, kind, letter):
        tgt, _, ts, _ = self.letters(kind)
        return ts[tgt.index(letter)]

    def colspin(self, kind, col):
        return (self.source.spin_v if kind == "v" else self.source.spin_o)[col]

    def tblocks(self, kind):
        """Per target spin block: (spin, letters, base free letters)."""
        tgt, _, ts, _ = self.letters(kind)
        out = []
        for s in sorted(set(ts), key=ts.index):
            pool = [x for x in tgt if ts[tgt.index(x)] == s]
            out.append((s, pool, [x for x in pool if x in self.free(kind)]))
        return out

    def cosets(self, kind):
        """Free-letter choices: products over spin blocks of combinations of same-spin letters."""
        out = [[]]
        for s, pool, fb in self.tblocks(kind):
            out = [o + list(c) for o in out for c in combinations(pool, len(fb))]
        return out

    def loops(self, kind):
        """Columns (0-based, within the virt/occ part of the source) of the contracted letters."""
        _, src, _, ss = self.letters(kind)
        cont = self.contracted(kind)
        out = [()]
        for s, start, cnt in self.source.blocks(kind):
            m = sum(1 for x in cont if ss[src.index(x)] == s)
            opts = [tuple(start + p for p in range(cnt) if p not in key)
                    for key in combinations(range(cnt), cnt - m)]
            out = [o + c for o in out for c in opts]
        return out

    def keycols(self, lv, lo):
        """(kind, column) of the key columns of the source, in key order."""
        v = [("v", c) for c in range(self.source.nv) if c not in lv]
        o = [("o", c) for c in range(len(self.source.spin_o)) if c not in lo]
        if self.key_order in ("vo", "ov"):
            return v + o if self.key_order == "vo" else o + v
        # explicit priority over source column letters, e.g. "BCAIJK"
        letter = lambda kc: ("ABCDEFGH"[kc[1]] if kc[0] == "v" else "IJKLMNOP"[kc[1]])
        return sorted(v + o, key=lambda kc: self.key_order.index(letter(kc)))

    def loop_name(self, lv, lo):
        letters = "ABCDEFGH"[:self.source.nv] + "IJKLMNOP"[:len(self.source.spin_o)]
        return "".join(letters[c if kd == "v" else self.source.nv + c] for kd, c in self.keycols(lv, lo))

    def loop_base(self, kind, loop):
        """Free letters of the loop's (1): the target letters sitting at the contracted columns
        (per spin block, when the source block matches the target block), else the base free."""
        _, src, ts, ss = self.letters(kind)
        out = []
        for s, pool, fb in self.tblocks(kind):
            blk = [b for b in self.source.blocks(kind) if b[0] == s]
            cols = [c for c in loop if self.colspin(kind, c) == s]
            if blk and blk[0][2] == len(pool) and len(cols) == len(fb):
                out += [pool[c - blk[0][1]] for c in cols]
            else:
                out += fb
        return out

    def label(self, kind, loop, free):
        base = self.loop_base(kind, loop)
        out_ = [x for x in base if x not in free]
        in_ = [x for x in free if x not in base]
        return "".join(f"({min(x, y)}{max(x, y)})" for x, y in zip(sorted(out_), sorted(in_)))

    def literal(self, kind, loop, free):
        """Loop-base free letters with the label transpositions applied literally."""
        base = self.loop_base(kind, loop)
        out_ = [x for x in base if x not in free]
        in_ = [x for x in free if x not in base]
        swap = dict(zip(sorted(out_), sorted(in_)))
        swap.update({v: k for k, v in swap.items()})
        return [swap.get(x, x) for x in base]

    def ordered_cosets(self, kind, loop):
        lab = [(self.label(kind, loop, c), c) for c in self.cosets(kind)]
        return [x for x in lab if x[0] == ""] + sorted(x for x in lab if x[0] != "")

    @property
    def sparse(self):
        """Dense index tables only up to 4 key columns, as in ccsdt_p_loops."""
        if self.index == "auto":
            return len(self.keycols(self.loops("v")[0], self.loops("o")[0])) > 4
        return self.index == "sparse"

    def chain(self, keycols):
        """.true. where a key column is ordered after the previous one (same kind and spin block)."""
        out, prev = [], None
        for kd, c in keycols:
            s = self.colspin(kd, c)
            out.append(".true." if prev == (kd, s) else ".false.")
            prev = (kd, s)
        return out

    def ranges(self, keycols):
        txt, prev = [], None
        for kd, c in keycols:
            s = self.colspin(kd, c)
            _, start, cnt = [b for b in self.source.blocks(kd) if b[0] == s][0]
            p = c - start + 1
            n = dim(kd, s)
            hi = n if cnt - p == 0 else f"{n}-{cnt - p}"
            lo = f"-{c - prev[1]}" if prev is not None and prev[0] == kd and prev[2] == s else str(p)
            txt.append(f"(/{lo},{hi}/)")
            prev = (kd, c, s)
        return txt

    def term(self, lv, lo, fv, fo):
        """Sign, bucket key, letter map and ket string of one term (free letters sorted).

        P maps the base free letters to fv/fo and the base key letters to the rest
        (per spin block, both ascending); the term is sgn(P) * X(P), with the source
        row then sorted into its canonical column order."""
        src = self.source
        sign = self.sign
        perm = {}
        for kd, fnew in (("v", fv), ("o", fo)):
            tgt = self.tv if kd == "v" else self.to
            for s, pool, fb in self.tblocks(kd):
                fn = sorted(x for x in fnew if x in pool)
                kb = [x for x in pool if x not in fb]
                kn = [x for x in pool if x not in fn]
                perm.update(zip(fb + kb, fn + kn))
            sign *= parity([tgt.index(perm[x]) for x in tgt])
        rename, ket, idx_args = {}, {}, {}
        for kd, loop in (("v", lv), ("o", lo)):
            _, srow, _, spins = self.letters(kd)
            cont = self.contracted(kd)
            first = max(self.tv if kd == "v" else self.to)
            ccols = {}
            for s, start, cnt in src.blocks(kd):
                cc = [x for x in cont if spins[srow.index(x)] == s]
                lc = [c for c in loop if start <= c < start + cnt]
                ccols.update(zip(cc, lc))
            for n_, (x, c) in enumerate(sorted(ccols.items(), key=lambda t: t[1])):
                rename[x] = chr(ord(first) + 1 + (c if self.cnames == "column" else n_))
            row = [None] * len(srow)
            for x, c in ccols.items():
                row[c] = x
            for s, start, cnt in src.blocks(kd):
                keys = sorted(perm.get(x, x) for i_, x in enumerate(srow) if spins[i_] == s and x not in cont)
                for c in range(start, start + cnt):
                    if row[c] is None:
                        row[c] = keys.pop(0)
            imaged = [perm.get(x, x) for x in srow]
            sign *= parity([row.index(x) for x in imaged])
            idx_args[kd] = {c: row[c] for c in range(len(row)) if row[c] not in cont}
            mixed = "a" in src.spin_v + src.spin_o
            ket[kd] = "".join(rename.get(x, x) + ("~" if mixed and spins[c] == "b" else "") for c, x in enumerate(row))
        args = [idx_args[kd][c] for kd, c in self.keycols(lv, lo)]
        for x in self.free("v") + self.free("o"):
            rename[x] = perm[x]
        return sign, args, rename, ket["o"] + ket["v"]

    # --------------------------------------------------------------- emitter
    def bufs(self):
        if self.buffer:
            return self.buffer_name + "_excits_buff", self.buffer_name + "_amps_buff"
        return self.source.excits, self.source.amps

    def order(self):
        lv_all, lo_all = self.loops("v"), self.loops("o")
        pairs = [(lv, lo) for lo in lo_all for lv in lv_all] if (self.loop_outer or self.outer) == "o" else \
                [(lv, lo) for lv in lv_all for lo in lo_all]
        if self.loop_order:
            byname = {self.loop_name(lv, lo): (lv, lo) for lv, lo in pairs}
            pairs = [byname[n] for n in self.loop_order]
        return pairs

    def emit(self):
        L = []
        w = L.append
        src = self.source
        for h in self.header:
            w(IND + h)
        sbuf_ex, sbuf_amp = self.bufs()
        if self.buffer:
            w(IND + f"! allocate and copy over {src.excits.split('_')[0]} arrays")
            w(IND + f"allocate({sbuf_amp}({src.n}),{sbuf_ex}({src.n},{len(src.spin_v) + len(src.spin_o)}))")
            w(IND + f"{sbuf_amp}(:) = {src.amps}(:)")
            w(IND + f"{sbuf_ex}(:,:) = {src.excits}(:,:)")
        w(IND + self.alloc_comment)
        order = self.order()
        keycols0 = self.keycols(*order[0])
        dims = ",".join(dim(kd, self.colspin(kd, c)) for kd, c in keycols0)
        if self.sparse:
            nk = len(keycols0)
            w(IND + f"kdims(1:{nk}) = (/{dims}/)")
            w(IND + f"kchain(1:{nk}) = (/{','.join(self.chain(keycols0))}/)")
        else:
            parts = []
            for kind in ("v", "o") if self.key_order == "vo" else ("o", "v"):
                for s, start, cnt in src.blocks(kind):
                    m = sum(1 for (kd, c) in keycols0 if kd == kind and start <= c < start + cnt)
                    if m:
                        parts.append(ncomb(dim(kind, s), cnt - m, m))
            w(IND + "nloc = " + "*".join(parts))
            w(IND + "allocate(loc_arr(2,nloc))")
            w(IND + f"allocate(idx_table({dims}))")
        for lv, lo in order:
            self.emit_loop(w, lv, lo, dims.replace(",", ", "))
        w(IND + "! deallocate sorting arrays")
        w(IND + ("deallocate(block_key,block_off)" if self.sparse else "deallocate(loc_arr,idx_table)"))
        if self.buffer:
            w(IND + f"! deallocate {self.buffer_name} buffer arrays")
            w(IND + f"deallocate({sbuf_amp},{sbuf_ex})")
        return "\n".join(L) + "\n"

    def with_h1(self, lv, lo):
        return self.h1 is not None and (not self.h1.loops or self.loop_name(lv, lo) in self.h1.loops)

    def emit_loop(self, w, lv, lo, dims):
        tgt, src = self.target, self.source
        sbuf_ex, sbuf_amp = self.bufs()
        keycols = self.keycols(lv, lo)
        cols = [c + 1 if kd == "v" else src.nv + c + 1 for kd, c in keycols]
        w(IND + f"!!! {self.loop_name(lv, lo)} LOOP !!!")
        sres = f", {self.resid}" if (not self.buffer and src.excits == tgt.excits) else ""
        if self.sparse:
            w(IND + f"call sort_pspace({sbuf_ex}, {sbuf_amp}, {src.n}, {len(keycols)}, (/{','.join(map(str, cols))}/), kdims, kchain, nblock, block_key, block_off{sres})")
        else:
            w(IND + f"call get_index_table(idx_table, {', '.join(self.ranges(keycols))}, {dims})")
            w(IND + f"call sort4({sbuf_ex}, {sbuf_amp}, loc_arr, idx_table, (/{','.join(map(str, cols))}/), {dims}, nloc, {src.n}{sres})")
        h1 = self.with_h1(lv, lo)
        w(IND + "!!!! BEGIN OMP PARALLEL SECTION !!!!")
        ex = tgt.excits + ("" if sbuf_ex == tgt.excits else "," + sbuf_ex)
        am = tgt.amps + ("" if sbuf_amp == tgt.amps else "," + sbuf_amp)
        spins = set(tgt.spin_v + tgt.spin_o + src.spin_v + src.spin_o)
        d = ("noa,nua," if "a" in spins else "") + ("nob,nub," if "b" in spins else "")
        w(IND + f"!$omp parallel shared({self.resid},&")
        w(IND + f"!$omp {ex},&")
        w(IND + f"!$omp {am},&")
        w(IND + ("!$omp nblock,block_key,block_off,kdims,kchain,&" if self.sparse else "!$omp loc_arr,idx_table,&"))
        w(IND + f"!$omp {(self.h1.shared + ',') if h1 else ''}{self.op.shared},&")
        w(IND + f"!$omp {d}{tgt.n}),&")
        lets = "a,b,c,d,i,j,k,l,e,f,m,n" + "".join("," + x for x in "ghop" if x in self.used_letters(lv, lo))
        if h1 and self.nh1(lv, lo) > 1:
            w(IND + "!$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&")
            w(IND + f"!$omp {lets},idet,jdet,&")
        else:
            w(IND + f"!$omp private(hmatel,{'hmatel1,' if h1 else ''}{lets},idet,jdet,&")
        w(IND + "!$omp idx)")
        w(IND + "!$omp do schedule(static)")
        w(IND + f"do idet = 1, {tgt.n}")
        I3 = IND + "   "
        w(I3 + " ".join(f"{x} = {tgt.excits}(idet,{p + 1});" for p, x in enumerate(self.tv)))
        w(I3 + " ".join(f"{x} = {tgt.excits}(idet,{len(self.tv) + p + 1});" for p, x in enumerate(self.to)))
        terms = list(self.terms(lv, lo))
        for n_, (label, body) in enumerate(terms):
            if len(terms) > 1:
                w(I3 + f"! {label}")
            for line in self.guard(body, n_ == 0):
                w(I3 + line)
        w(IND + "end do")
        w(IND + "!$omp end do")
        w(IND + "!$omp end parallel")
        w(IND + "!!!! END OMP PARALLEL SECTION !!!!")

    def used_letters(self, lv, lo):
        """Letters assigned in this loop (target reads and contracted reads)."""
        _, _, ren, _ = self.term(lv, lo, self.cosets("v")[0], self.cosets("o")[0])
        return set(self.tv + self.to) | {ren[x] for x in self.contracted("v") + self.contracted("o")}

    def guard(self, body, first):
        """Wrap a term body (lookup line + jdet loop) in the idx guard."""
        look, loop = body[0], body[1:]
        style = self.check_first if first else self.check_rest
        if style == "none":
            return [look] + loop
        if style == "cycle":
            return [look, "if (idx==0) cycle"] + loop
        pad = "   " if self.nest else ""
        return [look, "if (idx/=0) then"] + [pad + x for x in loop] + ["end if"]

    def nh1(self, lv, lo):
        fv, fo = self.cosets("v")[0], self.cosets("o")[0]
        return len(self.h1_terms(lv, lo, fv, fo, {}, 1))

    def h1_terms(self, lv, lo, fv, fo, rename, sign_sorted):
        """(condition, sign, contracted, free) of the h1 delta terms of one term."""
        h = self.h1
        kind = "o" if self.contracted("o") and len(self.contracted("o")) == 2 else "v"
        loop, fnew = (lo, fo) if kind == "o" else (lv, fv)
        lit = self.literal(kind, loop, fnew)
        srt = sorted(lit, key=lambda x: (self.tblock_index(kind, x), x))
        s_lit = sign_sorted * parity([srt.index(x) for x in lit])
        _, srow, _, sspins = self.letters(kind)
        cont = sorted(self.contracted(kind), key=lambda x: self.ccol(kind, loop, x))
        out = []
        for sg_c, C in ((1, cont), (-1, cont[::-1])):
            for sg_f, F in ((1, lit), (-1, lit[::-1])):
                ok = (self.cspin(kind, C[0]) == h.spin == self.tspin(kind, F[0])
                      and self.cspin(kind, C[1]) == self.tspin(kind, F[1]))
                if ok:
                    out.append((C, F, s_lit * h.sign * sg_c * sg_f))
        return out

    def tblock_index(self, kind, x):
        return [s for s, pool, fb in self.tblocks(kind)].index(self.tspin(kind, x))

    def cspin(self, kind, x):
        _, srow, _, ss = self.letters(kind)
        return ss[srow.index(x)]

    def ccol(self, kind, loop, x):
        _, srow, _, ss = self.letters(kind)
        for s, start, cnt in self.source.blocks(kind):
            cc = [y for y in self.contracted(kind) if ss[srow.index(y)] == s]
            lc = [c for c in loop if start <= c < start + cnt]
            if x in cc:
                return lc[cc.index(x)]

    def terms(self, lv, lo):
        src = self.source
        sbuf_ex, sbuf_amp = self.bufs()
        cv, co = self.ordered_cosets("v", lv), self.ordered_cosets("o", lo)
        outer = (self.outer_by_loop or {}).get(self.loop_name(lv, lo), self.outer)
        pairs = [(v, o) for o in co for v in cv] if outer == "o" else [(v, o) for v in cv for o in co]
        h1 = self.with_h1(lv, lo)
        for (labv, fv), (labo, fo) in pairs:
            inner, outer_ = (labv, labo) if outer == "o" else (labo, labv)
            label = (inner + outer_) or "(1)"
            sign, idx_args, rename, ket = self.term(lv, lo, fv, fo)
            nat = [rename.get(x, x) for x in self.op.natural]
            stored = [nat[p] for p in self.op.stored]
            sg = "-" if sign < 0 else ""
            if abs(self.scale) != 1.0:
                sg += f"{abs(self.scale):.16g}d0*"
            reads = []
            for kd, loop in (("v", lv), ("o", lo)):
                for x in sorted(self.contracted(kd), key=lambda y: self.ccol(kd, loop, y)):
                    c = self.ccol(kd, loop, x)
                    col = c + 1 if kd == "v" else src.nv + c + 1
                    reads.append(f"{rename[x]} = {sbuf_ex}(jdet,{col});")
            if self.sparse:
                look = f"idx = pspace_find(block_key, nblock, pspace_key((/{','.join(idx_args)}/), {len(idx_args)}, kdims, kchain))"
                rng = "do jdet = block_off(idx), block_off(idx+1)-1"
            else:
                look = f"idx = idx_table({','.join(idx_args)})"
                rng = "do jdet = loc_arr(1,idx), loc_arr(2,idx)"
            body = [look, rng,
                    "   " + " ".join(reads),
                    f"   ! compute < {self.to}{self.tv} | {self.op.label} | {ket} >",
                    f"   !hmatel = {sg}{self.op.name}({','.join(nat)})",
                    f"   hmatel = {sg}{self.op.name}({','.join(stored)})"]
            if h1:
                hs = self.h1_terms(lv, lo, fv, fo, rename, sign)
                names = [f"hmatel{t + 1}" for t in range(len(hs))]
                body.append("   " + "; ".join(f"{x} = 0.0d0" for x in names) + (";" if len(hs) > 1 else ""))
                for nm, (C, F, s1) in zip(names, hs):
                    c1, c2 = rename[C[0]], rename[C[1]]
                    f1, f2 = F
                    cond = f"{f2}=={c2}" if self.h1.free_first else f"{c2}=={f2}"
                    args = f"{c1},{f1}" if self.h1.stored_cf else f"{f1},{c1}"
                    body.append(f"   if ({cond}) {nm} = {'-' if s1 < 0 else ''}{self.h1.name}({args})")
                if self.h1.combine in ("half", "scaled"):
                    fac = "0.5d0" if self.h1.combine == "half" else self.h1.scale
                    body.append(f"   hmatel = hmatel + {fac}*({' + '.join(names)})")
                    body.append(f"   {self.resid}(idet) = {self.resid}(idet) + hmatel*{sbuf_amp}(jdet)")
                else:
                    body.append(f"   {self.resid}(idet) = {self.resid}(idet) + (hmatel + {' + '.join(names)})*{sbuf_amp}(jdet)")
            else:
                body.append(f"   {self.resid}(idet) = {self.resid}(idet) + hmatel * {sbuf_amp}(jdet)")
            body.append("end do")
            yield label, body


# ====================================================================== T4 terms
def class_label(base, new):
    """Hand-code style label: transpositions taking the base letters to the new ones."""
    out_ = [x for x in base if x not in new]
    in_ = [x for x in new if x not in base]
    return "".join(f"({min(x, y)}{max(x, y)})" for x, y in zip(sorted(out_), sorted(in_)))


def view_rows(nv, no):
    """Rows of the 2v2o view of a same-spin T3: (vpair, vrest, opair, orest) and sign.

    Each canonical t3(abc,ijk) appears as t3(p1,p2,f,o1,o2,n) with p1<p2, o1<o2."""
    out = []
    for vp in combinations(range(nv), nv - 1):
        vr = [x for x in range(nv) if x not in vp]
        for op in combinations(range(no), no - 1):
            orr = [x for x in range(no) if x not in op]
            out.append((list(vp) + vr, list(op) + orr, parity(list(vp) + vr) * parity(list(op) + orr)))
    return out


@dataclass
class Piece:
    """One contribution to a block intermediate I_q(:,:).

    dense:  I(:,:) += coef * R(:,:,x,y) * amp   summed over `loop` (letter, dim)
    sparse: I(:,:) += coef * R(:,:,f,n) * amp(jdet) over the rows of a T3 bucket q"""
    target: str
    coef: float
    R: str
    idx: tuple = ()
    amp: str = ""
    loop: tuple = ()
    source: str = ""     # "a" (2v2o view of t3a) or "b" (t3b) for sparse pieces


def axpy(target, coef, expr):
    """target(:,:) = target(:,:) +/- [|coef|*]expr"""
    op = "-" if coef < 0 else "+"
    c = "" if abs(coef) == 1.0 else f"{abs(coef):.1f}d0*"
    return f"{target}(:,:) = {target}(:,:) {op} {c}{expr}"


@dataclass
class T4BlockTerm:
    """resid(abcdijkl) += A(cd/ab)A(kl/ij) sum_em I(cd,kl;e,m) * t3(ab e, ij m)

    T4 is re-sorted by the q = (c,d,k,l) columns of each of the 36 classes; for every
    q bucket, I_q is built once (dense in e,m) and contracted with the T3 bucket p."""
    header: list
    pieces: list
    blocks: dict        # name -> (nv dim, no dim) of each I_q block, e.g. {"IA": ("nua","noa")}
    contract: dict      # T3 source ("a"/"b") -> block contracted with it
    shared: str         # integral names for the omp shared clause

    def classes(self):
        tv, to = "abcd", "ijkl"
        cv = [list(c) for c in combinations(range(4), 2)]
        out = []
        for qo in cv:
            for qv in cv:
                pv = [x for x in range(4) if x not in qv]
                po = [x for x in range(4) if x not in qo]
                sign = parity(pv + qv) * parity(po + qo)
                lab = class_label("cd", [tv[x] for x in qv]) + class_label("kl", [to[x] for x in qo])
                out.append((qv, qo, pv, po, sign, lab or "(1)"))
        return out

    def emit_view(self, w):
        w(IND + "! 2v2o view of t3a: t3_excits_buff(9*(idet-1)+r,:) = (p1,p2,f,o1,o2,n), p1<p2, o1<o2")
        w(IND + "allocate(t3_amps_buff(9*n3aaa),t3_excits_buff(9*n3aaa,6))")
        w(IND + "do idet = 1, n3aaa")
        I3 = IND + "   "
        w(I3 + "a = t3a_excits(idet,1); b = t3a_excits(idet,2); c = t3a_excits(idet,3);")
        w(I3 + "i = t3a_excits(idet,4); j = t3a_excits(idet,5); k = t3a_excits(idet,6);")
        w(I3 + "t_amp = t3a_amps(idet)")
        w(I3 + "jdet = 9*(idet-1)")
        for r, (vp, op, sg) in enumerate(view_rows(3, 3), 1):
            row = ",".join(["abc"[x] for x in vp] + ["ijk"[x] for x in op])
            w(I3 + f"t3_excits_buff(jdet+{r},:) = (/{row}/); t3_amps_buff(jdet+{r}) = {'-' if sg < 0 else ''}t_amp")
        w(IND + "end do")
        w(IND + "call get_index_table(idx_table3, (/1,nua-1/), (/-1,nua/), (/1,noa-1/), (/-1,noa/), nua, nua, noa, noa)")
        w(IND + "call sort4(t3_excits_buff, t3_amps_buff, loc_arr3a, idx_table3, (/1,2,4,5/), nua, nua, noa, noa, nloc3, 9*n3aaa)")
        w(IND + "! t3b keyed by its alpha pairs (a,b,i,j)")
        w(IND + "allocate(t3b_amps_buff(n3aab),t3b_excits_buff(n3aab,6))")
        w(IND + "t3b_amps_buff(:) = t3b_amps(:)")
        w(IND + "t3b_excits_buff(:,:) = t3b_excits(:,:)")
        w(IND + "call sort4(t3b_excits_buff, t3b_amps_buff, loc_arr3b, idx_table3, (/1,2,4,5/), nua, nua, noa, noa, nloc3, n3aab)")

    def emit(self):
        L = []
        w = L.append
        for h in self.header:
            w(IND + h)
        w(IND + "nloc3 = nua*(nua-1)/2*noa*(noa-1)/2")
        w(IND + "allocate(loc_arr3a(2,nloc3),loc_arr3b(2,nloc3),idx_table3(nua,nua,noa,noa))")
        self.emit_view(w)
        w(IND + "nloc = (nua-2)*(nua-3)/2*(noa-2)*(noa-3)/2")
        w(IND + "allocate(loc_arr(2,nloc),idx_table(nua,nua,noa,noa))")
        dummy = LinearDiagram([], Amp("", "", "", "aaaa", "aaaa"), "abcd", "ijkl",
                              Amp("", "", "", "aaaa", "aaaa"), "abcd", "ijkl", None)
        for qv, qo, pv, po, sign, lab in self.classes():
            self.emit_class(w, dummy, qv, qo, pv, po, sign, lab)
        w(IND + "deallocate(loc_arr,idx_table,loc_arr3a,loc_arr3b,idx_table3)")
        w(IND + "deallocate(t3_amps_buff,t3_excits_buff,t3b_amps_buff,t3b_excits_buff)")
        return "\n".join(L) + "\n"

    def emit_class(self, w, dummy, qv, qo, pv, po, sign, lab):
        tv, to = "abcd", "ijkl"
        keycols = [("v", c) for c in qv] + [("o", c) for c in qo]
        cols = [c + 1 for c in qv] + [c + 5 for c in qo]
        name = "".join("ABCD"[c] for c in qv) + "".join("IJKL"[c] for c in qo)
        q = [tv[c] for c in qv] + [to[c] for c in qo]
        p = [tv[c] for c in pv] + [to[c] for c in po]
        w(IND + f"!!! {name} LOOP !!! {lab}")
        w(IND + f"call get_index_table(idx_table, {', '.join(dummy.ranges(keycols))}, nua, nua, noa, noa)")
        w(IND + f"call sort4(t4a_excits, t4a_amps, loc_arr, idx_table, (/{','.join(map(str, cols))}/), nua, nua, noa, noa, nloc, n4aaaa, resid)")
        w(IND + "!!!! BEGIN OMP PARALLEL SECTION !!!!")
        w(IND + "!$omp parallel shared(resid,&")
        w(IND + "!$omp t4a_excits,t3_excits_buff,t3b_excits_buff,&")
        w(IND + "!$omp t3_amps_buff,t3b_amps_buff,t2a,&")
        w(IND + "!$omp loc_arr,idx_table,loc_arr3a,loc_arr3b,idx_table3,&")
        w(IND + f"!$omp {self.shared},&")
        w(IND + "!$omp noa,nua,nob,nub,n4aaaa,nloc),&")
        w(IND + f"!$omp private({','.join(self.blocks)},val,a,b,c,d,i,j,k,l,f,n,idet,jdet,kdx,&")
        w(IND + "!$omp idx)")
        for blk, (dv, do) in self.blocks.items():
            w(IND + f"allocate({blk}({dv},{do}))")
        w(IND + "!$omp do schedule(dynamic)")
        w(IND + "do kdx = 1, nloc")
        I3, I6 = IND + "   ", IND + "      "
        w(I3 + "if (loc_arr(2,kdx) < loc_arr(1,kdx)) cycle")
        w(I3 + "idet = loc_arr(1,kdx)")
        w(I3 + " ".join(f"{x} = t4a_excits(idet,{c});" for x, c in zip(q, cols)))
        w(I3 + f"! build I(q,e,m), q = ({','.join(q)})")
        self.emit_build(w, I3, q)
        w(I3 + "do idet = loc_arr(1,kdx), loc_arr(2,kdx)")
        pcols = [c + 1 for c in pv] + [c + 5 for c in po]
        w(I6 + " ".join(f"{x} = t4a_excits(idet,{c});" for x, c in zip(p, pcols)))
        w(I6 + "val = 0.0d0")
        for src, blk in self.contract.items():
            ex, am, loc = ("t3_excits_buff", "t3_amps_buff", "loc_arr3a") if src == "a" else \
                          ("t3b_excits_buff", "t3b_amps_buff", "loc_arr3b")
            w(I6 + f"! sum_fn {blk}(f,n) * t3{src}({p[0]},{p[1]},f,{p[2]},{p[3]},n)")
            w(I6 + f"idx = idx_table3({','.join(p)})")
            w(I6 + "if (idx/=0) then")
            w(I6 + f"do jdet = {loc}(1,idx), {loc}(2,idx)")
            w(I6 + f"   f = {ex}(jdet,3); n = {ex}(jdet,6);")
            w(I6 + f"   val = val + {blk}(f,n) * {am}(jdet)")
            w(I6 + "end do")
            w(I6 + "end if")
        w(I6 + f"resid(idet) = resid(idet) {'+' if sign > 0 else '-'} val")
        w(I3 + "end do")
        w(IND + "end do")
        w(IND + "!$omp end do")
        w(IND + f"deallocate({','.join(self.blocks)})")
        w(IND + "!$omp end parallel")
        w(IND + "!!!! END OMP PARALLEL SECTION !!!!")

    def emit_build(self, w, I3, q):
        sub = dict(zip(("q1", "q2", "q3", "q4"), q))

        def s(txt):
            for k_, v_ in sub.items():
                txt = txt.replace(k_, v_)
            return txt
        for blk in self.blocks:
            w(I3 + f"{blk} = 0.0d0")
        for pc in self.pieces:
            if pc.source:
                ex, am, loc = ("t3_excits_buff", "t3_amps_buff", "loc_arr3a") if pc.source == "a" else \
                              ("t3b_excits_buff", "t3b_amps_buff", "loc_arr3b")
                w(I3 + f"idx = idx_table3({','.join(q)})")
                w(I3 + "if (idx/=0) then")
                w(I3 + f"do jdet = {loc}(1,idx), {loc}(2,idx)")
                w(I3 + f"   f = {ex}(jdet,3); n = {ex}(jdet,6);")
                w(I3 + "   " + axpy(pc.target, pc.coef, f"{am}(jdet) * {pc.R}(:,:,f,n)"))
                w(I3 + "end do")
                w(I3 + "end if")
            else:
                x, dimx = pc.loop
                w(I3 + f"do {x} = 1, {dimx}")
                w(I3 + "   " + axpy(pc.target, pc.coef, f"{s(pc.amp)} * {pc.R}(:,:,{s(','.join(pc.idx))})"))
                w(I3 + "end do")


# ================================================================ dense blocks
@dataclass
class Moment:
    """resid(target) += sign * A A' sum_x I(...) * X(...) with dense factors, per target row.

    `left`/`right` are the two factors as written in the stored layout with the base
    letters, e.g. "I2A_vooo(m,a,i,j)" and "xbuf(m,k,b,c)"; the antisymmetrizer moves
    target letters between them. Cosets are applied as literal transpositions."""
    comment: str
    target: Amp
    tv: str
    to: str
    left: str
    right: str
    loop: tuple          # (letter, dim)
    sign: int = 1
    outer: str = "o"
    pre: list = field(default_factory=list)    # lines before (xbuf build)
    post: list = field(default_factory=list)   # lines after (deallocate)
    shared: str = ""
    ind: str = "    "    # body indentation step of the hand code

    def side(self, kind):
        tgt = self.tv if kind == "v" else self.to
        return [x for x in tgt if x in self.left[self.left.index("("):]]

    def cosets(self, kind):
        tgt, spins = (self.tv, self.target.spin_v) if kind == "v" else (self.to, self.target.spin_o)
        base = self.side(kind)
        out = [[]]
        for s in sorted(set(spins), key=spins.index):
            pool = [x for x in tgt if spins[tgt.index(x)] == s]
            nb = sum(1 for x in base if x in pool)
            out = [o + list(c) for o in out for c in combinations(pool, nb)]
        lab = []
        for new in out:
            out_ = [x for x in base if x not in new]
            in_ = [x for x in new if x not in base]
            pairs = list(zip(sorted(out_), sorted(in_)))
            lab.append(("".join(f"({min(x, y)}{max(x, y)})" for x, y in pairs), pairs))
        return [x for x in lab if x[0] == ""] + sorted(x for x in lab if x[0] != "")

    def emit(self):
        L = []
        w = L.append
        for line in self.pre:
            w(IND + line)
        t = self.target
        x, n = self.loop
        arrays = self.shared or ",".join(e[:e.index("(")] for e in (self.right, self.left))
        w(IND + f"!$omp parallel shared(resid,{t.excits},{arrays},{t.n}),&")
        w(IND + f"!$omp private(idet,a,b,c,i,j,k,{x})")
        w(IND + "!$omp do schedule(static)")
        w(IND + f"do idet = 1, {t.n}")
        I1, I2 = IND + self.ind, IND + 2 * self.ind
        w(I1 + " ".join(f"{y} = {t.excits}(idet,{p + 1});" for p, y in enumerate(self.tv)))
        w(I1 + " ".join(f"{y} = {t.excits}(idet,{len(self.tv) + p + 1});" for p, y in enumerate(self.to)))
        w(I1 + f"do {x} = 1, {n}")
        w(I2 + f"! {self.comment}")
        cv, co = self.cosets("v"), self.cosets("o")
        pairs = [(v, o) for o in co for v in cv] if self.outer == "o" else [(v, o) for v in cv for o in co]
        for (_, pv), (_, po) in pairs:
            swap = {}
            for a_, b_ in pv + po:
                swap[a_], swap[b_] = b_, a_
            sg = self.sign * (-1) ** (len(pv) + len(po))
            sub = lambda e: re.sub(r"\b([a-l])\b", lambda m: swap.get(m.group(1), m.group(1)), e)
            w(I2 + f"resid(idet) = resid(idet) {'+' if sg > 0 else '-'} {sub(self.left)} * {sub(self.right)}")
        w(I1 + "end do")
        w(IND + "end do")
        w(IND + "!$omp end do")
        w(IND + "!$omp end parallel")
        for line in self.post:
            w(IND + line)
        return "\n".join(L) + "\n"


def xbuf_build(dims, loops, lhs, rhs):
    """allocate(xbuf(dims)) and the transposition loop nest of the hand code."""
    out = [f"allocate(xbuf({dims}))"]
    for d_, (x, n) in enumerate(loops):
        out.append("   " * d_ + f"do {x} = 1,{n}")
    out.append("   " * len(loops) + f"xbuf({lhs}) = {rhs}")
    for d_ in reversed(range(len(loops))):
        out.append("   " * d_ + "end do")
    return out


def update_block(t, tv, to, comment="! Update t3 vector", ind="    "):
    """Divide the residual by the Moller-Plesset denominator and add it to the amplitudes."""
    f = lambda kind, s: f"f{s.upper()}_{kind}"
    occ = " + ".join(f"{f('oo', s)}({x},{x})" for x, s in zip(to, t.spin_o))
    vir = " - ".join(f"{f('vv', s)}({x},{x})" for x, s in zip(tv, t.spin_v))
    fs = ",".join(f"f{s.upper()}_{k}" for k in ("oo", "vv") for s in sorted(set(t.spin_o + t.spin_v)))
    L = [IND + comment,
         IND + f"!$omp parallel shared(resid,{t.excits},{t.amps},{fs},{t.n},shift),&",
         IND + "!$omp private(idet,a,b,c,i,j,k,denom)",
         IND + "!$omp do schedule(static)",
         IND + f"do idet = 1, {t.n}",
         IND + ind + " ".join(f"{y} = {t.excits}(idet,{p + 1});" for p, y in enumerate(tv)),
         IND + ind + " ".join(f"{y} = {t.excits}(idet,{len(tv) + p + 1});" for p, y in enumerate(to)),
         IND + ind + f"denom = {occ} - {vir}",
         IND + ind + "resid(idet) = resid(idet)/(denom - shift)",
         IND + ind + f"{t.amps}(idet) = {t.amps}(idet) + resid(idet)",
         IND + "end do",
         IND + "!$omp end do",
         IND + "!$omp end parallel"]
    return "\n".join(L) + "\n"


# ================================================================== routines
@dataclass
class Intermediate:
    """I = coef * H, then `call calc_<I>(I, args...)` (VT3 pieces of the moment terms)."""
    name: str
    coef: str
    H: str
    args: list


@dataclass
class Routine:
    """A complete update_t3?_p: prologue, linear blocks, moments, denominator update."""
    name: str
    target: Amp
    tv: str
    to: str
    args: list          # dummy arguments in order (the Python interface)
    shapes: dict        # dims of every array argument (stored layout)
    intermediates: list
    blocks: list
    moments: list
    update_comment: str = "! Update t3 vector"
    moment_ind: str = "    "
    init_grouped: bool = False   # all `I = coef*H` before the calc calls
    extra_blocks: list = field(default_factory=list)  # e.g. the T4 blocks of CCSDTQ, before the update
    extra_args: list = field(default_factory=list)    # (name, "integer"/"real", dims), placed after shift

    def all_args(self):
        if not self.extra_args:
            return self.args
        k = self.args.index("shift")
        return self.args[:k] + [a for a, _, _ in self.extra_args] + self.args[k:]

    def prologue(self):
        L = []
        w = L.append
        t = self.target
        args = self.all_args()
        w(" " * 14 + f"subroutine {self.name}({args[0]},&")
        rest = args[1:]
        for n_ in range(0, len(rest), 4):
            tail = ")" if n_ + 4 >= len(rest) else ",&"
            w(" " * 38 + ", ".join(rest[n_:n_ + 4]) + tail)
        ints = [a for a in self.args if a in ("noa", "nua", "nob", "nub") or a[:2] in ("n3", "n4")]
        w(IND + f"integer, intent(in) :: {', '.join(ints)}")
        others = [a for a in self.args if a.endswith("_excits") and a != t.excits]
        if others:
            w(IND + f"integer, intent(in) :: {', '.join(f'{a}({self.shapes[a]})' for a in others)}")
        reals = [a for a in self.args if a not in ints + others + ["resid", t.excits, t.amps]]
        items = [a if a == "shift" else f"{a}({self.shapes[a]})" for a in reals]
        lines = [", ".join(items[n_:n_ + 3]) for n_ in range(0, len(items), 3)]
        w(IND + "real(kind=8), intent(in) :: " + (",&\n" + IND + " " * 28).join(lines))
        w(IND + f"integer, intent(inout) :: {t.excits}({t.n},6)")
        w(IND + f"!f2py intent(in,out) :: {t.excits}(0:{t.n}-1,0:5)")
        w(IND + f"real(kind=8), intent(inout) :: {t.amps}({t.n})")
        w(IND + f"!f2py intent(in,out) :: {t.amps}(0:{t.n}-1)")
        w(IND + f"real(kind=8), intent(out) :: resid({t.n})")
        for a, typ, dims in self.extra_args:
            w(IND + f"{'integer' if typ == 'integer' else 'real(kind=8)'}, intent(in) :: {a}({','.join(dims)})")
        if self.extra_blocks:
            w(IND + "integer, allocatable :: block_off(:)")
            w(IND + "integer(kind=8), allocatable :: block_key(:)")
            w(IND + "integer :: kdims(6), nblock")
            w(IND + "logical :: kchain(6)")
            w(IND + "real(kind=8), allocatable :: t4_amps_buff(:)")
            w(IND + "integer, allocatable :: t4_excits_buff(:,:)")
            w(IND + "integer :: g, h, o, p")
        w(IND + "integer, allocatable :: idx_table(:,:,:,:)")
        w(IND + "integer, allocatable :: loc_arr(:,:)")
        w(IND + "real(kind=8), allocatable :: t3_amps_buff(:), xbuf(:,:,:,:)")
        w(IND + "integer, allocatable :: t3_excits_buff(:,:)")
        for I in self.intermediates:
            w(IND + f"real(kind=8) :: {I.name}({self.shapes[I.name]})")
        w(IND + "real(kind=8) :: val, denom, t_amp, hmatel")
        w(IND + "real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4")
        w(IND + "integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet")
        w(IND + "integer :: idx, nloc")
        sets = [IND + f"{I.name} = {I.coef}{I.H}" for I in self.intermediates]
        calls = [IND + f"call calc_{I.name}({I.name},&\n" + IND + "     " + ", ".join(I.args) + ")"
                 for I in self.intermediates]
        if self.init_grouped:
            L += sets + calls
        else:
            L += [x for pair in zip(sets, calls) for x in pair]
        w(IND + "resid = 0.0d0")
        return "\n".join(L) + "\n"

    def body(self):
        out = "".join(b.emit() + "\n" for b in self.blocks)
        out += IND + "!\n" + IND + "! Moment contributions\n" + IND + "!\n"
        out += "".join(m.emit() for m in self.moments)
        out += "".join("\n" + b.emit() for b in self.extra_blocks)
        out += "\n" + update_block(self.target, self.tv, self.to, self.update_comment, self.moment_ind)
        return out

    def emit(self):
        return self.prologue() + "\n" + self.body() + "\n" + " " * 14 + f"end subroutine {self.name}\n"
