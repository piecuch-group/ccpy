"""Compare generated CCSDT(P) blocks with ccpy/lib/fortran/ccsdt_p_loops.f90.

Two levels: "identical" (byte for byte) and "same code" after normalize(): comments,
whitespace, omp data-sharing clauses and `nloc =` allocation sizes are ignored.
usage: python check_ccsdt.py [-v] [name ...]
"""
import difflib
import re
import sys
from pathlib import Path

from specs_ccsdt import CCSDT_SPECS, SPEC_ROUTINE, CCSDT_ROUTINES, ROUTINES

HAND = Path(__file__).resolve().parents[2] / "ccpy/lib/fortran/ccsdt_p_loops.f90"


def hand_block(lines, header, start=0):
    """Lines from the diagram header through the line before the next blank line."""
    i = next(n for n in range(start, len(lines)) if lines[n].strip() == header.strip())
    j = i + 1
    while lines[j].strip() != "":
        j += 1
    return lines[i:j], j


# Hand-code lines that differ from the generator for a documented reason. Each entry maps
# a normalized hand line to the normalized generated line that replaces it.
KNOWN = {
    # (ik) term of the ABCI loop: n > m > k > i,j, so the h1 deltas can never fire; the
    # hand code carries the opposite sign of the (dead) h1 terms.
    "lad_o": ("dead h1 terms with flipped sign", [
        ("if(n==i)hmatel1=-h1{x}_oo(m,j)", "if(n==i)hmatel1=h1{x}_oo(m,j)"),
        ("if(n==j)hmatel2=h1{x}_oo(m,i)", "if(n==j)hmatel2=-h1{x}_oo(m,i)"),
        ("if(m==i)hmatel3=h1{x}_oo(n,j)", "if(m==i)hmatel3=-h1{x}_oo(n,j)"),
        ("if(m==j)hmatel4=-h1{x}_oo(n,i)", "if(m==j)hmatel4=h1{x}_oo(n,i)")]),
    # k is a beta index; the hand code sizes its idx_table axis with noa instead of nob
    "t3b_d1_5": ("idx_table axis of the beta k sized noa", [
        ("allocate(idx_table(nua,nua,nub,noa))", "allocate(idx_table(nua,nua,nub,nob))"),
        ("callget_index_table(idx_table,(/1,nua-1/),(/-1,nua/),(/1,nub/),(/1,nob/),nua,nua,nub,noa)",
         "callget_index_table(idx_table,(/1,nua-1/),(/-1,nua/),(/1,nub/),(/1,nob/),nua,nua,nub,nob)"),
        ("callsort4(t3b_excits,t3b_amps,loc_arr,idx_table,(/1,2,3,6/),nua,nua,nub,noa,nloc,n3aab,resid)",
         "callsort4(t3b_excits,t3b_amps,loc_arr,idx_table,(/1,2,3,6/),nua,nua,nub,nob,nloc,n3aab,resid)")]),
}


def known(name):
    for key, (why, subs) in KNOWN.items():
        if name == key or name.endswith("_" + key):
            x = "a" if name.startswith("t3a") else "b"
            return why, [(h.format(x=x), g.format(x=x)) for h, g in subs]
    return None, []


def split_alloc(t):
    """allocate(a(..),b(..)) -> allocate(a(..)), allocate(b(..))"""
    m = re.match(r"(allocate|deallocate)\((.*)\)$", t)
    if not m:
        return [t]
    parts, depth, cur = [], 0, ""
    for ch in m.group(2):
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    parts.append(cur)
    return [f"{m.group(1)}({p})" for p in parts]


def normalize(lines):
    out, in_clause = [], False
    for l in lines:
        t = l.strip()
        if in_clause:
            in_clause = t.endswith("&")
            continue
        if t.startswith("!$omp parallel"):
            in_clause = t.endswith("&")
            continue
        if t.startswith("!") and not t.startswith("!$omp"):
            continue
        t = re.sub(r"\s*!(?!\$omp).*$", "", t)
        t = re.sub(r"\s+", "", t)
        t = t.replace("=0.d0", "=0.0d0").replace("(:,:,:,:)", "").lower()
        if not t or t.startswith("nloc="):
            continue
        out += split_alloc(t)
    return out


def code_only(lines):
    return normalize(lines)


def compare(name, spec, lines, verbose=False):
    start = next(n for n, l in enumerate(lines) if f"subroutine {SPEC_ROUTINE[name]}(" in l)
    hand, _ = hand_block(lines, spec.header[0], start)
    gen = spec.emit().splitlines()
    if gen == hand:
        return "identical", len(hand)
    a, b = normalize(hand), normalize(gen)
    if a == b:
        return "same code", len(hand)
    why, subs = known(name)
    for h, g in subs:
        a = [g if x == h else x for x in a]
    if why and a == b:
        return f"same code + known hand quirk: {why}", len(hand)
    if verbose:
        sys.stdout.writelines("    " + l + "\n" for l in list(difflib.unified_diff(a, b, "hand", "gen", lineterm="", n=1))[:40])
    nd = sum(1 for op in difflib.SequenceMatcher(None, a, b).get_opcodes() if op[0] != "equal")
    return f"DIFFERS ({nd} hunks)", len(hand)


def routine_lines(lines, name):
    s = next(n for n, l in enumerate(lines) if f"subroutine {name}(" in l)
    e = next(n for n in range(s, len(lines)) if lines[n].strip() == f"end subroutine {name}")
    return lines[s:e + 1]


def split_pro(lines):
    z = next(n for n, l in enumerate(lines) if l.strip() == "resid = 0.0d0")
    return lines[:z + 1], lines[z + 1:]


def dummies(pro):
    """(signature args, {name: (type, intent, dims)}, f2py lines, init statements) of a prologue."""
    code = [re.sub(r"\s*!(?!\$|f2py).*$", "", l).strip() for l in pro]
    joined, buf = [], ""
    for c in code:
        if not c:
            continue
        buf += c[:-1] if c.endswith("&") else c
        if not c.endswith("&"):
            joined.append(buf); buf = ""
    sig = re.match(r"subroutine \w+\((.*)\)", joined[0]).group(1)
    args = [a.strip().lower() for a in sig.split(",")]
    decl, f2py, init = {}, [], []
    for st in joined[1:]:
        low = st.lower().replace(" ", "")
        if low.startswith("!f2py"):
            f2py.append(low)
            continue
        m = re.match(r"(integer|real\(kind=8\))(?:,(intent\(\w+\)))?(?:,allocatable)?::(.*)", low)
        if m:
            for nm, dims in re.findall(r"(\w+)(\([^()]*\))?", m.group(3)):
                if nm.lower() in args:
                    decl[nm] = (m.group(1), m.group(2), re.sub(r"\b1:", "", dims))
            continue
        init.append(low.replace("(:,:,:,:)", ""))
    return args, decl, sorted(f2py), init


def compare_routine(name, lines):
    hand, gen = routine_lines(lines, name), CCSDT_ROUTINES[name].emit().splitlines()
    hp, hb = split_pro(hand)
    gp, gb = split_pro(gen)
    out = []
    for what, x, y in zip(("arguments", "declarations", "f2py", "setup"), dummies(hp), dummies(gp)):
        if x != y:
            out.append(f"{what} differ: {x} / {y}" if what != "declarations" else
                       "declarations differ: " + str({k: (x.get(k), y.get(k)) for k in set(x) | set(y) if x.get(k) != y.get(k)}))
    hb = list(hb)
    for blk, spec in ROUTINES[name]:
        _, subs = known(f"{name[7:10]}_{blk}")
        if not subs:
            continue
        i = next(n for n, l in enumerate(hb) if l.strip() == spec.header[0].strip())
        j = next(n for n in range(i, len(hb)) if hb[n].strip() == "")
        for n in range(i, j):
            for h, g in subs:
                if normalize([hb[n]]) == [h.lower()]:
                    hb[n] = g
    a, b = normalize(hb), normalize(gb)
    if a != b:
        nd = [op for op in difflib.SequenceMatcher(None, a, b).get_opcodes() if op[0] != "equal"]
        out.append(f"body differs in {len(nd)} hunks, first at normalized line {nd[0][1]}: {a[nd[0][1]:nd[0][2]][:3]} / {b[nd[0][3]:nd[0][4]][:3]}")
    return out, len(hand)


def main():
    verbose = "-v" in sys.argv
    names = [x for x in sys.argv[1:] if x != "-v"] or list(CCSDT_SPECS)
    lines = HAND.read_text().splitlines()
    ok = True
    for name in names:
        res, n = compare(name, CCSDT_SPECS[name], lines, verbose)
        ok &= not res.startswith("DIFF")
        print(f"{name:12s} {n:5d} lines  {res}")
    for name in CCSDT_ROUTINES:
        diffs, n = compare_routine(name, lines)
        ok &= not diffs
        print(f"{name:12s} {n:5d} lines  " + ("whole routine: same code" if not diffs else "\n    ".join(["DIFFERS"] + diffs)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
