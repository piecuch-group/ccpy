"""Time the generated CCSDTQ(P) T4 routines with full T3/T4 lists against dense ccsdtq.update_t4?.

Dense: one call of ccpy.cc.ccsdtq.update_t4<x> (all its terms + the denominator update).
Sparse: the generated <x>_p_linear + <x>_p_t3blocks + <x>_p_t2blocks with every T3/T4 in P.
Random amplitudes/integrals of the right shapes and antisymmetry (timing only).

usage: PATH=~/miniforge3/envs/ccpy/bin:$PATH OMP_NUM_THREADS=4 \\
       ~/miniforge3/envs/ccpy/bin/python devtools/ccp_codegen/benchmark.py --noa 5 --nua 8
"""
import argparse
import itertools
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build import build  # noqa: E402
from block_terms import block_routine  # noqa: E402
from t4_routines import linear_routine  # noqa: E402
from test_codegen import (SPIN_OF, antisym, dense_from_rows, random_h, random_pspace,  # noqa: E402
                          spin_groups)

FUNCS = {"a": "update_t4a", "b": "update_t4b", "c": "update_t4c", "d": "update_t4d", "e": "update_t4e"}


def routines(spins):
    texts, meta = [], {}
    for x in spins:
        f = FUNCS[x]
        lin = linear_routine(f, f"t4{x}_lin")
        b3 = block_routine(f, f"t4{x}_b3", kinds=("t3",))
        b2 = block_routine(f, f"t4{x}_b2", kinds=("t2", "t2t2"))
        texts += [lin[0], b3[0], b2[0]]
        meta[x] = {"lin": lin[1], "b3": b3[1], "b2": b2[1]}
    return texts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noa", type=int, default=5)
    ap.add_argument("--nua", type=int, default=8)
    ap.add_argument("--spins", default="abc")
    ap.add_argument("--opt", default="-O3 -march=native")
    ap.add_argument("--workdir", default=tempfile.gettempdir() + "/ccp_codegen_bench")
    a = ap.parse_args()
    noa = nob = a.noa
    nua = nub = a.nua
    dims = {"noa": noa, "nua": nua, "nob": nob, "nub": nub}
    nv, no = {"a": nua, "b": nub}, {"a": noa, "b": nob}
    rng = np.random.default_rng(1)

    t0 = time.time()
    texts, meta = routines(a.spins)
    names = [f"t4{x}_{k}" for x in a.spins for k in ("lin", "b3", "b2")]
    src = "module bench\n  use omp_lib\n  use pspace_sort\n  implicit none\ncontains\n" + "\n".join(texts) + "end module bench\n"
    import hashlib
    import importlib.util
    wd = Path(a.workdir) / hashlib.sha1((src + a.opt).encode()).hexdigest()[:12]
    so = list(wd.glob("bench*.so")) if wd.exists() else []
    if so:   # same source and flags: reuse the compiled module
        spec = importlib.util.spec_from_file_location("bench", so[0])
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mod = m.bench
    else:
        mod = build("bench", src, names, wd, opt=a.opt)
    print(f"generate + compile: {time.time() - t0:.1f} s ({len(src.splitlines())} lines)")

    sparse, dense = {}, {}
    for key, sp in SPIN_OF.items():
        x, t = random_pspace(rng, nv, no, (sp, sp), 1.0)
        t *= 0.01
        sparse[key] = (x, t)
        dense[sp] = dense_from_rows(x, t, [nv[s] for s in sp] + [no[s] for s in sp], spin_groups(sp))
    for sp in ("aa", "ab", "bb"):
        x = 0.1 * rng.standard_normal([nv[sp[0]], nv[sp[1]], no[sp[0]], no[sp[1]]])
        dense[sp] = antisym(x, [(0, 1), (2, 3)]) if sp != "ab" else x
    dense["a"] = 0.1 * rng.standard_normal((nua, noa))
    dense["b"] = 0.1 * rng.standard_normal((nub, nob))
    Hd = {}

    def hget(name):
        if name not in Hd:
            Hd[name] = random_h(rng, name, dims)
            if name.split(".")[2] in ("oo", "vv"):
                n = Hd[name].shape[0]
                Hd[name] = Hd[name] + 5.0 * np.eye(n) * (1 if name.endswith("vv") else -1)
        return Hd[name]

    def ham(p):
        out = SimpleNamespace()
        for sp in ("a", "b"):
            setattr(out, sp, SimpleNamespace(**{k: hget(f"{p}.{sp}.{k}") for k in ("oo", "vv", "ov", "vo")}))
        for sp in ("aa", "ab", "bb"):
            setattr(out, sp, SimpleNamespace(**{"".join(k): hget(f"{p}.{sp}.{''.join(k)}")
                                                for k in itertools.product("ov", repeat=4)}))
        return out

    H, H0 = ham("H"), ham("H0")
    from ccpy.cc import ccsdtq

    print(f"{'spin':6s}{'n4':>10s}{'dense':>10s}{'linear':>10s}{'I*T3':>10s}{'I*T2':>10s}{'sparse':>10s}{'ratio':>8s}")
    F = np.asfortranarray
    for x in a.spins:
        sp = {"a": "aaaa", "b": "aaab", "c": "aabb", "d": "abbb", "e": "bbbb"}[x]
        T = SimpleNamespace(**{k: v.copy() for k, v in dense.items()})
        dT = SimpleNamespace(**{k: np.zeros_like(v) for k, v in dense.items()})
        t0 = time.time()
        getattr(ccsdtq, FUNCS[x])(T, dT, H, H0, 0.0)
        td = time.time() - t0
        times = []
        resid = None
        for k in ("lin", "b3", "b2"):
            call = []
            for arg in meta[x][k]:
                nm, kind, dd, extra = (arg.name, arg.kind, arg.dims, arg.extra) if hasattr(arg, "name") else arg
                if kind == "resid":
                    if k == "lin":
                        continue
                    call.append(resid)
                elif kind in ("amps", "excits"):
                    xx, tt = sparse[nm[:3]]
                    call.append(tt.copy() if kind == "amps" else F(xx))
                elif kind == "t2":
                    call.append(F(dense[extra]))
                elif k == "lin":
                    h = hget(lin_opname(nm))
                    if extra[1] == "cf":
                        call.append(F(h if extra[0] == "oo" else h.T))
                    else:
                        call.append(F(np.transpose(h, extra[1])))
                else:
                    opn, perm = extra
                    call.append(F(np.transpose(hget(opn), perm)))
            fn = getattr(mod, f"t4{x}_{k}")
            t0 = time.time()
            out = fn(*call, noa=noa, nua=nua, nob=nob, nub=nub)
            times.append(time.time() - t0)
            resid = out[0]
            # the target list comes back re-sorted; keep the sparse copy in that order
            sparse[{"a": "t4a", "b": "t4b", "c": "t4c", "d": "t4d", "e": "t4e"}[x]] = (np.array(out[2]), np.array(out[1]))
        n4 = len(sparse[f"t4{x}"][1])
        ts = sum(times)
        print(f"{sp:6s}{n4:10d}{td:10.3f}{times[0]:10.3f}{times[1]:10.3f}{times[2]:10.3f}{ts:10.3f}{ts / td:8.2f}")


def lin_opname(nm):
    """H2A_voov -> H.aa.voov"""
    sp = {"1A": "a", "1B": "b", "2A": "aa", "2B": "ab", "2C": "bb"}[nm[1:3]]
    return f"H.{sp}.{nm.split('_')[1]}"


if __name__ == "__main__":
    main()
