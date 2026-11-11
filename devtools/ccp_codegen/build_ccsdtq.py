"""Compile the generated ccsdtq_p_loops.f90 into a standalone f2py module (outside the ccpy build).

usage: python build_ccsdtq.py <outdir>   -> <outdir>/ccsdtq_p_loops*.so
"""
import sys
from pathlib import Path

from build import build
from write_ccsdtq import TARGET, ENTRY_POINTS, kernel_files

if __name__ == "__main__":
    out = Path(sys.argv[1])
    build("ccsdtq_p_loops", TARGET.read_text(), ENTRY_POINTS, out, extra=kernel_files(), opt="-O2")
    print("built", next(out.glob("ccsdtq_p_loops*.so")))
