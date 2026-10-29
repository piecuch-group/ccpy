"""Locate a libint2 installation for the meson build of ccpy.integrals.

Usage: python find_libint2.py [prefix]

Prints three lines read by ccpy/integrals/meson.build:

    <libint2 prefix>
    <Eigen3 include directory>
    <max AM>;<1e derivative order>;<ERI derivative order>

Search order: the given prefix (-Dlibint2_prefix), $LIBINT2_ROOT, $CONDA_PREFIX,
the running Python's prefix, Homebrew, /usr/local, /usr. Among the installations
found, the first with derivative integrals is preferred (needed for the SCF and CC
gradients); otherwise the first one found is used.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def candidate_prefixes(user_prefix):
    prefixes = [user_prefix, os.environ.get("LIBINT2_ROOT"), os.environ.get("CONDA_PREFIX"), sys.prefix]
    if shutil.which("brew"):
        try:
            prefixes.append(subprocess.run(["brew", "--prefix"], capture_output=True, text=True).stdout.strip())
        except OSError:
            pass
    prefixes += ["/opt/homebrew", "/usr/local", "/usr"]
    seen = []
    for p in prefixes:
        if p and Path(p).is_dir() and Path(p).resolve() not in seen:
            seen.append(Path(p).resolve())
    return seen


def has_library(prefix):
    for libdir in ("lib", "lib64"):
        for ext in ("dylib", "so", "a"):
            if (prefix / libdir / f"libint2.{ext}").exists():
                return True
    return False


def read_config(prefix):
    """Max angular momentum and derivative orders from libint2/config.h."""
    text = (prefix / "include" / "libint2" / "config.h").read_text()

    def define(name, default=0):
        m = re.search(rf"^#define\s+{name}\s+(\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else default

    return define("LIBINT_MAX_AM"), define("LIBINT_INCLUDE_ONEBODY"), define("LIBINT_INCLUDE_ERI")


def find_eigen(prefix):
    for p in [prefix] + candidate_prefixes(None):
        inc = p / "include" / "eigen3"
        if (inc / "Eigen" / "Core").exists():
            return inc
    return None


def main():
    user_prefix = sys.argv[1] if len(sys.argv) > 1 else None
    found = []
    for prefix in candidate_prefixes(user_prefix):
        if (prefix / "include" / "libint2.hpp").exists() and has_library(prefix):
            found.append((prefix, read_config(prefix)))
        if user_prefix and found:
            break  # an explicitly requested prefix wins
    if not found:
        sys.exit(
            "libint2 not found. Install it with `conda install -c conda-forge libint` "
            "(recommended: built with derivative integrals) or `brew install libint`, "
            "or pass -Dlibint2_prefix=/path/to/libint2."
        )
    with_derivs = [f for f in found if f[1][1] >= 1 and f[1][2] >= 1]
    prefix, (max_am, deriv_1e, deriv_eri) = (with_derivs or found)[0]
    eigen = find_eigen(prefix)
    if eigen is None:
        sys.exit("Eigen3 not found. Install it with `conda install -c conda-forge eigen` or `brew install eigen`.")
    print(prefix)
    print(eigen)
    print(f"{max_am};{deriv_1e};{deriv_eri}")


if __name__ == "__main__":
    main()
