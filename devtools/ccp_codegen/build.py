"""Compile generated Fortran (with pspace_sort.f90) into a throwaway f2py module."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

FORTRAN = Path(__file__).resolve().parents[2] / "ccpy/lib/fortran"


def build(name, source, routines, workdir, extra=(), opt="-O2"):
    """Write source to workdir, build module `name` exposing `routines`, import and return it."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / f"{name}.f90"
    src.write_text(source)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    # pspace_sort goes in as an object: f2py must not scan it (the module it builds for a public
    # derived type such as pspace_index crashes on import)
    obj = workdir / "pspace_sort.o"
    res = subprocess.run(["gfortran", "-c", "-fPIC", "-fopenmp", opt, str(FORTRAN / "pspace_sort.f90"), "-o", str(obj)],
                         cwd=workdir, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-4000:] + res.stderr[-4000:])
    cmd = [sys.executable, "-m", "numpy.f2py", "-c", str(obj), *map(str, extra), str(src), f"-I{workdir}",
           "-m", name, f"--f90flags=-fopenmp {opt}", "-lgomp", "only:", *routines, ":"]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-4000:] + res.stderr[-4000:])
    so = next(workdir.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name, mod)  # routines live under the Fortran module of the same name
