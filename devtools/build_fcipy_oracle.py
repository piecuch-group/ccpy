"""Build CCpy's optional connector using an existing FCIpy source checkout.

python devtools/build_fcipy_oracle.py /path/to/FCIpy --output build/fcipy
No files in the FCIpy checkout are modified or vendored.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/fcipy"))
    args = parser.parse_args()
    source = args.source.resolve() / "fcipy/lib/ci.f90"
    if not source.is_file():
        parser.error(f"Missing {source}")
    connector = Path(__file__).resolve().parents[1] / "ccpy/gradients/ccpq/fortran/fcipy_connected.f90"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "numpy.f2py", "-c", str(source), str(connector),
                    "-m", "_ccpq_fcipy", "--backend", "meson",
                    "--f90flags=-O3 -ffree-line-length-none"], cwd=output, check=True)
    extensions = list(output.glob("_ccpq_fcipy*.so")) + list(output.glob("_ccpq_fcipy*.pyd"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one connector extension in {output}, found {extensions}")
    print("export CCPY_FCIPY_EXTENSION=" + shlex.quote(str(extensions[0])))


if __name__ == "__main__":
    main()
