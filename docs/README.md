# Compiling CCpy's Documentation

From the repository root, build the HTML documentation with:

```bash
make docs
open docs/_build/html/index.html
```

This uses the active environment's `python`. To select another interpreter or
enable strict Sphinx checks, use
`make docs PYTHON=/path/to/python SPHINXOPTS="-n -W"`.

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
conda install sphinx sphinx_rtd_theme 
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself 
be inside a directory called `html/` depending on what version of Sphinx is installed).

## Object-oriented methods guide

Start at [methods/index.rst](methods/index.rst) for the native `/methods` API.
The older getting-started and walkthrough pages use the legacy Driver interface.

The guide contains setup/composition, all ground and excited method families,
initial guesses and root selection, FNO and selected spaces, response/gradients,
solver controls, and a complete generated public class catalog. Regenerate the
catalog from an installed checkout with:

```bash
python docs/methods/generate_catalog.py
python docs/methods/generate_catalog.py --check
```

Run small end-to-end examples with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/methods_api/run_calculation.py --recipe all
```

This exercises nine standard chains; it does not validate every high-rank or
custom scientific model described in the catalog.

Analytic/finite-difference derivatives and pysisyphus workflows are documented
in [methods/gradients.rst](methods/gradients.rst) and
[methods/geometry.rst](methods/geometry.rst). Runnable minimum/TS/IRC examples:

```bash
python examples/methods_api/geometry_workflows.py --task gradient-check --method ccsd --out-dir gradient-check
python examples/methods_api/geometry_workflows.py --task minimum --method ccsd --out-dir minimum-run
python examples/methods_api/geometry_workflows.py --task irc --method rhf --out-dir irc-run
```

Each IRC example verifies the TS mode, both branches, relaxed endpoint frequencies
and the ammonia inversion coordinate. Small default basis sets demonstrate the
API; quantitative reaction energetics require independent convergence checks.

Relaxed multipoles, native harmonic IR intensities and finite-field examples:
[methods/multipoles.rst](methods/multipoles.rst), with runnable
`examples/methods_api/multipole_ir.py`.
