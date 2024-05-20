cython_c_args = ['-DCYTHON_CCOMPLEX=0'] # see gh-18975 for why we need this

incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# Share this object across multiple modules.
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  c_args: numpy_nodepr_api,
  dependencies: py3_dep,
  include_directories: [inc_np, inc_f2py],
  gnu_symbol_visibility: 'hidden',
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)
# Workaround for numpy#24761 on numpy<1.26.1 (see also gh-20515)
_f2py_c_args = []
fortranobject_dep = declare_dependency(
  dependencies: fortranobject_dep,
  compile_args: _f2py_c_args,
)

f2py = find_program('f2py')

f2py_version = run_command([f2py, '-v'], check: true).stdout().strip()
if f2py_version.version_compare('<'+min_numpy_version)
  error(f'Found f2py executable is too old: @f2py_version@')
endif


# ----- OLD FILE -----


# Note that this works as long as BLAS and LAPACK are detected properly via
# pkg-config. By default we look for OpenBLAS, other libraries can be configured via
# `meson configure -Dblas=blas -Dlapack=lapack` (example to build with Netlib
# BLAS and LAPACK).
# For MKL and for auto-detecting one of multiple libs, we'll need a custom
# dependency in Meson (like is done for scalapack) - see
# https://github.com/mesonbuild/meson/issues/2835
blas_name = get_option('blas')
lapack_name = get_option('lapack')
# pkg-config uses a lower-case name while CMake uses a capitalized name, so try
# that too to make the fallback detection with CMake work
if blas_name == 'openblas'
  blas_name = ['openblas', 'OpenBLAS']
endif
if lapack_name == 'openblas'
  lapack_name = ['openblas', 'OpenBLAS']
endif

blas_dep = dependency(blas_name)
lapack_dep = dependency(lapack_name)


# Build CCPY optimizations

optimizations_sources = [
   '_updates/cc_loops.f90',
   '_updates/cc_loops2.f90',
   '_updates/crcc_loops.f90',
   '_updates/cripcc_loops.f90',
   '_updates/creacc_loops.f90',
   '_updates/ccsdpt_loops.f90',
   '_updates/crcc24_loops.f90',
   '_updates/crcc24_opt_loops.f90',
   '_updates/cc3_loops.f90',
   '_updates/hbar_cc3.f90',
   '_updates/mbgf_loops.f90',
   '_updates/mbpt_loops.f90',
   '_updates/ccsdt_p_loops.f90',
   '_updates/eomccsdt_p_loops.f90',
   '_updates/leftccsdt_p_loops.f90',
   '_updates/leftccsdt_p_intermediates.f90',
   '_updates/eaeom3_p_loops.f90',
   '_updates/eaeom3_p_intermediates.f90',
   '_updates/ipeom3_p_loops.f90',
   '_updates/ipeom3_p_intermediates.f90',
   '_updates/lefteaeom3_p_loops.f90',
   '_updates/lefteaeom3_p_intermediates.f90',
   '_updates/leftipeom3_p_loops.f90',
   '_updates/leftipeom3_p_intermediates.f90',
   '_updates/deaeom4_p_loops.f90',
   '_updates/hbar_ccsdt_p.f90',
   '_updates/eomcc_active_loops.f90',
   '_updates/cc_active_loops.f90',
   '_updates/cct3_loops.f90',
   '_updates/cc_loops_t4.f90',
   '_updates/ccp3_loops.f90',
   '_updates/ccp3_opt_loops.f90',
   '_updates/ccp3_adaptive_loops.f90',
   '_updates/eaccp3_loops.f90',
   '_updates/ipccp3_loops.f90',
   '_updates/clusteranalysis.f90',
   '_updates/ecccp3_loops.f90',
   '_updates/reorder.f90',
   '_updates/vvvv_contraction.f90'
  ]

optimizations_lib = static_library('optimizations',
 optimizations_sources,
 fortran_args: '-ffree-line-length-512'
 )

optimizations_module = custom_target('optimizations_module',
  output: ['_optimizationsmodule.c', '_optimizations-f2pywrappers2.f90'],
  input: optimizations_sources,
  command: [f2py_script, '--f2py-exe', f2py, '@INPUT@', '--build-dir', '@OUTDIR@'],
  build_by_default: true
  )

py3.extension_module('_optimizations',
  optimizations_module,
  c_args: numpy_nodepr_api,
  include_directories: [inc_np, inc_f2py],
  dependencies: [fortranobject_dep, lapack_dep, blas_dep],
  link_with: optimizations_lib,
  link_language: 'fortran',
  install: true,
  subdir: 'ccpy/utilities'
  )


python_sources = [
  '__init__.py',
  'active_space.py',
  'determinants.py',
  'dumping.py',
  'permutations.py',
  'printing.py',
  'pspace.py',
  'selection.py',
  'symmetry.py',
  'updates.py',
  'utilities.py'
]

py3.install_sources(
    python_sources,
    subdir: 'ccpy/utilities'
)
