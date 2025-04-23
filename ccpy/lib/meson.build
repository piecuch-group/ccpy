cython_c_args = ['-DCYTHON_CCOMPLEX=0']

incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''
import numpy as np
import os
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

blas_name = get_option('blas')
lapack_name = get_option('lapack')
# pkg-config uses a lower-case name while CMake uses a capitalized name, so try
# that too to make the fallback detection with CMake work

if blas_name == 'cibuild'
  openblas_inc = include_directories('/usr/include')
  openblas_lib_dir = '/usr/lib64'

  blas_dep = declare_dependency(
    include_directories: openblas_inc,
    link_args: [
      '-L' + openblas_lib_dir,
      '-lopenblas'
    ]
  )

  lapack_dep = blas_dep

  message('Using CI build openblas configuration')
else
  blas_dep = dependency(blas_name)

  if 'mkl' in blas_dep.name() or blas_dep.name().to_lower() == 'accelerate'
    # For these libraries we know that they contain LAPACK, and it's desirable to
    # use that - no need to run the full detection twice.
    lapack_dep = blas_dep
  elif lapack_name == 'openblas'
    lapack_dep = dependency('openblas')
  else
    lapack_dep = dependency(lapack_name)
  endif
endif

# Build CCPY fortran
sort_sources = [
   'sort/qsort_module.f90',
   'sort/qsort.c',
]

sort_lib = static_library('sort',
 sort_sources,
 c_args: '-O3',
 fortran_args: '-ffree-line-length-512'
)

fortran_sources = [
  'fortran/cc3_loops.f90',
  'fortran/cc_loops2.f90',
  'fortran/cc_loops_t4.f90',
  'fortran/ccp3_adaptive_loops.f90',
  'fortran/ccp3_full_correction.f90',
  'fortran/ccp3_full_correction_high_mem.f90',
  'fortran/ccp3_loops.f90',
  'fortran/ccp3_opt_loops.f90',
  'fortran/ccsdpt_loops.f90',
  'fortran/ccsdt_p_chol_loops.f90',
  'fortran/ccsdt_p_loops.f90',
  'fortran/cct3_loops.f90',
  'fortran/clusteranalysis.f90',
  'fortran/crcc24_loops.f90',
  'fortran/crcc24_opt_loops.f90',
  'fortran/crcc_loops.f90',
  'fortran/creacc_loops.f90',
  'fortran/cripcc_loops.f90',
  'fortran/deaeom4_p_loops.f90',
  'fortran/deaeom4_p_intermediates.f90',
  'fortran/eaccp3_correction.f90',
  'fortran/eaccp3_loops.f90',
  'fortran/eaeom3_p_intermediates.f90',
  'fortran/eaeom3_p_loops.f90',
  'fortran/ecccp3_loops.f90',
  'fortran/eomcc_active_loops.f90',
  'fortran/eomccsdt_p_loops.f90',
  'fortran/hbar_cc3.f90',
  'fortran/hbar_ccsdt_p.f90',
  'fortran/ipccp3_loops.f90',
  'fortran/ipeom3_p_intermediates.f90',
  'fortran/ipeom3_p_loops.f90',
  'fortran/ipeomt_p_loops.f90',
  'fortran/leftccsdt_p_intermediates.f90',
  'fortran/leftccsdt_p_loops.f90',
  'fortran/lefteaeom3_p_intermediates.f90',
  'fortran/lefteaeom3_p_loops.f90',
  'fortran/leftipeom3_p_intermediates.f90',
  'fortran/leftipeom3_p_loops.f90',
  'fortran/reorder.f90',
  'fortran/mbpt_loops.f90',
  'fortran/vvvv_contraction.f90',
  ]

 fortran_lib = static_library('fortran',
  fortran_sources,
  fortran_args: '-ffree-line-length-512',
 )

fortran_module = custom_target('fortran_module',
  output: ['_fortranmodule.c', '_fortran-f2pywrappers2.f90'],
  input: fortran_sources,
  command: [f2py_script, '--f2py-exe', f2py, '@INPUT@', '--build-dir', '@OUTDIR@'],
  build_by_default: true
)

py3.extension_module('_fortran',
  fortran_module,
  c_args: numpy_nodepr_api,
  include_directories: [inc_np, inc_f2py],
  dependencies: [fortranobject_dep, lapack_dep, blas_dep],
  link_with:[fortran_lib, sort_lib],
  link_language: 'fortran',
  install: true,
  subdir: 'ccpy/lib'
)


python_sources = [
  '__init__.py',
  'core.py',
]

py3.install_sources(
    python_sources,
    subdir: 'ccpy/lib'
)
