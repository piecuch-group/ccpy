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
# auto: Apple Accelerate when it is available (macOS), otherwise OpenBLAS
if blas_name == 'auto'
  blas_name = dependency('accelerate', required: false).found() ? 'accelerate' : 'openblas'
endif
if lapack_name == 'auto'
  lapack_name = blas_name
endif
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

omp_dep = dependency('openmp', language: 'fortran', required: false)
# One OpenMP runtime per process: the kernels call BLAS inside OpenMP parallel regions, and
# OpenBLAS keeps separate buffers only for threads of its own runtime (concurrent calls from
# another runtime's threads share them and return wrong results). Homebrew's OpenBLAS uses LLVM
# libomp, which also provides GCC's GOMP interface: link it instead of libgomp. The toolchain's
# own libomp comes first (in conda-forge environments it is the one OpenBLAS uses, and libgomp
# is an alias of it); Homebrew's only otherwise, never a second copy.
if host_machine.system() == 'darwin' and blas_name == 'openblas'
  libomp = meson.get_compiler('fortran').find_library('omp', required: false)
  if not libomp.found()
    libomp = meson.get_compiler('fortran').find_library('omp',
      dirs: ['/opt/homebrew/opt/libomp/lib', '/usr/local/opt/libomp/lib'])
  endif
  omp_dep = declare_dependency(compile_args: '-fopenmp', dependencies: libomp)
endif

# Shared sorting module used by the Fortran loops; not wrapped by f2py
pspace_sort_lib = static_library('pspace_sort',
  'fortran/pspace_sort.f90',
  fortran_args: '-ffree-line-length-512',
  dependencies: [omp_dep],
)

# Generated CCSDTQ(P) T4 kernels (devtools/ccp_codegen/write_ccsdtq.py), one module per file so they
# compile in parallel; used by ccsdtq_p_loops.f90, not wrapped by f2py
ccsdtq_p_kernels_lib = static_library('ccsdtq_p_kernels',
  'fortran/ccsdtq_p/t4a_p_lin.f90',
  'fortran/ccsdtq_p/t4a_p_b3.f90',
  'fortran/ccsdtq_p/t4a_p_b2.f90',
  'fortran/ccsdtq_p/t4b_p_lin.f90',
  'fortran/ccsdtq_p/t4b_p_b3.f90',
  'fortran/ccsdtq_p/t4b_p_b2.f90',
  'fortran/ccsdtq_p/t4c_p_lin.f90',
  'fortran/ccsdtq_p/t4c_p_b3.f90',
  'fortran/ccsdtq_p/t4c_p_b2.f90',
  'fortran/ccsdtq_p/t4d_p_lin.f90',
  'fortran/ccsdtq_p/t4d_p_b3.f90',
  'fortran/ccsdtq_p/t4d_p_b2.f90',
  'fortran/ccsdtq_p/t4e_p_lin.f90',
  'fortran/ccsdtq_p/t4e_p_b3.f90',
  'fortran/ccsdtq_p/t4e_p_b2.f90',
  fortran_args: '-ffree-line-length-512',
  dependencies: [omp_dep],
  link_with: [pspace_sort_lib],
)

fortran_sources = [
  'fortran/cc3_loops.f90',
  'fortran/cc_loops2.f90',
  'fortran/cc_loops_t4.f90',
  'fortran/ccp3_adaptive_loops.f90',
  'fortran/ccp3_loops.f90',
  'fortran/ccp3_opt_loops.f90',
  'fortran/ccsdpt_loops.f90',
  'fortran/ccsdt_p_chol_loops.f90',
  'fortran/ccsdt_p_loops.f90',
  'fortran/ccsdtq_p_loops.f90',
  'fortran/cct3_loops.f90',
  'fortran/clusteranalysis.f90',
  'fortran/combinatorics.f90',
  'fortran/constants.f90',
  'fortran/crcc24_loops.f90',
  'fortran/crcc24_opt_loops.f90',
  'fortran/crcc_loops.f90',
  'fortran/crcc23_gradient_loops.f90',
  'fortran/creacc_loops.f90',
  'fortran/cripcc_loops.f90',
  'fortran/deaeom4_p_loops.f90',
  'fortran/deaeom4_p_intermediates.f90',
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
  'fortran/stochastic_triples_loops.f90',
  'fortran/sumrcc_loops.f90',
  'fortran/ccsd_p.f90',
  'fortran/mbpt_loops.f90',
  'fortran/vvvv_contraction.f90',
]

# Sources whose OpenMP directives are not valid (missing private clauses); built without OpenMP
fortran_serial_sources = [
  'fortran/ccp3_full_correction.f90',
  'fortran/ccp3_full_correction_high_mem.f90',
  'fortran/eaccp3_correction.f90',
]


fortran_lib = static_library('fortran',
  fortran_sources,
  fortran_args: '-ffree-line-length-512',
  dependencies: [omp_dep],
  link_with: [pspace_sort_lib, ccsdtq_p_kernels_lib],
)

fortran_serial_lib = static_library('fortran_serial',
  fortran_serial_sources,
  fortran_args: '-ffree-line-length-512',
  link_with: [pspace_sort_lib],
)


fortran_module = custom_target('fortran_module',
  output: ['_fortranmodule.c', '_fortran-f2pywrappers2.f90'],
  input: fortran_sources + fortran_serial_sources,
  command: [f2py_script, '--source-root', '@SOURCE_ROOT@', '--f2py-exe', f2py, '@INPUT@', '--build-dir', '@OUTDIR@'],
  build_by_default: true
)


py3.extension_module('_fortran',
  fortran_module,
  c_args: numpy_nodepr_api,
  include_directories: [inc_np, inc_f2py],
  dependencies: [fortranobject_dep, lapack_dep, blas_dep, omp_dep],
  link_with:[fortran_lib, fortran_serial_lib, pspace_sort_lib, ccsdtq_p_kernels_lib, sort_lib],
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
