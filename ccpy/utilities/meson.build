incdir_numpy = run_command(py3,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  c_args: numpy_nodepr_api,
  dependencies: py3_dep,
  include_directories: [inc_np, inc_f2py],
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

# TODO: 64-bit BLAS and LAPACK
#
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
blas = dependency(blas_name)
lapack = dependency(lapack_name)

#if blas.name() == 'mkl' or lapack.name() == 'mkl' or get_option('use-g77-abi')
#  g77_abi_wrappers = files([
#    '_build_utils/src/wrap_g77_abi_f.f',
#    '_build_utils/src/wrap_g77_abi_c.c'
#    ])
#else
#  g77_abi_wrappers = files('_build_utils/src/wrap_dummy_g77_abi.f')
#endif

#generate_config = custom_target(
#  'generate-config',
#  install: true,
#  build_always_stale: true,
#  build_by_default: true,
#  output: '__config__.py',
#  input: '../tools/config_utils.py',
#  command: [py3, '@INPUT@', '@OUTPUT@'],
#  install_dir: py3.get_install_dir() / 'scipy'
#)
#
#generate_version = custom_target(
#  'generate-version',
#  install: true,
#  build_always_stale: true,
#  build_by_default: true,
#  output: 'version.py',
#  input: '../tools/version_utils.py',
#  command: [py3, '@INPUT@', '--source-root', '@SOURCE_ROOT@'],
#  install_dir: py3.get_install_dir() / 'scipy'
#)

updates_sources = [
  'updates/cc_active_loops.f90',
  'updates/cc_loops2.f90',
  'updates/cc_loops.f90',
    'updates/cc_loops_t4.f90',
    'updates/ccp3_loops.f90',
    'updates/ccp_loops.f90',
    'updates/ccp_matrix.f90',
    'updates/ccp_opt_loops.f90',
    'updates/cct3_loops.f90',
    'updates/crcc_loops.f90',
    'updates/eomcc_active_loops.f90',
    'updates/eomcc_initial_guess.f90',
    'updates/mbgf_loops.f90',
    'updates/mbpt_loops.f90',
]

updates_lib = static_library('updates',
  updates_sources,
  fortran_args: '-ffree-line-length-512'
  )

updates_module = custom_target('updates_module',
  output: ['_updatesmodule.c', '_updates-f2pywrappers2.f90'],
  input: 'updates/updates.pyf',
  command: [f2py, '@INPUT@', '--build-dir', '@OUTDIR@']
  )

py3.extension_module('_updates',
  updates_module,
  c_args: numpy_nodepr_api,
  include_directories: [inc_np, inc_f2py],
  dependencies: [fortranobject_dep],
  link_with: updates_lib,
  link_language: 'fortran',
  install: true,
  subdir: 'ccpy/utilities'
  )
