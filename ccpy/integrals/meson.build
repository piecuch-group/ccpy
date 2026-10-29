# libint2-based AO integral library, exposed to Python as ccpy.integrals._libint
# through nanobind. Adapted from forte2 (github.com/evangelistalab/forte2, LGPL-3.0).

cxx = meson.get_compiler('cpp')

# ---------------------------------------------------------------------------
# nanobind: compiled from the sources shipped with the Python package, so the
# only requirement is `pip install nanobind` (it is a build requirement in
# pyproject.toml).
# ---------------------------------------------------------------------------
nanobind_dir = run_command(py3,
  ['-c', 'import nanobind, os; print(os.path.dirname(nanobind.__file__))'],
  check: true
).stdout().strip()

nanobind_inc = include_directories(
  nanobind_dir / 'include',
  nanobind_dir / 'ext' / 'robin_map' / 'include',
)

nanobind_lib = static_library('nanobind',
  nanobind_dir / 'src' / 'nb_combined.cpp',
  include_directories: nanobind_inc,
  dependencies: py3_dep,
  cpp_args: ['-fvisibility=hidden', '-ffunction-sections', '-fdata-sections', '-fno-strict-aliasing'],
  override_options: ['cpp_std=c++20', 'b_ndebug=if-release'],
  gnu_symbol_visibility: 'hidden',
)
nanobind_dep = declare_dependency(
  link_with: nanobind_lib,
  include_directories: nanobind_inc,
)

# ---------------------------------------------------------------------------
# libint2 + Eigen3, located by devtools/find_libint2.py (conda env, Homebrew, or
# system install; -Dlibint2_prefix=... overrides). The SCF/CC gradients need a
# libint2 built with derivative integrals, as the conda-forge `libint` package is.
# ---------------------------------------------------------------------------
libint2_search = run_command(py3,
  [meson.project_source_root() / 'devtools' / 'find_libint2.py', get_option('libint2_prefix')],
  check: false
)
if libint2_search.returncode() != 0
  error(libint2_search.stderr().strip())
endif
libint2_info = libint2_search.stdout().strip().split('\n')
libint2_prefix = libint2_info[0]
eigen_include = libint2_info[1]
libint2_caps = libint2_info[2].split(';')
message('libint2: ' + libint2_prefix + ' (max AM ' + libint2_caps[0] +
        ', 1e derivative order ' + libint2_caps[1] + ', ERI derivative order ' + libint2_caps[2] + ')')
if libint2_caps[1].to_int() < 1 or libint2_caps[2].to_int() < 1
  warning('libint2 at ' + libint2_prefix + ' lacks derivative integrals; ' +
          'energies will work but analytic gradients will raise at run time. ' +
          'Install libint from conda-forge for gradient support.')
endif

libint2_libdir = libint2_prefix / 'lib'
libint2_lib = cxx.find_library('int2', dirs: [libint2_libdir, libint2_prefix / 'lib64'], required: true)
libint2_dep = declare_dependency(
  dependencies: libint2_lib,
  include_directories: include_directories(libint2_prefix / 'include', eigen_include),
  link_args: ['-Wl,-rpath,' + libint2_libdir],
)

libint_sources = [
  'src/api/libint_api.cc',
  'src/api/ints_api.cc',
  'src/integrals/basis.cc',
  'src/integrals/nuclear_repulsion.cc',
  'src/integrals/one_electron.cc',
  'src/integrals/one_electron_deriv.cc',
  'src/integrals/two_electron.cc',
  'src/integrals/two_electron_deriv.cc',
  'src/integrals/value_at_points.cc',
]

py3.extension_module('_libint',
  libint_sources,
  include_directories: include_directories('src'),
  dependencies: [nanobind_dep, libint2_dep, py3_dep],
  cpp_args: ['-O3', '-fvisibility=hidden', '-Wno-unused-parameter', '-Wno-sign-compare'],
  override_options: ['cpp_std=c++20', 'b_ndebug=true'],
  gnu_symbol_visibility: 'hidden',
  install: true,
  subdir: 'ccpy/integrals',
)

python_sources = [
  '__init__.py',
  'atom_data.py',
  'autoaux.py',
  'basis.py',
  'df.py',
  'integrals.py',
  'molecule.py',
  'utils.py',
]

py3.install_sources(
  python_sources,
  subdir: 'ccpy/integrals'
)

install_subdir('basis_data', install_dir: py3.get_install_dir() / 'ccpy' / 'integrals')

subdir('symmetry')
