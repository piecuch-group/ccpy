project(
  'ccpy',
  'c', 'fortran',
  version: '0.0.3',
  license: 'GPL-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'fortran_std=legacy',
    'c_std=c17',
    'blas=openblas',
    'lapack=openblas'
  ],
)

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

f2py_script = find_program('devtools/f2py_script.sh')

min_numpy_version = '1.23.5'  # keep in sync with pyproject.toml

add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')
if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif
add_project_arguments('-Wno-unused-dummy-argument', language: 'fortran')
add_project_arguments('-Wno-unused-variable', language: 'fortran')

subdir('ccpy')
