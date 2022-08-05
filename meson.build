project(
  'ccpy',
  'c', 'fortran',
  default_options: [
    'c_std=c99',
    'blas=MKL',
    'lapack=MKL'
  ],
)

py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()
f2py = find_program('f2py')

subdir('ccpy/utilities')
