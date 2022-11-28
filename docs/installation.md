# Installation
The cleanest way to install `CCpy` is to create a conda virtual environment. To do so, run

`conda create --name <env_name> python=3.9` 

and install `numpy`, `scipy`, `mkl`, `pytest`, and `cclib` with the command

`conda install numpy scipy mkl pytest` and `conda install --channel conda-forge cclib`. You will 
also need access to the environment-local version of pip (e.g., `/opt/anaconda3/venv/my_venv/bin/pip`).
Using this binary, run `pip install art` to get the art package used in the title printing.

Currently, `CCpy` has interfaces to GAMESS and PySCF (an interface to Psi4 will be added
soon). Most likely, you will want PySCF as this is the most convenient source of self-consistent
field states and the resulting one- and two-body integral arrays needed for all `CCpy` calculations.
To install PySCF, run `pip install pyscf`.

Finally, move to the main directory of `ccpy` (the one that contains `setup.py`) 
and run `pip install -e .` to install the package in editable mode.

Ensure that the Numpy installed in the conda environment is using MKL by running `np.show_config()`. When I do this, I have the following:

`
blas_mkl_info:

    libraries = ['mkl_rt', 'pthread']

    library_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/lib']

    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]

    include_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/include']


blas_opt_info:

    libraries = ['mkl_rt', 'pthread']

    library_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/lib']

    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]

    include_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/include']


lapack_mkl_info:

    libraries = ['mkl_rt', 'pthread']

    library_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/lib']

    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]

    include_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/include']


lapack_opt_info:

    libraries = ['mkl_rt', 'pthread']

    library_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/lib']

    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]

    include_dirs = ['/home2/gururang/.conda/envs/ccpy_dev/include']


Supported SIMD extensions in this NumPy install:

    baseline = SSE,SSE2,SSE3

    found = SSSE3,SSE41,POPCNT,SSE42

    not found = AVX,F16C,FMA3,AVX2,AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CNL

This will show if Numpy is configured to use the MKL installed in the conda virtual environment. Then, to configure
Fortran to use the correct MKL, note the root in `library_dirs` and insert that path into the Makefile for `${MKLROOT}`. 
If all is well, all Fortran files should compile when the make file is executed. If any MKL libraries are misnamed, this
can be fixed by going to the MKL library path and linking each executable into the correct name (via `link <OLD.so> <NEW.so>`). 

Affiliated with Piecuch Group at MSU (https://www2.chemistry.msu.edu/faculty/piecuch/)
