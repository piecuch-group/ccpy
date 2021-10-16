
# ```CCpy``` Coupled-cluster package written in Python.
![image](Diagrams-CCD.png)
<p style="text-align: right;">Image from: https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html/CCM.html</p>

---
# Overview
Research-level implementation of ground-state coupled-cluster (CC) methods and their excited-state extensions
via the equation-of-motion (EOM) CC formalism. Supports CC(P;Q) calculations aimed at converging CCSDT energetics
for ground states and EOMCCSDT energetics for excited states. All implementations are spin-integrated and make 
use of fast BLAS routines for reasonable speed for research problems. 

# Instruction
The cleanest way to install `CCpy` is to create a conda virtual environment. To do so, run

`conda create --name <env_name> python=3.9` 

and install `numpy`, `scipy`, `mkl`, and `cclib` with the command

`conda install numpy scipy mkl` and `conda install --channel conda-forge cclib`.

Intel MKL is used to make all BLAS routines executed in Python and Fortran fast. Ensure that the Numpy installed
in the conda environment is using MKL by running `np.show_config()`. When I do this, I have the following:

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

`

This will show if Numpy is configured to use the MKL installed in the conda virtual environment. Then, to configure
Fortran to use the correct MKL, note the root in `library_dirs` and insert that path into the Makefile for `${MKLROOT}`.

If all is well, all Fortran files should compile when the make file is executed. If any MKL libraries are misnamed, this
can be fixed by going to the MKL library path and linking each executable into the correct name (via `link <OLD.so> <NEW.so>`). 

Affiliated with Piecuch Group at MSU (https://www2.chemistry.msu.edu/faculty/piecuch/)
