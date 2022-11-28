
# CCpy: A coupled-cluster package written in Python.
![image](docs/assets/img/Diagrams-CCD.png)
<p style="text-align: right;">Image from: https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html/CCM.html</p>

---
# Overview
'''ccpy''' aims to provide a suite of Python implementations of non-relativistic
electronic structure methods for molecular systems
based on the traditional ground-state coupled-cluster (CC) theory and its excited-state extension
using the equation-of-motion (EOM) CC formalism. The driving philosophy behind CCpy is to use routines that
are transparent enough so that they can be easily used and modified by the user while still maintaining high
efficiency. As a result, all CC/EOMCC methods are implemented in the spin-integrated formalism and are
thus compatible with both restricted, unrestricted, and restricted open-shell HF references. Furthermore,
the bulk of the routines are written using Numpy, which makes use of high-performance BLAS libraries, thus
the CC/EOMCC routines in CCpy should be reasonably close to the 
computational speed of any lower-level Fortran or C++ implementation.
In addition, the DIIS-accelerated Jacobi solver for the ground-state CC nonlinear equations and the 
non-Hermitian Davidson diagonalization routine for the excited-state EOMCC equations have both in-memory
and out-of-core variants, characterized by the use of RAM and disk storage, respectively. The out-of-core
options are useful for performing computations using the higher-order methods for 
larger (e.g., > 100 orbital) systems. 

CCpy specializes in applying the CC(P;Q) hierarchy of methods developed in the Piecuch group at MSU. In the 
CC(P;Q) approaches, the energetics obtained by solving the ground- or excited-state CC/EOMCC equations in
one subspace of the many-electron Hilbert space, called the P space, are corrected for the missing correlation
effects captured with the help of a complementary subspace called the Q space using the state-selective, non-iterative,
and non-perturbative energy corrections based on the CC moment expansion formalism. The CC(P;Q) formalism includes
the highly successful completely-renormalized (CR) CC methods, including the CR-CC(2,3) triples correction and the CR-CC(2,4)
triples and quadruples correction to CCSD, but its main advantage is the ability to make unconventional choices
for the P and Q spaces, allowing one to coupled the lower-order T1 and T2 clusters with their higher-order T3 and T4
counterparts. It has been demonstrated that by using P spaces obtained by extracting information from
external wave function of the Configuration Interaction (CI) Quantum Monte Carlo (QMC), CCMC, or selected CI types, one
can devise efficient algorithms to converge the high-level CCSDT/EOMCCSDT or CCSDTQ energetics using small fractions
of higher-than-doubly excited Slater determinants in the underlying P spaces. CCpy supports calculations of these types
using the general CC(P) and CC(P;Q) modules. Furthermore, CCpy offers a novel black-box alternative called the 
adaptive CC(P;Q) approaches in which the P spaces are evolved iteratively with the help of the CC(P;Q) moment
corrections themselves.

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
