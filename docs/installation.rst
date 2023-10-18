Installation
############
To install CCpy, you will need a Python environment with the following packages:

* Python 3.8+
* NumPy
* gfortran (:code:`conda install -c conda-forge gfortran`)
* mkl (:code:`conda install mkl`)

Currently, CCpy relies on external software to provide Hartree-Fock molecular orbitals 
and the asssociated one- and two-electron integrals as inputs to the CC calculations. 
CCpy provides interfaces that process the outputs of mean-field solutions computed using either GAMESS or 
PySCF (interfaces to other open-source software, such as psi4, will be added in the future). 

In order to use the interface to GAMESS, you must have:

* A working installation of GAMESS (https://www.msg.chem.iastate.edu/gamess/download.html)
* cclib (:code:`conda install --channel conda-forge cclib`)

In order to use the interface to PySCF, you must have:

* PySCF (:code:`pip install pyscf`)

Once you have these packages installed, you can install CCpy in the same environment using::

    make install
    make all

The former command compiles the Fortran modules used in CCpy and the latter command 
installs a locally editable copy of CCpy using pip (i.e. equivalent to :code:`pip install -e .`).
The location for MKL libraries is taken to be the local version of these libraries installed
within the Conda environment. If you need to modify this, or any other MKL linking flags,
you must edit the Fortran module Makefile in :code:`/utilities/updates/Makefile`. 
For a given computer architecture, the Intel Link Line Advisor is a useful tool to figure out 
which compiler flags need to be included 
(see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.jxf4xw).


