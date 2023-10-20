Installation
############
To install CCpy, you will need a Python environment with the following packages:

* Python 3.8+
* NumPy (:code:`conda install numpy`)
* gfortran (:code:`conda install -c conda-forge gfortran`)

Currently, CCpy relies on external software to provide Hartree-Fock molecular orbitals 
and the asssociated one- and two-electron integrals as inputs to the CC calculations. 
CCpy provides interfaces that process the outputs of mean-field solutions computed using
either GAMESS or PySCF (interfaces to other open-source software, such as Psi4, will be
added in the future).

In order to use the interface to GAMESS, you must have:

* A working installation of GAMESS (https://www.msg.chem.iastate.edu/gamess/download.html)
* cclib (:code:`conda install -c conda-forge cclib`)

In order to use the interface to PySCF, you must have:

* PySCF (:code:`conda install -c conda-forge pyscf`)

Once you have these packages installed, you can install CCpy in the same environment using::

    make all
    make install

The former command compiles the Fortran modules used in CCpy and the latter command 
installs a locally editable copy of CCpy using pip (i.e. equivalent to :code:`pip install -e .`).
The location for MKL libraries is taken to be the local version of these libraries installed
within the Conda environment. If Numpy is installed from scratch and configured correctly, it
should come properly loaded with MKL libraries and this will be handled automatically.
If you need to modify this, or any other MKL linking flags, you must edit the Fortran module
Makefile in :code:`/utilities/updates/Makefile`. For a given computer architecture,
the Intel Link Line Advisor is a useful tool to figure out which compiler flags need to be included
(see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.jxf4xw).

The following is a complete sequence of steps needed to successfully install CCpy compatible with
PySCF using Numpy linked to MKL ::

    conda create --name=ccpy_env python=3.11
    conda activate ccpy_env

    conda install numpy
    conda install -c conda-forge gfortran
    conda install -c conda-forge cclib
    conda install -c conda-forge pyscf

    make all
    make install

It may happen that CCpy fails to execute the DGEMM or other BLAS operations. This
can occur if Numpy is not properly linked to the MKL libraries for its backend. A
common reason for this is if PySCF is installed before Numpy, since PySCF installs
a redundant copy of Numpy that is not linked to optimized BLAS libraries. In order to
fix this, uninstall and reinstall Numpy within the Conda environment and similarly
reinstall PySCF. Before compiling the Fortran modules, it is good to check that the
Numpy backend is linked to MKL libraries using the command ::

    import numpy
    numpy.show_config()
