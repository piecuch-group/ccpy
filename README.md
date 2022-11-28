
# CCpy: A coupled-cluster package written in Python.
![image](docs/assets/img/Diagrams-CCD.png)
<p style="text-align: right;">Image from: https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html/CCM.html</p>

---
# Overview
CCpy is a research-level Python package for performing non-relativistic electronic structure calculations for molecular systems 
using methods based on the ground-state coupled-cluster (CC) theory and its equation-of-motion (EOM) extension
to excited electronic states. As a design philosophy, CCpy favors simplicity over efficiency, and this is reflected in the
usage of computational routines that are transparent enough so that they can be easily used, modifiied, and extended, while 
still maintaining reasonable efficiency. To this end, CCpy operates within a hybrid Python-Fortran environment made possible
with the f2py package, which allows one to painlessly compile Fortran code into shared object libraries containing subroutines
that are callable from Python and have seamless interoperability with Numpy arrays. This approach is particularly useful when
devectorized, loop-driven implementations are used, as Python is notoriously slow at executing deep nested loops. On the other
hand, the dense tensor contractions forming the bulk of the computational cost of all CC implementations are very efficiently
implemented using standard Numpy functions, especially when the latter is compiled with efficient BLAS libraries. As a result, CCpy
can achieve a serial performance comparable to a standard Fortran implementation. 

CCpy specializes in applying the CC(P;Q) and externally corrected (ec) CC methodologies developed in the Piecuch group at MSU.
In CC(P;Q), the energetics obtained by solving the ground- or excited-state CC/EOMCC equations in
one subspace of the many-electron Hilbert space, called the P space, are corrected for the missing correlation
effects captured with the help of a complementary subspace called the Q space using the state-selective, non-iterative,
and non-perturbative energy corrections based on the CC moment expansion formalism. Currently, CCpy offers implementations
of several CC(P;Q) methods, the majority of which are aimed at converging the high-level CCSDT and EOMCCSDT energetics. 
These include the conventional CR-CC(2,3) and CR-CC(2,4) triples and quadruples corrections to CCSD, the
active-space CCSDt and CC(t;3) approaches, which are based on a user-defined selection of active orbitals, and the black-box 
selected-configuration-interaction-driven and adaptive CC(P;Q) methodologies, which construct the P and Q spaces relevant
to the CC(P;Q) theory of interest using information extracted of selected CI wave functions or the adaptive CC(P;Q) moment 
expansions themselves, respectively. The ec-CC approaches on the other hand seek to converge the exact, full CI energetics
directly by solving for the T1 and T2 clusters in the presence of the leading T3 and T4 clusters extracted from an
external non-CC wave function. Current implementations of the ec-CC approaches in CCpy are designed to iterate T1 and T2 clusters 
in the presence of T3 and T4 obtained from CI wave functions of the selected CI or multireference CI types, and correct the resulting
energetics for the missing many-electron correlations using the generalized moment expansions of the ec-CC equations.

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

Affiliated with Piecuch Group at MSU (https://www2.chemistry.msu.edu/faculty/piecuch/)
