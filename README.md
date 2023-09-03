
# CCpy: A coupled-cluster package written in Python.
![image](docs/assets/img/Diagrams-CCD.png)
<p style="text-align: right;">Image from: https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html/CCM.html</p>

---
# Overview
<p align="justify">
CCpy is a research-level Python package for performing non-relativistic electronic structure calculations for molecular systems 
using methods based on the ground-state coupled-cluster (CC) theory and its equation-of-motion (EOM) extension
to electronic excited, attached, and ionized states. As a design philosophy, CCpy favors simplicity over efficiency, and this is reflected in the
usage of computational routines that are transparent enough so that they can be easily used, modified, and extended, while 
still maintaining reasonable efficiency. To this end, CCpy employs a hybrid Python-Fortran programming approach made possible
with the f2py package, which allows one to compile Fortran code into shared object libraries containing subroutines
that are callable from Python and interoperable with Numpy arrays. 
</p>

# Available Computational Options
<p align="justify">
CCpy specializes in applying the CC(P;Q) and externally corrected (ec) CC methodologies developed in the Piecuch group at 
Michigan State University. In CC(P;Q), the energetics obtained by solving the ground- or excited-state CC/EOMCC equations in
one subspace of the many-electron Hilbert space, called the P space, are corrected for the missing many-electron correlation
effects captured with the help of a complementary subspace called the Q space using the state-selective, non-iterative,
and non-perturbative energy corrections based on the CC moment expansion formalism. Currently, CCpy offers implementations
of several CC(P;Q) methods, the majority of which are aimed at converging the high-level CCSDT and EOMCCSDT energetics. 
These include the completely-renormalized (CR) methods such as the CR-CC(2,3) and CR-CC(2,4) triples and quadruples 
corrections to CCSD, the active-space CCSDt and CC(t;3) approaches, which are based on a user-defined selection of active orbitals, 
and the black-box selected configuration interaction (CI) driven and adaptive CC(P;Q) methodologies, which construct the P and Q spaces 
entering the CC(P;Q) computations using information extracted from selected CI wave functions or the adaptive CC(P;Q) moment 
expansions themselves, respectively. The ec-CC approaches on the other hand seek to converge the exact, full CI energetics
directly by solving for the T<sub>1</sub> and T<sub>2</sub> clusters in the presence of the leading T<sub>3</sub> and T<sub>4</sub> clusters extracted from an
external non-CC wave function. Current implementations of the ec-CC approaches in CCpy are designed to iterate T<sub>1</sub> and T<sub>2</sub> clusters 
in the presence of T<sub>3</sub> and T<sub>4</sub> obtained from CI wave functions of the selected CI or multireference CI types, and correct the resulting
energetics for the missing many-electron correlations using the generalized moment expansions of the ec-CC equations.
</p>

### Møller-Plesset (MP) perturbation theory
  - MP2 
  - MP3 

### Ground-state CC methodologies
  - CCD
  - CCSD
  - CCSD(T)
  - CR-CC(2,3)
  - CCSDt
  - CC(t;3)
  - CIPSI-driven CC(P;Q) aimed at converging CCSDT (see Ref. [1])
  - Adaptive CC(P;Q) aimed at converging CCSDT (see Ref. [2])
  - CCSDT
  - CR-CC(2,4)
  - CCSDTQ (available for closed shells only)
  - ec-CC-II
  - ec-CC-II<sub>3</sub> (see Ref. [3])
  - ec-CC_II<sub>3,4</sub> (see Ref. [3])

### EOMCC approaches for electronic excited, attached, and ionized states
  - EOMCCSD
  - Spin-Flip (SF) EOMCCSD
  - CR-EOMCC(2,3) and its size-intensive δ-CR-EOMCC(2,3) extension
  - EOMCCSDt
  - EOMCCSDT
  - IP-EOMCCSD(2h-1p)
  - EA-EOMCCSD(2p-1h)
  - EA-EOMCCSD(3p-2h)
  - DEA-EOMCCSD(3p-1h)

<p align="justify">
Because CCpy is primarily used for CC method development work, we use interfaces to GAMESS and Pyscf to obtain the mean-field (typically Hartree-Fock)
reference state and associated molecular orbital one- and two-electron integrals prior to performing the correlated CC calculations. All implementations
in CCpy are based on the spin-integrated spinorbital formulation and are compatible with RHF and ROHF references. 
</p>

#### References
[1] J. Chem. Phys. 155, 174114 (2021); see https://doi.org/10.1063/5.0064400; cf. also 
https://doi.org/10.48550/arXiv.2107.10994 <br />
[2] J. Chem. Phys. 159, 084108 (2023); see https://doi.org/10.1063/5.0162873; cf. also
https://doi.org/10.48550/arXiv.2306.09638 <br />
[3] J. Chem. Theory Comput. 17, 4006 (2021); see https://doi.org/10.1021/acs.jctc.1c00181; cf. also
https://doi.org/10.48550/arXiv.2102.10143  

# Installation
<p align="justify">
Installation should be simple. Simply clone this git repository and run `make install` followed by `make all` inside of it. You will
need a working gfortran compiler as well as locations for BLAS (preferably MKL) libraries, which enter in the Makefile as the environment
variable $MKLROOT. For a given computer architecture, the Intel Link Line Advisor is a useful tool to figure out what compiler flags need to be included.
(https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.jxf4xw)

If you want to run mean-field calculations using GAMESS as an input to CCpy, you will need to install the `cclib` package 
(`conda install --channel conda-forge cclib`) in order to parse the output of GAMESS calculations.  If you want to use PySCF to run
the mean-field calculation, then simply install it with `pip install pyscf`.

In all selected CI based computations, including the ec-CC-II and CIPSI-driven CC(P;Q), we currently rely on the CIPSI wave functions obtained 
using the open-source Quantum Package software (https://github.com/QuantumPackage/qp2).

### Getting Started
Documentation is scarce at the moment, so please bear with us as we put it together. You can hopefully get started using the majority of the options
available in CCpy by looking at examples found in the `test_small_molecules.py` script within the `tests` directory. In the meantime, feel free to 
e-mail me (contact details given below) if you need additional information or more specific assistance. 
</p>

# Contact Information
Karthik Gururangan - gururang@msu.edu
<p align="justify">
In addition to using GitHub's Issues feature, feel free to send me an e-mail if you have any questions about using
CCpy or are seeking additional information about its functionality. CCpy is an open-source code and any contributions
or suggestions that would improve it are welcomed and appreciated!
</p>
