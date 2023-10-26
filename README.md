
# CCpy: A coupled-cluster package written in Python.

## Overview
<p align="justify">
CCpy is a research-level Python package for performing non-relativistic electronic structure calculations for molecular systems 
using methods based on the ground-state coupled-cluster (CC) theory and its equation-of-motion (EOM) extension
to electronic excited, attached, and ionized states. As a design philosophy, CCpy favors simplicity over efficiency, and this is reflected in the
usage of computational routines that are transparent enough so that they can be easily used, modified, and extended, while 
still maintaining reasonable efficiency. To this end, CCpy employs a hybrid Python-Fortran programming approach made possible
with the f2py package, which allows one to compile Fortran code into shared object libraries containing subroutines
that are callable from Python and interoperable with Numpy arrays. 
</p>

## Available Computational Options
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
  - CC3
  - CCSDt
  - CC(t;3)
  - CIPSI-driven CC(*P*;*Q*) aimed at converging CCSDT (see Ref. [1])
  - Adaptive CC(*P*;*Q*) aimed at converging CCSDT (see Ref. [2])
  - CCSDT
  - CR-CC(2,4)
  - CCSDTQ (available for closed shells only)
  - ec-CC-II
  - ec-CC-II<sub>3</sub> (see Ref. [3])
  - ec-CC_II<sub>3,4</sub> (see Ref. [3])

### EOMCC approaches for ground, excited, attached, and ionized states
  - EOMCCSD
  - Spin-Flip (SF) EOMCCSD
  - CR-EOMCC(2,3) and its size-intensive *δ*-CR-EOMCC(2,3) extension
  - EOM-CC3
  - EOMCCSDt
  - EOMCCSDT
  - IP-EOMCCSD(2h-1p)
  - IP-EOMCCSD(3h-2p)
  - EA-EOMCCSD(2p-1h)
  - EA-EOMCCSD(3p-2h)
  - DEA-EOMCCSD(3p-1h)
  - DEA-EOMCCSD(4p-2h)

<p align="justify">
Because CCpy is primarily used for CC method development work, we use interfaces to GAMESS and PySCF to obtain the mean-field (typically Hartree-Fock)
reference state and associated one- and two-electron integrals in the molecular orbital basis prior to performing the correlated CC calculations. All implementations
in CCpy are based on the spin-integrated spinorbital formulation and are compatible with RHF and ROHF references. 
</p>

#### References
[1] K. Gururangan, J. E. Deustua, J. Shen, and P. Piecuch, J. Chem. Phys. **155**, 174114 (2021) <br />
(see https://doi.org/10.1063/5.0064400; cf. also https://doi.org/10.48550/arXiv.2107.10994) <br />
[2] K. Gururangan and P. Piecuch, J. Chem. Phys. **159**, 084108 (2023) <br />
(see https://doi.org/10.1063/5.0162873; cf. also https://doi.org/10.48550/arXiv.2306.09638) <br />
[3] I. Magoulas, K. Gururangan, P. Piecuch, J. E. Deustua, and J. Shen, J. Chem. Theory Comput. **17**, 4006 (2021) <br />
(see https://doi.org/10.1021/acs.jctc.1c00181; cf. also https://doi.org/10.48550/arXiv.2102.10143)

## Installation
<p align="justify">
  
Installation instructions are provided in the CCpy documentation, which is created using `sphinx`.
Please see the `docs` directory for instructions on how to compile and view the documentation.

</p>

## CCpy Development Team

Karthik Gururangan  
Doctoral student, Department of Chemistry, Michigan State University  
e-mail: gururang@msu.edu  

Dr. J. Emiliano Deustua  
COO and Co-founder, Examol  

Professor Piotr Piecuch  
University Distinguished Professor and Michigan State University Foundation Professor, Department of Chemistry, Michigan State University  
Adjunct Professor, Department of Physics and Astronomy, Michigan State University

<p align="justify">
  
CCpy is an open-source code under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license
developed and maintained by the [Piecuch Group](https://www2.chemistry.msu.edu/faculty/piecuch/) 
at Michigan State University. In addition to using GitHub's Issues feature, feel free to send an e-mail to gururang@msu.edu if 
you have any questions about using CCpy or are seeking additional information about its functionality.

</p>
