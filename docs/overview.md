# Overview

CCpy aims to provide a suite of Python implementations of non-relativistic
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
