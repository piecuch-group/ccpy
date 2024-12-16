
# CCpy: A coupled-cluster package written in Python.

## Overview
<p align="justify">
CCpy is a research-level Python package for performing nonrelativistic and spin-free scalar relativistic electronic structure calculations for molecular systems
using methods based on the ground-state coupled-cluster (CC) theory and its equation-of-motion (EOM) extension
to electronic excited, attached, and ionized states. CCpy employs a hybrid Python-Fortran programming approach made possible
with the f2py package, which allows one to compile Fortran code into shared object libraries containing subroutines
that are callable from Python and interoperable with Numpy arrays.

CCpy provides easy-to-use interfaces to both PySCF and GAMESS for obtaining the mean-field (typically Hartree-Fock) reference state and associated transformed
one- and two-electron integrals in the molecular orbital basis that are used to set up the correlated CC calculations. A general interface that can be used to 
initialize CCpy calculations using reference state information and one- and two-electron integrals provided by an FCIDUMP file is also included. 

CCpy is distributed as an official extension module of PySCF (see https://pyscf.org/install.html#extension-modules).
</p>

## Available Computational Options
<p align="justify">
Below, we list the computational options that are currently available in CCpy (see the dropdown menus below along with the
`tests` directory for sample input scripts). All implementations in CCpy are based on the spin-integrated spinorbital 
formulation and are compatible with RHF/ROHF and UHF references, unless otherwise indicated.
</p>

### Møller-Plesset (MP) perturbation theory
<details>
<summary>MP2</summary>

### Summary

<p align="justify">
Second-order MBPT energy correction. Compatible with RHF and UHF only.
</p>

### Example Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    # Load CCpy driver from PySCF
    driver = Driver.from_pyscf(mf, nfrozen=0)
    # Run MP2 calculation
    driver.run_mbpt(method="mp2")
```
### Reference
</details>

<details>
<summary>MP3</summary>

### Summary

<p align="justify">
Third-order MBPT energy correction. Compatible with RHF and UHF only.
</p>

### Example Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # Load CCpy driver from PySCF
    driver = Driver.from_pyscf(mf, nfrozen=0)
    # Run MP3 calculation
    driver.run_mbpt(method="mp3")
```
### Reference
</details>

### Ground-state CC methodologies
<details>
<summary>CCD</summary>

### Summary

<p align="justify">
The CC with doubles (CCD) method truncates the cluster operator as T = T<sub>2</sub>.
It has iterative computational costs that scale as
n<sub>o</sub><sup>2</sup>n<sub>u</sub><sup>4</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals.
Due to the importance of pair correlations in the many-electron problem, the
CCD approximation was first introduced in Prof. Čížek's landmark 1966 paper
under the name coupled-pair many-electron theory, or CPMET. Although CCD is
often superceeded by the more accurate CC with singles and doubles (CCSD) method,
which has the same computational scaling, CCD is still relevant to modern CC
calculations within the context of correlating orbital-optimized reference
functions, as in Brückner CCD.
</p>

### Example Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCD calculation
    driver.run_cc(method="ccd")
```
### Reference
1. J. Čížek, *J. Chem. Phys.* **45**, 4256 (1966).
</details>

<details>
<summary>CCSD</summary>

### Summary

<p align="justify">
The CC with singles and doubles (CCSD) method approximates the cluster
operator as T = T<sub>1</sub> + T<sub>2</sub>. It is the most commonly used truncation level
in the CC hierarchy and often forms the starting point for more sophisticated
treatments of many-electron correlation effects. CCSD has iterative computational costs that
scale as n<sub>o</sub><sup>2</sup>n<sub>u</sub><sup>4</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals.
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSD calculation
    driver.run_cc(method="ccsd")
```
### References

1. G. D. Purvis and R. J. Bartlett, *J. Chem. Phys.* **76**, 1910 (1982).
2. J. M. Cullen and M. C. Zerner, *J. Chem. Phys.* **77**, 4088 (1982).
3. G. E. Scuseria, A. C. Scheiner, T. J. Lee, J. E. Rice, and H. F. Schaefer, *J. Chem. Phys.* **86**, 2881 (1987).
4. P. Piecuch and J. Paldus, *Int. J. Quantum Chem.* **36**, 429 (1989).
</details>

<details>
<summary>CCSDT</summary>

### Summary
<p align="justify">
The CC with singles, doubles, and triples (CCSDT) method approximates the cluster
operator as T = T<sub>1</sub> + T<sub>2</sub> + T<sub>3</sub>. CCSDT is a high-level
method capable of providing nearly exact results for closed-shell molecules
as well as chemically accurate energetics for single bond breaking and a variety
of open-shell systems. CCSDT has iterative computational costs that scale as
n<sub>o</sub><sup>3</sup>n<sub>u</sub><sup>5</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals.
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSDT calculation
    driver.run_cc(method="ccsdt")
```

### References
1. M. R. Hoffmann and H. F. Schaefer, *Adv. Quantum Chem.* **18**, 207 (1986).
2. J. Noga and R. J. Bartlett, *J. Chem. Phys.* **86**, 7041 (1987).
3. G. E. Scuseria and H. F. Schaefer, *Chem. Phys. Lett.* **152**, 382 (1988).
4. J. D. Watts and R. J. Bartlett, *J. Chem. Phys.* **93**, 6104 (1990).

</details>

<details>
<summary>CCSDTQ</summary>

### Summary
<p align="justify">
The CC with singles, doubles, triples, and quadruples (CCSDTQ) method
approximates the cluster operator as
T = T<sub>1</sub> + T<sub>2</sub> + T<sub>3</sub> + T<sub>4</sub>.
CCSDTQ is a very high-level method and is often capable of providing
near-exact energetics for most problems of chemical interest, as long
as the number of strongly correlated electrons is not too large (for
methods designed to treat genuine strong correlations, see the
approximate coupled-pair, or ACP approaches).
CCSDTQ has iterative computational costs that scale as
n<sub>o</sub><sup>4</sup>n<sub>u</sub><sup>6</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals.
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSDTQ calculation
    driver.run_cc(method="ccsdtq")
```

### References
1. N. Oliphant and L. Adamowicz, *J. Chem. Phys.* **95**, 6645 (1991).
2. S. A. Kucharski and R. J. Bartlett, *Theor. Chem. Acc.* **80**, 387 (1991).
3. S. A. Kucharski and R. J. Bartlett, *J. Chem. Phys.* **97**, 4282 (1992).
4. P. Piecuch and L. Adamowicz, *J. Chem. Phys.* **100**, 5792 (1994).

</details>

<details>
<summary>CCSD(T)</summary>

### Summary

<p align="justify">
The CCSD(T) method corrects the CCSD energy for the correlation effects
due to T<sub>3</sub> clusters using formulas derived using many-body perturbation
theory (MBPT). In particular, the CCSD(T) correction includes the leading
4th-order energy correction for T<sub>3</sub> along with 5th-order contribution
due to disconnected triples. The inclusion
of the latter term distinguishes CCSD(T) from its CCSD[T] precedessor.
CCSD(T) has noniterative computational costs that
scale as n<sub>o</sub><sup>3</sup>n<sub>4</sub><sup>4</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals.
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSD calculation
    driver.run_cc(method="ccsd")
    # perform CCSD(T) correction
    driver.run_ccp3(method="ccsd(t)")
```
### References

1. K. Raghavachari, G. W. Trucks, J. A. Pople, and M. Head-Gordon, *Chem. Phys. Lett.* **157**, 479 (1989).
2. J. F. Stanton, *Chem. Phys. Lett.* **281**, 130 (1997).
3. S. A. Kucharski and R. J. Bartlett, *J. Chem. Phys.* **108**, 5243 (1998).
4. T. D. Crawford and J. F. Stanton, *Int. J. Quantum Chem.* **70**, 601 (1998).
</details>

<details>
<summary>CR-CC(2,3)</summary>

### Summary

<p align="justify">
The CR-CC(2,3) approach is a nonperturbative and noniterative correction to the
CCSD energetics that accounts for the correlation effects due to T<sub>3</sub>
clusters using formulas derived from the biorthogonal moment energy expansions of CC
theory. In particular, CR-CC(2,3) represents the most robust scheme to noniteratively
include the effects of connected triples on top of CCSD, and it is capable of providing an
accurate description of closed-shell molecules in addition to commonly encountered
multireference problems, such as single bond breaking and open-shell radical
and diradical species, which are generally beyond the scope of perturbative
methods like CCSD(T). The CR-CC(2,3) triples correction uses noniterative steps
that scale as n<sub>o</sub><sup>3</sup>n<sub>4</sub><sup>4</sup>, where n<sub>o</sub> is
the number of correlated occupied orbitals and n<sub>u</sub> is the number of
correlated unoccupied orbitals, however, due to the precise form of the
expressions defining the CR-CC(2,3) triples correction, it is approximately
twice as expensive as its CCSD(T) counterpart. One must also solve the companion
left-CCSD system of linear equations (roughly as expensive as CCSD) prior
to computing the CR-CC(2,3) correction.

The CR-CC(2,3) calculation returns four distinct energetics, labelled as
CR-CC(2,3)<sub>X</sub>, for X = A, B, C, and D, where each variant A-D corresponds to
a different treatment of the energy denominators entering the formula for
the CR-CC(2,3) triples correction. The variant CR-CC(2,3)<sub>A</sub> uses the simplest
Møller-Plesset form of the energy denominator and is equivalent to the method
called CCSD(2)<sub>T</sub>. Meanwhile, the CR-CC(2,3)<sub>D</sub> result, which employs
the full Epstein-Nesbet energy denominator, is generally most accurate and often
reported as the CR-CC(2,3) energy (or by its former name, CR-CCSD(T)<sub>L</sub>).
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSD calculation
    driver.run_cc(method="ccsd")
    # build CCSD similarity-transformed Hamiltonian (this overwrites original MO integrals)
    driver.run_hbar(method="ccsd")
    # run companion left-CCSD calculation
    driver.run_leftcc(method="left_ccsd")
    # run CR-CC(2,3) triples correction
    driver.run_ccp3(method="crcc23")
```
### References

1. P. Piecuch and M. Włoch, *J. Chem. Phys.* **123**, 224105 (2005).
2. P. Piecuch, M. Włoch, J. R. Gour, and A. Kinal, *Chem. Phys. Lett* **418**, 467 (2006).
3. M. Włoch, M. D. Lodriguito, P. Piecuch, and J. R. Gour, *Mol. Phys.* **104**, 2149 (2006), **104**, 2991 (2006) [Erratum].
4. M. Włoch, J. R. Gour, and P. Piecuch, *J. Phys. Chem. A.* **111**, 11359 (2007).
5. P. Piecuch, J. R. Gour, and M. Włoch, *Int. J. Quantum Chem.* **108**, 2128 (2008).
</details>

<details>
<summary>CR-CC(2,4)</summary>

### Summary

### Sample Code

### References

</details>

<details>
<summary>CC3</summary>

### Summary

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CC3 calculation
    driver.run_cc(method="cc3")
```
### References

</details>

<details>
<summary>CCSDt</summary>

### Summary
The active-orbital-based CCSDt calculation

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["F", (0.0, 0.0, -2.66816)],
              ["F", (0.0, 0.0, 2.66816)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set the active space
    driver.set_active_space(nact_occupied=5, nact_unoccupied=8)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSDt calculation
    driver.run_cc(method="ccsdt1")
```
or
```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver
    from ccpy.utilities.pspace import get_active_triples_pspace

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["F", (0.0, 0.0, -2.66816)],
              ["F", (0.0, 0.0, 2.66816)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set the active space
    driver.set_active_space(nact_occupied=5, nact_unoccupied=8)
    # get triples entering P space corresponding to the CCSDt truncation scheme
    t3_excitations = get_active_triples_pspace(driver.system,
                                              driver.system.reference_symmetry,
                                              num_active=1)
    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # Run CC(P) calculation equivalent to CCSDt
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
```
The latter CC(*P*)-based approach offers two advantages: (i) it can take advantage of
the Abelian point group symmetry of a molecule by restricting the CC calculation to
include only those triply excited cluster amplitudes belonging to a particular irrep,
as specified by the keyword `target_irrep` and (ii) it can be used to perform other
types of active-orbital-based CCSDt calculations based on restricting `num_active`
occupied/unoccupied indices to the active set. The standard choice of
`num_active=1` results in the usual CCSDt method, however `num_active=2` and
`num_active=3` result in the CCSDt(II) and CCSDt(III) approaches introduced in Ref. [X].

### References

</details>

<details>
<summary>CC(t;3)</summary>

### Summary

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["F", (0.0, 0.0, -2.66816)],
              ["F", (0.0, 0.0, 2.66816)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set the active space
    driver.set_active_space(nact_occupied=5, nact_unoccupied=8)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CCSDt calculation
    driver.run_cc(method="ccsdt1")
    # build CCSD-like similarity-transformed Hamiltonian (this overwrites original MO integrals)
    driver.run_hbar(method="ccsd")
    # run companion left-CCSD-like calculation
    driver.run_leftcc(method="left_ccsd")
    # run CC(t;3) triples correction
    driver.run_ccp3(method="cct3")
```
or
```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver
    from ccpy.utilities.pspace import get_active_triples_pspace

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["F", (0.0, 0.0, -2.66816)],
              ["F", (0.0, 0.0, 2.66816)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set the active space
    driver.set_active_space(nact_occupied=5, nact_unoccupied=8)
    # get triples entering P space corresponding to the CCSDt truncation scheme
    t3_excitations = get_active_triples_pspace(driver.system,
                                              driver.system.reference_symmetry)
    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # Run CC(P) calculation equivalent to CCSDt
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    # build CCSD-like similarity-transformed Hamiltonian (this overwrites original MO integrals)
    driver.run_hbar(method="ccsd")
    # run companion left-CCSD-like calculation
    driver.run_leftcc(method="left_ccsd")
    # run CC(t;3) triples correction
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations)
```
As in the case of the CCSDt calculations, the general CC(*P*) approach allows one
to perform alternative active-orbital-based truncation schemes of the CCSDt(II)
and CCSDt(III) types in addition to the standard CCSDt method. The corresponding
CC(*P*;*Q*) corrections result in the CC(t;3)(II), CC(t;3)(III), and CC(t;3)
approaches, respectively.

### References

</details>

<details>
<summary>CIPSI-driven CC(P;Q) aimed at converging CCSDT</summary>

### Summary

### Sample Code

```python3
from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_pspace_from_cipsi

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cipsi_ccpq_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-Re.log",
        onebody=TEST_DATA_DIR + "/h2o/onebody.inp",
        twobody=TEST_DATA_DIR + "/h2o/twobody.inp",
        nfrozen=0,
    )

    civecs = TEST_DATA_DIR + "/h2o/civecs-10k.dat"
    _, t3_excitations, _ = get_pspace_from_cipsi(civecs, driver.system, nexcit=3)

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)
```

### References
1. K. Gururangan, J. E. Deustua, J. Shen, and P. Piecuch, J. Chem. Phys. **155**, 174114 (2021) <br />
(see https://doi.org/10.1063/5.0064400; cf. also https://doi.org/10.48550/arXiv.2107.10994) <br />
</details>

<details>
<summary>Adaptive CC(P;Q) aimed at converging CCSDT</summary>

### Summary

### Sample Code

```python3
import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptDriver

def test_adaptive_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    percentages = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.options["RHF_symmetry"] = False
    adaptdriver = AdaptDriver(driver, percentage=percentages)
    adaptdriver.options["energy_tolerance"] = 1.0e-08
    adaptdriver.options["two_body_approx"] = True
    adaptdriver.run()
```
### References

1. K. Gururangan and P. Piecuch, J. Chem. Phys. **159**, 084108 (2023) <br />
(see https://doi.org/10.1063/5.0162873; cf. also https://doi.org/10.48550/arXiv.2306.09638) <br />
</details>

<details>
<summary>CC4</summary>

### Summary
<p align="justify">
Approximate CC method with quadruples. Currently compatible with RHF references only.
</p>

### Sample Code

```python3
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # get the CCpy driver object using PySCF meanfield
    driver = Driver.from_pyscf(mf, nfrozen=1)

    # set calculation parameters
    driver.options["energy_convergence"] = 1.0e-07 # (in hartree)
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 80

    # run CC4 calculation
    driver.run_cc(method="cc4")
```
### References
</details>

#### Externally Corrected (ec) CC Approaches

<details>
<summary>CIPSI-driven ec-CC-II and ec-CC-II<sub>3</sub> </summary>

### Summary

### Sample Code

```python3
from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_pspace_from_cipsi

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eccc23_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-Re.log",
        onebody=TEST_DATA_DIR + "/h2o/onebody.inp",
        twobody=TEST_DATA_DIR + "/h2o/twobody.inp",
        nfrozen=0,
    )

    civecs = TEST_DATA_DIR + "/h2o/civecs-10k.dat"
    _, t3_excitations, _ = get_pspace_from_cipsi(civecs, driver.system, nexcit=3)

    driver.run_eccc(method="eccc2", ci_vectors_file=civecs)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)
```

### References
1. I. Magoulas, K. Gururangan, P. Piecuch, J. E. Deustua, and J. Shen, J. Chem. Theory Comput. **17**, 4006 (2021) <br />
(see https://doi.org/10.1021/acs.jctc.1c00181; cf. also https://doi.org/10.48550/arXiv.2102.10143)
</details>

#### Approximate Coupled-Pair (ACP) Approaches
  - ACCD
  - ACCSD
  - ACCSDt
  - ACC(2,3)
  - ACC(t;3)

### EOMCC approaches for ground, excited, attached, and ionized states
  - EOMCCSD
  - CR-EOMCC(2,3) and its size-intensive *δ*-CR-EOMCC(2,3) extension
  - EOMCCSD(T)(a)*
  - EOM-CC3
  - EOMCCSDt
  - Excited-state CC(t;3)
  - Adaptive CC(*P*;*Q*) aimed at converging EOMCCSDT
  - EOMCCSDT
  - SF-EOMCCSD
  - SF-EOMCC(2,3)
  - IP-EOMCCSD(2h-1p)
  - IP-EOMCCSD(T)(a)*
  - Active-space IP-EOMCCSD(3h-2p){N<sub>o</sub>} (also known as IP-EOMCCSDt)
  - IP-EOMCCSD(3h-2p)
  - IP-EOMCCSDT
  - EA-EOMCCSD(2p-1h)
  - EA-EOMCCSD(T)(a)*
  - Active-space EA-EOMCCSD(3p-2h){N<sub>u</sub>} (also known as EA-EOMCCSDt)
  - EA-EOMCCSD(3p-2h)
  - EA-EOMCCSDT
  - DEA-EOMCCSD(3p-1h)
  - DEA-EOMCCSD(4p-2h)
  - DIP-EOMCCSD(3h-1p)
  - DIP-EOMCCSD(4h-2p)
  - DIP-EOMCCSD(T)(a)(4h-2p)
  - DIP-EOMCCSDT(4h-2p)

## Installation
<p align="justify">

CCpy is currently run and tested on Linux and Mac OS devices. Linux users (including WSL users)
can choose to install a pre-compiled version of CCpy from the PyPI server (simplest option) or
download the source code and install it manually. For now, Mac OS users must download and install
the source code (wheels for Mac OS will be uploaded to PyPI in the near future).

### Installing from PyPI
For Linux machines, the
latest version of CCpy available on PyPI is obtained by running

```commandline
pip install coupled-cluster-py
```
### Installing via Source Code

#### Step 0: Set up the environment
In order to install CCpy from source, we recommend creating a new environment for CCpy by running
```commandline
conda create --name=ccpy_env python=3.12
```
You need reasonably up-to-date Fortran/C compilers as well as `openblas`, `cmake`, and `pkgconfig`.
If you already have these packages installed, then you can skip this step. Otherwise, local copies 
of these packages can be obtained within the Conda environment using
```commandline
conda install -c conda-forge gfortran
conda install openblas
conda install pkgconfig cmake
```
Now, we are ready to install CCpy.
#### Step 1: Clone the CCpy repository
```commandline
git clone https://github.com/piecuch-group/ccpy.git
```
Next, enter the `ccpy` directory
```commandline
cd ccpy
```
#### Step 2: Install dependencies
Dependencies are listed in `requirements-dev.txt`. You can install all of them via
```commandline
pip install -r requirements-dev.txt
```
#### Step 3: Build CCpy
Run the following command to install CCpy in editable mode (via `--editable`). This way, the
Meson backend will automatically update the package with any changes you make to either Python 
or Fortran/C code without any additional compilation steps. Additional details about developing 
within CCpy can be found on the online documentation page.
```commandline
pip install --no-build-isolation --verbose --editable .
```
Note: If Meson is having issues with finding `openblas`, make sure that the environment variable 
`PKG_CONFIG_PATH` points to the directory that includes the `openblas.pc` file. This should be 
located within `openblas/lib`, or something similar. 
</p>

## CCpy Development Team

Karthik Gururangan\
Doctoral student, Department of Chemistry, Michigan State University
e-mail: gururang@msu.edu\
(lead developer)

Dr. J. Emiliano Deustua\
COO and Co-founder, Examol\
(co-developer)

Professor Piotr Piecuch\
University Distinguished Professor and MSU Research Foundation Professor, Department of Chemistry, Michigan State University\
Adjunct Professor, Department of Physics and Astronomy, Michigan State University\
e-mail: piecuch@chemistry.msu.edu\
(co-developer and principal investigator)

Additional contributors: Tiange Deng (doctoral student, Department of Chemistry, Michigan State University; ACCD and ACCSD options).

## Acknowledgements

We acknowledge support from the Chemical Sciences, Geosciences and Biosciences Division, Office of Basic Energy Sciences, Office of Science, U.S. Department of Energy
(Grant No. DE-FG02-01ER15228 to Piotr Piecuch).

<p align="justify">

CCpy is an open-source code under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license
developed and maintained by the [Piecuch Group](https://www2.chemistry.msu.edu/faculty/piecuch/)
at Michigan State University.

</p>
