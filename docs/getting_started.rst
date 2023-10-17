Getting Started
===============

A Simple Example
----------------
Let's build a simple example showing how we can run a ground-state
CC calculation for the water molecule described with the cc-pVDZ basis
set using the interface to PySCF.

First, we need to import the relevant modules from PySCF ::

        from pyscf import scf, gto

and set up the molecular geometry ::
        
        geometry = '''O  0.0   0.0       -0.0180
                      H  0.0   3.030526  -2.117796
                      H  0.0  -3.030526  -2.117796'''

Next, we instantiate the PySCF molecule object and run the RHF calculation ::

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

In order to use CCpy, we must import the main driver class using ::
        
        from ccpy.drivers.driver import Driver

and we initialize it using the PySCF molecule class with ::
        
        driver = Driver.from_pyscf(mf, nfrozen=0)

An output of the system information can be printed using ::
        
        driver.system.print_info()

In order to execute a CCSD calculation, we can run ::

        driver.run_cc(method="ccsd")


