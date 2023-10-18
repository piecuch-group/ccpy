Getting Started
===============

A simple example using PySCF
----------------------------
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

In order to use CCpy, we must import the main calculation driver class of
CCpy, which is called :code:`Driver`, using the following command ::
        
        from ccpy.drivers.driver import Driver

The :code:`Driver` object can be instantiated using the PySCF mean-field object
as an input along with the number of frozen spatial orbitals, which
should correspond to the chemical core of the molecule. For water, we
will freeze the 1s orbital of the oxygen atom, therefore, we should 
execute::

        driver = Driver.from_pyscf(mf, nfrozen=1)
        driver.system.print_info()

which will create the :code:`Driver` object and print an output of the system
information. In order to execute a CCSD calculation, we can run::

        driver.run_cc(method="ccsd")

A simple example using GAMESS
-----------------------------
Let's now run the same example using the interface to GAMESS. This is
slightly more complicated because we need to run GAMESS externally
in order to obtain the output log file as well as an FCIDUMP file
containing the molecular orbital integrals. We can accomplish this
using the following GAMESS input script called :code:`h2o-ccpvdz.inp`::

         $contrl sctfyp=rhf runtyp=fcidump units=bohr ispher=+1 $end
         $system mwords=1 $end
         $basis gbasis=ccd $end
         $guess guess=huckel $end
         $data
        H2O molecule / cc-pvdz
        cnv 2
        O  8.0   0.000000   0.000000   -0.018000
        H  1.0   0.000000   3.030526   -2.117796
         $end

Run this input script using whatever configuration of GAMESS is convenient
for your machine, for example::

        /home2/gururang/gamess_R1_2023/rungms h2o-ccpvdz.inp | tee h2o-ccpvdz.log

This will produce the output file :code:`h2o-ccpvdz.log` as well as the FCIDUMP file
:code:`h2o-ccpvdz.FCIDUMP`. Now, to use the GAMESS interface of CCpy, we will need to
again instantiate the :code:`Driver` class using mean-field information contained in
the GAMESS log file and FCIDUMP. This is accomplished with the following code::

        from ccpy.drivers.driver import Driver

        driver = Driver.from_gamess(logfile="h2o-ccpvdz.log",
                                    fcidump="h2o-ccpvdz.FCIDUMP",
                                    nfrozen=1)
        driver.system.print_info()

which again loads the :code:`Driver` object with the information about the RHF solution
computed with GAMESS, freezing the core 1s oxygen orbital for further correlated calculations.
Once we have the :code:`Driver` object set up, we can easily run CC calculations as before using::

        driver.run_cc(method="ccsd")
