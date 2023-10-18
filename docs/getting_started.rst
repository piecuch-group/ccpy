Getting Started
===============

Using the PySCF Interface
----------------------------
Let's build a simple example showing how we can run a ground-state
CCSD calculation for the water molecule described with the cc-pVDZ basis
set using the interface to PySCF. For this example, we will use the
symmetrically stretched geometry of Reference [1].

First, we need to import the relevant modules from PySCF and prepare a RHF
calculation in calculation. This is done by first specifying the nuclear
geometry and building the :code:`Molecule` object. Next, the :code:`Molecule`
object is used build the RHF and :code:`MeanField` object, which then runs the
mean-field calculation. This is all accomplished using the code below ::

        from pyscf import scf, gto

        geometry = '''O  0.0   0.0       -0.0180
                      H  0.0   3.030526  -2.117796
                      H  0.0  -3.030526  -2.117796'''

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

In order to run CC calculations on top of this mean-field state using CCpy,
we must import the main calculation driver class, called :code:`Driver`,
using the following command ::
        
        from ccpy.drivers.driver import Driver

The :code:`Driver` object can be instantiated using the PySCF mean-field object
as an input along with the number of frozen spatial orbitals, which
should correspond to the chemical core of the molecule. For water, we
will freeze the 1s orbital of the oxygen atom. We can accomplish this by
using the PySCF constructor for the :code:`Driver` class accessed using
:code:`Driver.from_pyscf()` as follows ::

        driver = Driver.from_pyscf(mf, nfrozen=1)
        driver.system.print_info()

The above code will create the :code:`Driver` object out of the PySCF mean-field
and print an output of the system information. In order to execute a CCSD
calculation, we can run ::

        driver.run_cc(method="ccsd")

Using the GAMESS Interface
-----------------------------
Let's now run the same example using the interface to GAMESS. This is
slightly more complicated because we need to run GAMESS externally
in order to obtain the output log file as well as an FCIDUMP file
containing the molecular orbital integrals. In order to do this, we should
have a working installation of GAMESS and then prepare an input script
to run the RHF calculation and produce the associated FCIDUMP. This is
done using the following GAMESS input, called :code:`h2o-ccpvdz.inp` ::

         $contrl scftyp=rhf runtyp=fcidump units=bohr ispher=+1 $end
         $system mwords=1 $end
         $basis gbasis=ccd $end
         $guess guess=huckel $end
         $data
        H2O molecule / cc-pvdz
        cnv 2
        O  8.0   0.000000   0.000000   -0.018000
        H  1.0   0.000000   3.030526   -2.117796
         $end

Now, we should run this input using whatever configuration of GAMESS
is convenient for your machine. For example, ::

        /home2/gururang/gamess_R1_2023/rungms h2o-ccpvdz.inp | tee h2o-ccpvdz.log

will produce the output file :code:`h2o-ccpvdz.log` as well as the FCIDUMP file
:code:`h2o-ccpvdz.FCIDUMP` located in the same directory. Now, to use the GAMESS
interface of CCpy to begin the CC calculation, we will need to again instantiate
the :code:`Driver` class using mean-field information contained in the GAMESS log
file and FCIDUMP. This can be done using the :code:`Driver.from_gamess()` constructor
as follows ::

        from ccpy.drivers.driver import Driver

        driver = Driver.from_gamess(logfile="h2o-ccpvdz.log",
                                    fcidump="h2o-ccpvdz.FCIDUMP",
                                    nfrozen=1)
        driver.system.print_info()

which again loads the :code:`Driver` object with the information about the RHF solution
computed with GAMESS, freezing the core 1s oxygen orbital for further correlated calculations.
Once we have the :code:`Driver` object set up, we can easily run CC calculations as before using::

        driver.run_cc(method="ccsd")

