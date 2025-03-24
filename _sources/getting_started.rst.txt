Getting Started
###############

Using the PySCF Interface
*************************
Let's build a simple example showing how we can run a ground-state
CCSD calculation for the water molecule described with the cc-pVDZ basis
set using the interface to PySCF. For this example, we will use the
symmetrically stretched geometry from
J. Olsen, P. Jørgensen, H. Koch, A. Balkova, and R. J. Bartlett, *J. Chem. Phys.* **104**, 8007 (1996).

First, we need to import the relevant modules from PySCF and prepare a RHF
calculation. This is done by first specifying the nuclear
geometry and building the :code:`Molecule` object. Next, an RHF-type :code:`MeanField`
object is constructed, which accepts :code:`Molecule` as an input in order to
execute an RHF calculation. This is all accomplished using the code below ::

        # Import relevant SCF modules from PySCF
        from pyscf import scf, gto
        # Specify the molecular geometry
        geometry = '''O  0.0   0.0       -0.0180
                      H  0.0   3.030526  -2.117796
                      H  0.0  -3.030526  -2.117796'''
        # Construct the PySCF Molecule object described within a specified basis set
        mol = gto.M(
                atom=geometry,
                basis="cc-pvdz",
                charge=0,
                spin=0,
                symmetry="C2V",
                cart=False,
                unit="Bohr",
        )
        # Construct the RHF MeanField object
        mf = scf.RHF(mol)
        # Run the RHF calculation
        mf.kernel()

In order to run CC calculations on top of this mean-field state using CCpy,
we must import the main calculation driver class, called :code:`Driver`,
using the following command ::
        
        from ccpy.drivers.driver import Driver

The :code:`Driver` object can be instantiated using the PySCF mean-field object
as an input along with the number of frozen spatial orbitals, which
should correspond to the chemical core of the molecule. For water, we
will freeze the 1s orbital of the oxygen atom. The :code:`Driver.from_pyscf()`
constructor not processes the PySCF :code:`MeanField` object to obtain and store
important information about the system needed for correlated calculations,
but also performs AO-to-MO integral transformation and sorting steps, storing the
resulting integrals as part of the :code:`Driver` object.
We can execute this driver instantiation step through the PySCF interface as follows ::

        driver = Driver.from_pyscf(mf, nfrozen=1)
        driver.system.print_info()

The above code will create the :code:`Driver` object out of the PySCF :code:`MeanField`
object and print an output of the system information. We are now ready to use the
:code:`Driver` object to run correlated CC calculations. For example, a basic CCSD
calculation is performed using ::

        driver.run_cc(method="ccsd")

Using the GAMESS Interface
**************************
Let's now run the same example using the interface to GAMESS. This is
slightly more complicated because we need to run GAMESS separately
to obtain the output log file as well as an FCIDUMP file
containing the molecular orbital integrals. To accomplish this, we should
have a working installation of GAMESS and prepare an input script
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
is convenient for your machine, for example, using the :code:`rungms`
script included in the GAMESS package distribution. The following execution
of the GAMESS program ::

        rungms h2o-ccpvdz.inp | tee h2o-ccpvdz.log

will produce the output file :code:`h2o-ccpvdz.log` as well as the FCIDUMP file
:code:`h2o-ccpvdz.FCIDUMP` located in the same directory. Now, to use the GAMESS
interface of CCpy to begin the CC calculations, we will need to again instantiate
the :code:`Driver` class using mean-field information contained within the GAMESS log
file and FCIDUMP. This can be done using the :code:`Driver.from_gamess()` constructor
as follows ::

        from ccpy.drivers.driver import Driver

        driver = Driver.from_gamess(logfile="h2o-ccpvdz.log",
                                    fcidump="h2o-ccpvdz.FCIDUMP",
                                    nfrozen=1)
        driver.system.print_info()

which again loads the :code:`Driver` object with the information about the RHF solution
computed with GAMESS, freezing the core 1s oxygen orbital for further correlated calculations.
In this case, GAMESS has already executed the AO-to-MO transformation and written the unique
integrals in Mulliken notation within the FCIDUMP file, so this construction simply reads in
the integrals from the FCIDUMP file and sorts and stores them within the :code:`Driver` object
as before. Once we have the :code:`Driver` object instantiated, we can use it to run CC
calculations. Again, a CCSD calculation is performed with ::

        driver.run_cc(method="ccsd")



