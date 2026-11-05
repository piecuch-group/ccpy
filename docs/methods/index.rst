The methods API: a calculation guide
====================================

This guide describes the object-oriented API exported by ``ccpy.methods`` and
re-exported by ``ccpy``. It covers the implementation in this checkout, including
native DF-SCF, FNO preparation, ADC, EOMCC and STEOM. The older ``Driver.run_*``
interface remains available, but its method strings, option names and state
indices must not be mixed with the interface described here.

Start with :doc:`setup`, choose a method from :doc:`ground` or :doc:`excited`,
then use :doc:`guesses` to select excited roots. :doc:`spaces` explains frozen
cores, FNOs and selected determinants; :doc:`properties` covers left states,
densities and response. :doc:`multipoles` covers relaxed moments and IR
intensities.
:doc:`gradients` details analytic and numerical derivatives;
:doc:`geometry` covers minima, transition states and two-sided IRC validation
through pysisyphus. :doc:`frequencies` gives the full harmonic vibrational
analysis workflow. :doc:`options` documents solver controls,
restarts and extending the API. :doc:`catalog` lists every exported class and
its actual constructor, dependencies and equation implementation.

.. toctree::
   :maxdepth: 2

   setup
   ground
   excited
   guesses
   spaces
   properties
   multipoles
   gradients
   geometry
   frequencies
   options
   catalog

A complete first calculation
----------------------------

This example uses the native integrals and SCF implementation. No PySCF object
is needed.

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD, CRCC23, EOMCCSD

   mol = Molecule(
       "O 0 0 0; H 0 0 .96; H .93 0 -.24",
       "cc-pvdz", auxiliary_basis="cc-pvdz-rifit",
   )
   mf = RHF(mol)
   cc = CCSD(mf, nfrozen=1, amp_convergence=1.e-8,
             energy_convergence=1.e-9)
   ground_total = cc.energy()
   corrected_ground_total = CRCC23(cc).energy()

   eom = EOMCCSD(cc)
   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 3}, roots=[0, 2])
   excited_totals = eom.energy()
   print(ground_total, corrected_ground_total)
   print(eom.omega, excited_totals, eom.root_converged)

With ``symmetry=False`` (the default), use the C1 irrep label ``"A"``.
All electronic energies are in hartree. Geometry input defaults to angstrom;
nuclear gradients and displacement arrays default to bohr conventions described
in :doc:`properties`.

Calling ``energy()`` triggers the required upstream calculations. Calling it
again reuses a converged, current result. Construction can itself run SCF;
FNO-enabled construction also prepares the low-level full and truncated spaces.
``solve()`` explicitly runs the equations and returns the method object.
Inspect ``converged`` and, for excited methods, ``root_converged``: ordinary CC/EOM
wrappers can return an unconverged iterate rather than raise an exception.

What is shared
--------------

.. code-block:: text

   Molecule -> RHF / ROHF / UHF -> Reference
                                    |
                                    +-> MP2 / MP3 / MP4
                                    +-> ADC2 / IPADC2 / EAADC2
                                    +-> CC -> HBar -> EOM -> left EOM -> correction
                                         |      |
                                         |      +-> left CC / response
                                         +-> ground correction / STEOM

A ``Reference`` owns the determinant, orbital partition and bare DF factors.
A CC method owns ``T`` and caches HBars and its default left method. EOM owns
``R`` and ``omega``; its left method owns ``L``. A correction owns additive
energy increments. These objects are distinct calculations with shared upstream
data, rather than one global driver containing unrelated states.

Native DF versus materialized equations
---------------------------------------

All native references start from factorized integrals. Some methods still run
four-index equations and materialize integrals, including ``vvvv``. They issue
``DFPortWarning``. They can still benefit from FNO dimension reduction, but
are not fully DF implementations. Find the current class-level list with:

.. code-block:: python

   from ccpy import methods_requiring_df_port
   print(methods_requiring_df_port())

The catalog records this flag and the underlying kernel. Instance options can
change it, notably ``CCP3(two_body_approx=False)``. A method being constructible
with a reference does not establish benchmarked accuracy or spin/reference
coverage for every equation kernel. Explicit restrictions and research-level
implementations are identified in the following chapters.

Runnable recipes and documentation maintenance
----------------------------------------------

``examples/methods_api/run_calculation.py`` provides small runnable ground,
neutral/charged EOM, ADC, FNO, selected-space, STEOM and response calculations.
Run from an installed checkout, for example:

.. code-block:: bash

   python examples/methods_api/run_calculation.py --recipe fno --basis 6-31g

Regenerate the complete class catalog after changing API definitions:

.. code-block:: bash

   python docs/methods/generate_catalog.py
   python docs/methods/generate_catalog.py --check

The catalog is a source-derived inventory, not a claim that every possible
combination has been numerically validated. The handwritten recipes explain
compatible standard chains and the places where explicit choices matter.
