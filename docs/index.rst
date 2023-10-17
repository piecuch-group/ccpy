.. ccpy documentation master file, created by
   sphinx-quickstart on Tue Oct 17 13:30:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CCpy: A coupled-cluster package written in Python
=================================================
CCpy is a research-level Python package for performing non-relativistic electronic structure 
calculations for molecular systems using methods based on the ground-state coupled-cluster 
(CC) theory and its equation-of-motion (EOM) extension to electronic excited, attached, 
and ionized states. As a design philosophy, CCpy favors simplicity over efficiency, and this 
is reflected in the usage of computational routines that are transparent enough so that they 
can be easily used, modified, and extended, while still maintaining reasonable efficiency. 
To this end, CCpy employs a hybrid Python-Fortran programming approach made possible with the 
:code:`f2py` package, which allows one to compile Fortran code into shared object libraries 
containing subroutines that are callable from Python and interoperable with Numpy arrays. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   contact_information



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
