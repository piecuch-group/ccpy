"""PyTest Module for testing some of the routines in CCpy"""

import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import sys

CCPY_DIR = '/home2/gururang/CCpy'
sys.path.append(CCPY_DIR+'/src')
from system import build_system
from integrals import get_integrals
from parser_module import parse_input_file

class H2O(unittest.TestCase):

    def setUp(self):
        print('RUNNING TESTS on H2O-Re / cc-PVDZ')
        self.work_dir = CCPY_DIR+'/tests/H2O-Re-pVDZ'
        self.sys = build_system(self.work_dir+'/h2o-Re.log',0)
        self.ints = get_integrals(self.work_dir+'/onebody.inp',self.work_dir+'/twobody.inp',self.sys)
    
    def test_ccsd(self):        
        from ccsd_module import ccsd
        _ , Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,\
                        maxit=80,diis_size=6)
        self.assertAlmostEqual(Eccsd,-76.238116,places=5)

    def test_crcc23(self):
        from ccsd_module import ccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc23_module import crcc23
        cc_t, Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,self.ints,self.sys)
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,flag_RHF=False)
        self.assertAlmostEqual(Ecrcc23[0]['D'],-76.241516,places=5)

    def test_ccsdt(self):        
        from ccsdt_module import ccsdt
        _ , Eccsdt = ccsdt(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,\
                        maxit=80,diis_size=6)
        self.assertAlmostEqual(Eccsdt,-76.241367,places=5)

class F2plus(unittest.TestCase):

    def setUp(self):
        print('RUNNING TESTS ON F2+/6-31G')
        self.work_dir = CCPY_DIR+'/tests/F2+-1.0-631g'
        self.sys = build_system(self.work_dir+'/F2+-1.0-6-31G.log',2)
        self.ints = get_integrals(self.work_dir+'/onebody.inp',self.work_dir+'/twobody.inp',self.sys)
    
    def test_ccsd(self):        
        from ccsd_module import ccsd
        _ , Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,\
                        maxit=80,diis_size=6)
        self.assertAlmostEqual(Eccsd,-198.331929,places=5)

    def test_crcc23(self):
        from ccsd_module import ccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc23_module import crcc23
        cc_t, Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,self.ints,self.sys)
        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,flag_RHF=False)
        self.assertAlmostEqual(Ecrcc23[0]['D'],-198.343268,places=5)

    def test_ccsdt(self):        
        from ccsdt_module import ccsdt
        _ , Eccsdt = ccsdt(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,\
                        maxit=80,diis_size=6)
        self.assertAlmostEqual(Eccsdt,-198.342107,places=5)


class CHplus(unittest.TestCase):

    def setUp(self):
        print('RUNNING TESTS on CHplus-Re / olsen')
        self.work_dir = CCPY_DIR+'/tests/chplus-1.0-olsen'
        self.sys = build_system(self.work_dir+'/chplus_re.log',0)
        self.ints = get_integrals(self.work_dir+'/onebody.inp',self.work_dir+'/twobody.inp',self.sys)

    def test_creomcc23(self):
        from ccsd_module import ccsd
        from eomccsd_module import eomccsd
        from HBar_module import HBar_CCSD
        from left_ccsd_module import left_ccsd
        from crcc23_module import crcc23

        cc_t, Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        self.assertAlmostEqual(Eccsd,-38.017670,places=5)

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,self.ints,self.sys)
        
        cc_t, omega = eomccsd([15,0,0,0],H1A,H1B,H2A,H2B,H2C,cc_t,self.ints,self.sys,\
                        tol=1.0e-07,\
                        noact=3,nuact=3,\
                        maxit=300)

        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,\
                        shift=0.0,tol=1.0e-07,maxit=500,\
                        diis_size=6,nroot=len(omega),omega=omega,\
                        eom_tol=1.0e-07,eom_lccshift=0.0,eom_maxit=500)
        Ecrcc23,E23 = crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,self.ints,self.sys,flag_RHF=False,\
                        nroot=len(omega),omega=omega)

        self.assertAlmostEqual(omega[0],0.119829,places=5)
        self.assertAlmostEqual(omega[1],0.119829,places=5)
        self.assertAlmostEqual(omega[2],0.217977,places=5)
        self.assertAlmostEqual(omega[3],0.289861,places=5)
        self.assertAlmostEqual(Ecrcc23[0]['D'],-38.019453,places=5)
        self.assertAlmostEqual(Ecrcc23[1]['D'],-37.900129,places=5)
        self.assertAlmostEqual(Ecrcc23[2]['D'],-37.900129,places=5)
        self.assertAlmostEqual(Ecrcc23[3]['D'],-37.834461,places=5)

class H2O_EOMCCSd(unittest.TestCase):

    def setUp(self):
        print('RUNNING TESTS on H2O-2Re / DZ')
        self.work_dir = CCPY_DIR+'/tests/H2O-2Re-DZ'
        self.sys = build_system(self.work_dir+'/H2O-2Re-DZ.log',0)
        self.ints = get_integrals(self.work_dir+'/onebody.inp',self.work_dir+'/twobody.inp',self.sys)

    def test_eomccsd_initial_guess(self):
        from ccsd_module import ccsd
        from eomccsd_module import eomccsd
        from HBar_module import HBar_CCSD

        cc_t, Eccsd = ccsd(self.sys,self.ints,\
                        shift=0.0,tol=1.0e-07,maxit=80,\
                        diis_size=6)
        self.assertAlmostEqual(Eccsd,-75.895914,places=5)

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,self.ints,self.sys)
        
        cc_t, omega = eomccsd([5,0,0,0],H1A,H1B,H2A,H2B,H2C,cc_t,self.ints,self.sys,\
                        tol=1.0e-07,\
                        noact=100,nuact=100,\
                        maxit=300)


if __name__ == '__main__':

    suite1 = unittest.TestLoader().loadTestsFromTestCase(H2O)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(F2plus)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(CHplus)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(H2O_EOMCCSd)
    suite = unittest.TestSuite([suite1, suite2, suite3])
    #suite = unittest.TestSuite([suite3])
    
    unittest.TextTestRunner(verbosity=2).run(suite)


