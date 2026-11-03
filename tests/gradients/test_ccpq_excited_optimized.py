"""Excited full CC(P;3) derivatives versus independent optimized tensor energies."""
import os
import numpy as np
import pytest
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient
from ccpy.gradients.finite_difference import finite_difference
from examples.ccpq_excited_optimized_comparison import H4, optimized, optimized_spectrum, overlap

pytestmark=pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'),reason='Optional FCIpy connector not configured')
COMPONENTS=[(1,2),(2,0)]


@pytest.fixture(scope='module')
def benchmark():
    mol=Molecule(H4,'6-31g',auxiliary_basis='def2-universal-jkfit',symmetry=False,verbose=False)
    ref=optimized(mol,0)
    spectrum,residual=optimized_spectrum(ref)
    assert residual<1e-10
    states=[int(np.argmin(abs(spectrum-r.omega)))+1 for r in ref.roots]
    # The singlet is not the second root in the full Ms=0 spectrum.
    assert states==[1,3]
    assert abs(ref.roots[1].r0)>.01
    assert all(abs(r.delta[k])>1e-6 for r in ref.roots for k in ('A','D'))
    def energies(coords):
        b=optimized(mol.with_geometry(coords,unit='bohr'),0,
            triples=ref.triples,layout=ref.layout,reference=ref)
        assert all(overlap(r.right,r0.right)>.99 for r,r0 in zip(b.roots,ref.roots))
        return np.array([[r.base,r.delta['A'],r.delta['D']] for r in b.roots])
    xyz=np.array([a[1] for a in mol.atoms])
    # The first singlet is close to another root. A step-convergence check in
    # the demonstration resolves the fourth-order error at 0.001 bohr.
    numerical=finite_difference(energies,xyz,step=2.5e-4,npoints=4,components=COMPONENTS)
    calc=CCPQGradient(mol,ref.layout.P,(3,),'mp',state=states[0],backend='matrix_free',
        sparse_cache=True,integral_model='factorized')
    return mol,ref,spectrum,states,numerical,calc


@pytest.mark.parametrize('root_index,variant,denominator,column',[
    (0,'A','mp',1),(0,'D','full',2),(1,'A','mp',1),(1,'D','full',2)])
def test_excited_full_ccp3_against_optimized_fd(benchmark,root_index,variant,denominator,column):
    mol,ref,spectrum,states,numerical,calc=benchmark
    root=ref.roots[root_index]
    calc.state=calc.backend.state=states[root_index]
    calc.backend.denominator=denominator
    gradients=calc.gradient_components()
    amp=calc.stages['amplitudes']
    assert mol.nuclear_repulsion+amp.energy_base==pytest.approx(root.base,abs=1e-10)
    assert amp.delta==pytest.approx(root.delta[variant],abs=1e-11)
    assert overlap(root.right,amp.data.r)>1-1e-9
    assert overlap(root.left,amp.data.l)>1-1e-9
    np.testing.assert_allclose(amp.data.eom_values[:4],spectrum+amp.data.E0,atol=1e-9,rtol=0.)
    for key,num in [('base',numerical[:,root_index,0]),('correction',numerical[:,root_index,column]),
                    ('total',numerical[:,root_index,0]+numerical[:,root_index,column])]:
        analytical=np.array([gradients[key][index] for index in COMPONENTS])
        np.testing.assert_allclose(analytical,num,atol=1e-8,rtol=0.)
