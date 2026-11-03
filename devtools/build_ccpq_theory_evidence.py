"""Generate the theory note's tables and source/report SHA-256 manifest."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/theory'
FILES=['h2o_631g_optimized_df.json','h2o_631g_excited_optimized.json','h4_631g_excited_optimized.json','h4_631g_excited_step_convergence.json','h6_healthy_v6.json','h6_healthy_v7.json','h2o_631g_frozen_core.json','h2o_631g_reduced_all_components.json']
def read(n):return json.loads((ROOT/'examples/results'/n).read_text())
def num(x):
 m,e=f'{x:.3e}'.split('e');return rf'$\ee{{{m}}}{{{int(e)}}}$'
def table(caption,head,rows,label):
 return '\n'.join([r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{'+('l'*len(head))+'}',r'\toprule',' & '.join(head)+r'\\',r'\midrule',*[' & '.join(r)+r'\\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}'+r'\label{'+label+'}',r'\end{table}',''])
def maxerr(cases,key):return max(c['gradients'][key]['max_error'] for c in cases)
def errors(c):return [num(c['gradients'][k]['max_error']) for k in ('base','correction','total')]
def build():
 OUT.mkdir(parents=True,exist_ok=True);sec=[]
 g=read(FILES[0]);assert g['nfrozen']==1 and len(g['components'])==9 and len(g['cases'])==4
 sec += [r'\subsection{Independent optimized ground-state tensor comparisons}',
 'The primary DF water calculation freezes O(1s), retaining all 13 spatial orbitals.',
 'CR-CC(2,3) has P = 64 S + 1,360 D and Q = all 11,200 T. The active-SDt',
 'case has P = 64 S + 1,360 D + 2,436 T and Q = all 8,764 remaining T.',
 'All nine components use four-point differences at 0.001 bohr of independently',
 'converged optimized tensor energies, with the same DF auxiliary basis and reference.',
 'The full CC(P;3) comparator includes selected triples in the left state and moments.','']
 rows=[['CR-CC(2,3)' if c['theory']=='crcc23' else 'Full CC(P;3)',c['variant'],*errors(c)] for c in g['cases']]
 sec += [table('DF H2O/6-31G: independent optimized-energy finite differences; maximum errors over nine components.',['Method','Variant','Base','Correction','Total'],rows,'tab:groundnumerical')]
 ga=next(c for c in g['cases'] if c['theory']=='crcc23' and c['variant']=='A')
 sec += ['The determinant and existing DF CR-CC(2,3)-A analytical gradients differ by at most '+num(ga['optimized_analytic_error'])+r' $\gunit$.',
 'The largest base/correction energy differences over this table are '+num(max(c['base_energy_error'] for c in g['cases']))+' / '+num(max(c['correction_energy_error'] for c in g['cases']))+r' $\eunit$.','']
 w=read(FILES[1]);assert w['nfrozen']==1 and w['ndet']==245025 and len(w['components'])==9
 sec += [r'\subsection{Independent optimized excited-state tensor comparisons}',
 'The water EOM test uses the same 2,436 active triples in ground T and excited R/L,',
 'all S/D, and all 8,764 remaining triples in Q. Both multiplicities use full spin blocks.',
 'The lowest triplet and singlet map to roots 1 and 2 of the unfiltered fixed-$M_s=0$',
 'spectrum. All nine components use a 0.001-bohr four-point stencil.','']
 rows=[['Triplet' if c['multiplicity']==3 else 'Singlet',c['variant'],*errors(c)] for c in w['cases']]
 sec += [table('Full EOMCCSDt/EOMCC(P;3), frozen-O(1s) H2O/6-31G. The base column separately checks EOMCCSDt derivatives.',['State','Variant','Base','Correction','Total'],rows,'tab:waterexcited')]
 se=max(c['independent_spectrum_error'] for c in w['cases']);ov=max(abs(c[k]-1) for c in w['cases'] for k in ('native_right_overlap','native_left_overlap'))
 sec += ['Independent Arnoldi on optimized HR and determinant Davidson spectra agree within '+num(se)+r' $\eunit$.',
 'Physical right/left determinant-coordinate overlaps differ from unity by at most '+num(ov)+'.',
 'The maximum base/correction energy differences are '+num(max(c['base_energy_error'] for c in w['cases']))+' / '+num(max(c['correction_energy_error'] for c in w['cases']))+r' $\eunit$.',
 'Displaced root overlaps exceed 0.999997; energy-gap and left/right residual guards are checked at every point.','']
 h=read(FILES[2]);assert h['nfrozen']==0 and h['ndet']==784 and len(h['components'])==12
 sec += [r'\subsection{A nonzero-reference-component excited-state control}',
 'The nonplanar H4/6-31G test retains all four electrons and eight orbitals.',
 'P has 24 S + 174 D + 160 T; Q has all 200 remaining T. Its lowest triplet',
 'and lowest singlet are roots 1 and 3 of the unfiltered $M_s=0$ spectrum.',
 r'The singlet has $|r_0|\simeq0.083$. Both corrections are nonzero.',
 'All twelve Cartesian components use a 0.00025-bohr four-point step.','']
 rows=[['Triplet' if c['multiplicity']==3 else 'Singlet',c['variant'],*errors(c)] for c in h['cases']]
 sec += [table('Independent full excited-state H4/6-31G comparison, including a nonzero right reference coefficient.',['State','Variant','Base','Correction','Total'],rows,'tab:h4excited')]
 st=read(FILES[3]);rows=[]
 for r in st['records']:
  if r['multiplicity']==1 and r['variant']=='D':rows.append([f"{r['step_bohr']:.5f}",*[num(r['errors'][k]) for k in ('base','correction','total')]])
 sec += ['The singlet has a nearby root separated by about 0.00228 hartree. The initial',
 '0.001-bohr step has visible truncation error: halving it reduces the correction',
 'error by approximately sixteen. Smaller-step agreement is obtained without',
 'altering the analytical equations or loosening the comparison tolerance.','',
 table('H4 singlet D, atom 0/z: explicit four-point step convergence; errors in '+r'$\gunit$'+'.',['Step (bohr)','Base','Correction','Total'],rows,'tab:step')]
 sec += [r'\subsection{Substantial triples/quadruples selections}',
 'The all-electron H6/6-31G models retain six electrons and twelve spatial orbitals',
 '(48,400 determinants). Every P includes all 54 S and 945 D plus the selected',
 'T/Q below. Q is the complete remaining triples/quadruples complement.','']
 rows=[];erows=[]
 for av,fn in [(6,FILES[4]),(7,FILES[5])]:
  d=read(fn);assert d['nfrozen']==0 and len(d['components'])==2 and d['ndet']==48400
  model=d['models']['sdtq'];pc=model['P_rank_counts'];qc=model['Q_rank_counts']
  rows.append([str(av),str(pc['3']),str(pc['4']),str(qc['3']),str(qc['4'])])
  for state in (0,1,2):
   cases=[c for c in d['cases'] if c['state']==state]
   erows.append([str(av),'Ground' if state==0 else f'Root {state}',*[num(maxerr(cases,k)) for k in ('base','correction','total')]])
 sec += [table('H6 complete active selections: all three occupied MOs and either six or seven virtual MOs define P-space triples and quadruples.',['Active virtuals','P triples','P quadruples','Q triples','Q quadruples'],rows,'tab:h6spaces'),
 'The virtual-6 model retains 27.67\\% of triples and 16.94\\% of quadruples;',
 'the virtual-7 model retains 45.27\\% and 33.57\\%, respectively. Only two',
 'components (atom 1/z and atom 2/x) are directly finite-differenced, not all eighteen.',
 'Here the energies being finite-differenced come from the determinant model itself:',
 'this is a general-space derivative-consistency test, not an optimized quadruples-kernel comparison.','',
 table('H6 substantial-T/Q derivative consistency. Each row takes the worst error across MP/full over the two checked components.',['Active virtuals','State','Base','Correction','Total'],erows,'tab:h6errors')]
 wt=read(FILES[6]);assert wt['nfrozen']==1 and len(wt['components'])==2
 sec += ['The conventional frozen-O(1s) water T/Q model has 64 S + 1,360 D +',
 '2,480 T + 5,210 quadruples in P, and 8,720 remaining T + 37,490 remaining',
 'quadruples in Q (245,025 support determinants). Only atom 1/z and atom 2/y',
 'are directly finite-differenced; all nine analytical components are recorded.',
 'No all-nine-component claim is made for this one-frozen-orbital T/Q model.','']
 rows=[]
 for state in (0,1,2):
  cases=[c for c in wt['cases'] if c['state']==state]
  rows.append(['Ground' if state==0 else f'Root {state}',*[num(maxerr(cases,k)) for k in ('base','correction','total')]])
 sec += [table('Conventional water general T/Q model: worst errors over MP/full and the two checked components.',['State','Base','Correction','Total'],rows,'tab:watertq')]
 small=read(FILES[7]);assert small['nfrozen']==3 and len(small['components'])==9
 sec += ['A separate reduced-correlation water regression freezes three occupied',
 'orbitals (six electrons), correlating four electrons in the same 13-orbital basis.',
 'Its 2,025-determinant model checks all nine components for ground and two excited',
 'states with both denominators. The largest total/correction errors are '+num(maxerr(small['cases'],'total'))+' / '+num(maxerr(small['cases'],'correction'))+r' $\gunit$.',
 'This is not the standard one-frozen-O(1s) model.','']
 for d in [g,w,h,wt,small,read(FILES[4]),read(FILES[5])]:assert all(max(v['max_error'] for v in c['gradients'].values())<2e-8 for c in d['cases'])
 (OUT/'ccpq_validation_tables.tex').write_text('\n'.join(sec)+'\n')
 sources=[ROOT/'examples/results'/fn for fn in FILES]
 sources += [OUT/'ccpq_formula_checks.json',OUT/'ccpq_analytical_gradients.tex',OUT/'ccpq_validation_tables.tex']
 sources += [OUT/'ccpq_lean_response.tex',OUT/'ccpq_lean_response_checks.json',OUT/'ccpq_lean_implementation.md',ROOT/'devtools/check_ccpq_lean_response.py']
 sources += list((ROOT/'ccpy/gradients/ccpq').glob('*.py'))
 sources += [ROOT/'ccpy/gradients/ccpq/fortran/fcipy_connected.f90',ROOT/'ccpy/gradients/orbital_response.py',ROOT/'ccpy/gradients/derivative_integrals.py',ROOT/'ccpy/methods/corrections.py',ROOT/'ccpy/moments/ccp3.py',ROOT/'ccpy/gradients/ccpq_gradients_revised.pdf',ROOT/'devtools/check_ccpq_theory.py',Path(__file__).resolve()]
 unique=sorted(set(sources));manifest=dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),description='Local implementation/report snapshot. Content hashes, rather than a clean Git revision, identify the modified workspace sources.',files=[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in unique])
 formula=json.loads((OUT/'ccpq_formula_checks.json').read_text());assert formula['passed'] and formula['configurations']==12
 manifest['formula_checks']=dict(configurations=12,max_error=formula['max_error'])
 lean=json.loads((OUT/'ccpq_lean_response_checks.json').read_text());assert lean['passed'] and lean['configurations']==8
 manifest['lean_response_checks']=dict(configurations=8,max_error=lean['max_error'],scope=lean['scope'])
 log=Path('/tmp/test_excited_full_regression.log')
 if log.exists():
  txt=log.read_text();assert '144 passed' in txt
  (OUT/'ccpq_regression_run.txt').write_text(txt.replace(str(ROOT)+'/', ''))
  manifest['gradient_regression']='144 passed; saved in ccpq_regression_run.txt'
 (OUT/'ccpq_validation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 index=[r'\begingroup\small\setlength{\parskip}{0pt}',r'\subsection*{Saved evidence index}',r'\path{docs/theory/ccpq_validation_manifest.json} records SHA-256 hashes.', 'Repository-relative paths and hashes record provenance, not correctness.','',r'Reports in \path{examples/results/}:']
 index += [r'\par\noindent\path{'+FILES[i]+r'}\quad\path{'+FILES[i+1]+'}' for i in range(0,len(FILES),2)]
 index += [r'\par Checks in \path{docs/theory/}:',r'\par\noindent\path{ccpq_formula_checks.json}',r'\par\noindent\path{ccpq_lean_response_checks.json}',r'\par\noindent\path{ccpq_regression_run.txt}',r'\par Tables: \path{devtools/build_ccpq_theory_evidence.py}.',r'\par Master: \path{docs/theory/ccpq_analytical_gradients.tex}.',r'\endgroup','']
 (OUT/'ccpq_validation_manifest.tex').write_text('\n'.join(index))
 print('Generated tables and manifest for',len(unique),'source/report files')
if __name__=='__main__':build()
