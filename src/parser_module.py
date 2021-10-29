import os

#os.environ['OPENBLAS_NUM_THREADS'] = '10'

import numpy as np
from scipy.io import FortranFile

def parse_input_file(inpfile):
    #import os

    # Initialize inputs
    inputs = {}

    # Set default input settings
    inputs['work_dir'] = os.path.dirname(os.path.abspath(inpfile))
    inputs['nfroz'] = 0 
    inputs['maxit'] = 100
    inputs['tol'] = 10.0**-8
    inputs['diis_size'] = 6 
    inputs['ccshift'] = 0.0
    inputs['lccshift'] = 0.0
    inputs['eom_lccshift'] = 0.0
    inputs['isRHF'] = False
    inputs['calc_type'] = None
    inputs['adaptive_restart_dir'] = None
    inputs['save_data'] = False
    inputs['nroot'] = None
    inputs['eom_tol'] = 1.0e-06
    inputs['eom_maxit'] = 80
    inputs['root_select'] = None
    inputs['eom_guess_naoct'] = 0
    inputs['eom_guess_nuact'] = 0

    with open(inpfile,'r') as f:
        for line in f.readlines():

            if '#' in line: continue

            if 'work_dir' in line:
                inputs['work_dir'] = line.split('=')[1].strip()
            if 'nfroz' in line:
                inputs['nfroz'] = int(line.split('=')[1].strip())
            if 'calc_type' in line:
                inputs['calc_type'] = line.split('=')[1].strip()
            if 'maxit' in line and 'adaptive_maxit' not in line and 'eom_maxit' not in line:
                inputs['maxit'] = int(line.split('=')[1].strip())
            if 'tol' in line and 'eom_tol' not in line:
                inputs['tol'] = int(line.split('=')[1].strip())
                inputs['tol'] = 10**(-1.0*inputs['tol'])
            if 'diis_size' in line:
                inputs['diis_size'] = int(line.split('=')[1].strip())
            if 'ccshift' in line and 'lccshift' not in line:
                inputs['ccshift'] = float(line.split('=')[1].strip())
            if 'lccshift' in line and 'eom_lccshift' not in line:
                inputs['lccshift'] = float(line.split('=')[1].strip())
            if 'adaptive_maxit' in line:
                inputs['adaptive_maxit'] = int(line.split('=')[1].strip())
            if 'adaptive_outfile' in line:
                inputs['adaptive_outfile'] = inputs['work_dir']+'/'+line.split('=')[1].strip()
            if 'adaptive_growthperc' in line:
                inputs['adaptive_growthperc'] = float(line.split('=')[1].strip())
            if 'adaptive_triples_percentages' in line:
                temp = line.split('=')[1].strip()
                temp2 = temp.split(',')
                inputs['adaptive_triples_percentages'] = [float(x) for x in temp2]
            if 'adaptive_restart_dir' in line:
                inputs['adaptive_restart_dir'] = inputs['work_dir']+'/'+line.split('=')[1].strip()
            if 'save_data' in line:
                temp = line.split('=')[1].strip()
                if temp == 'True':
                    inputs['save_data'] = True
                else:
                    inputs['save_data'] = False
            if 'nroot' in line:
                inputs['nroot'] = int(line.split('=')[1].strip())
            if 'eom_tol' in line:
                inputs['eom_tol'] = int(line.split('=')[1].strip())
                inputs['eom_tol'] = 10**(-1.0*inputs['eom_tol'])
            if 'eom_maxit' in line:
                inputs['eom_maxit'] = int(line.split('=')[1].strip())
            if 'eom_lccshift' in line:
                inputs['eom_lccshift'] = float(line.split('=')[1].strip())
            if 'root_select' in line:
                temp = line.split('=')[1].strip()
                temp2 = temp.split(',')
                inputs['root_select'] = [int(x) for x in temp2]
            if 'eom_guess_noact' in line and 'eom_guess_nuact' not in line:
                inputs['eom_guess_noact'] = int(line.split('=')[1].strip())
            if 'eom_guess_nuact' in line and 'eom_guess_noact' not in line:
                inputs['eom_guess_nuact'] = int(line.split('=')[1].strip())


    with open(inpfile,'r') as f:
        for line in f.readlines():

            if '#' in line: continue

            if 'onebody' in line:
                inputs['onebody_file'] = inputs['work_dir']+'/'+line.split('=')[1].strip()
            if 'twobody' in line:
                inputs['twobody_file'] = inputs['work_dir']+'/'+line.split('=')[1].strip()
            if 'gamess_file' in line:
                inputs['gamess_file'] = inputs['work_dir']+'/'+line.split('=')[1].strip()


    return inputs

def parse_moefile(moefile):
	with FortranFile(moefile,'r') as f90_file:
		first_line_reals = f90_file.read_reals(dtype=np.float)
		tvec = f90_file.read_reals(dtype=np.float)
	#	with FortranFile(moefile,'r') as f90_file:
	#		first_line_ints = f90_file.read_ints(dtype=np.int64)
		corr_energy = first_line_reals[1]
	#	corr_energy_terms = first_line_reals[2:10]
	#	tvec_size = first_line_ints[10] 				
	#	assert len(tvec) == tvec_size, \
	#		   "Length of the T vector doesn't match!"
	f90_file.close()

#	return tvec, corr_energy, corr_energy_terms
	return tvec, corr_energy

def extract_t_vector(tvec,sys):

    noa = sys['Nocc_a']
    nua = sys['Nunocc_a']
    nob = sys['Nocc_b']
    nub = sys['Nunocc_b']

    t1a = np.zeros((nua,noa))
    t1b = np.zeros((nub,nob))
    t2a = np.zeros((nua,nua,noa,noa))
    t2b = np.zeros((nua,nub,noa,nob))
    t2c = np.zeros((nub,nub,nob,nob))

    rec = 0
    for i in range(noa):
        for a in range(nua):
            t1a[a,i] = tvec[rec]
            rec += 1

    for i in range(nob):
        for a in range(nub):
            t1b[a,i] = tvec[rec]
            rec += 1

    for i in range(noa):
        for j in range(noa):
            for a in range(nua):
                for b in range(nua):
                    t2a[a,b,i,j] = tvec[rec]
                    rec += 1

    for i in range(noa):
        for j in range(nob):
            for a in range(nua):
                for b in range(nub):
                    t2b[a,b,i,j] = tvec[rec]
                    rec += 1

    for i in range(nob):
        for j in range(nob):
            for a in range(nub):
                for b in range(nub):
                    t2c[a,b,i,j] = tvec[rec]
                    rec += 1

    cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c}

    return cc_t

def get_p_space_triples_array(p_file,sys):

    print('Parsing p space triples file from {}'.format(p_file))
    
    nua = sys['Nunocc_a']
    noa = sys['Nocc_a']
    nub = sys['Nunocc_b']
    nob = sys['Nocc_b']

    p_space = np.zeros((nua,nua,nua,noa,noa,noa))
    with open(p_file,'r') as P:
        for line in P.readlines():
            p = [int(x) for x in line.split()]
            p_space[p[0]-nob-1,p[1]-nob-1,p[2]-nob-1,p[3]-1,p[4]-1,p[5]-1] = 1
    return p_space

def get_p_space_quadruples_array(p_file,sys):

    print('Parsing p space quadruples file from {}'.format(p_file))
    
    nua = sys['Nunocc_a']
    noa = sys['Nocc_a']
    nub = sys['Nunocc_b']
    nob = sys['Nocc_b']

    p_space = np.zeros((nua,nua,nua,nua,noa,noa,noa,noa))
    with open(p_file,'r') as P:
        for line in P.readlines():
            p = [int(x) for x in line.split()]
            p_space[p[0]-nob-1,p[1]-nob-1,p[2]-nob-1,p[3]-nob-1,p[4]-1,p[5]-1,p[6]-1,p[7]-1] = 1
    return p_space

def calc_cc_energy(cc_t,ints):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    Ecorr = 0.0
    Ecorr += np.einsum('me,em->',fA['ov'],t1a,optimize=True)
    Ecorr += np.einsum('me,em->',fB['ov'],t1b,optimize=True)
    Ecorr += 0.25*np.einsum('mnef,efmn->',vA['oovv'],t2a,optimize=True)
    Ecorr += np.einsum('mnef,efmn->',vB['oovv'],t2b,optimize=True)
    Ecorr += 0.25*np.einsum('mnef,efmn->',vC['oovv'],t2c,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vA['oovv'],t1a,t1a,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vC['oovv'],t1b,t1b,optimize=True)
    Ecorr += np.einsum('mnef,em,fn->',vB['oovv'],t1a,t1b,optimize=True)

    return Ecorr


def parse_extcorr_file(moefile,ints,sys):
	
    print('Parsing extcorr file: {}'.format(moefile))
    tvec, Ecorr_file = parse_moefile(moefile)
    cc_t = extract_t_vector(tvec,sys)
    Ecorr_parse = calc_cc_energy(cc_t,ints)
    print('Correlation energy from file = {}'.format(Ecorr_file))
    print('Correlation energy from parsed file = {}'.format(Ecorr_parse))

    return cc_t

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parser to test reading input files for CCpy')
    parser.add_argument('inpfile',type=str,help='path to input file')
    args = parser.parse_args()

    inputs = parse_input_file(args.inpfile)

    for key,value in inputs.items():
        print(key,'-->',value)

