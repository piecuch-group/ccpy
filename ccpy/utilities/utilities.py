# gives a single float value
#psutil.cpu_percent()
# gives an object with many fields
#psutil.virtual_memory()
# you can convert that object to a dictionary 
#dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
#psutil.virtual_memory().percent
#79.2
# you can calculate percentage of available memory
#psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
#20.8


def dumpSystemToPGFiles(sys, integrals, projectName, nfrozen):
    """Dumps the molecule information into the cc.inp, *.inf, and *.gjf files used
        by the Piecuch Group codes."""
    from scipy.io import FortranFile
    # Dictionary of default values used for typical CC and EOMCC calculations
    defaultValues = {'CC Convergence' : 8,
                     'Active CC Occupied' : 0,
                     'Active CC Unoccupied' : 0,
                     'Active EOM Guess Occupied' : 0,
                     'Active EOM Guess Unoccupied' : 0,
                     'EOM Multiplicity' : 1,
                     'EOM Maximum Iterations' : 200,
                     'EOM Convergence' : 8,
                     'EOM Number of Roots' : 10,
                     'Memory' : 2000}
    # Write the first line of the cc.inp file (this is all that Jun's code reads)
    ccinp = [sys.nelectrons,
             2*sys.norbitals-sys.nelectrons,
             2*nfrozen,
             defaultValues['Memory'],
             defaultValues['CC Convergence'],
             0,
             sys.multiplicity]
    with open("cc.inp",'w') as ccinpfile:
        ccinpfile.write('  '.join(map(str,ccinp)))
    # Write the values in the *.gjf file
    entranceString = {'Active CC Occupied' : 'm1',
                      'Active CC Unoccupied' : 'm2',
                      'Active EOM Guess Occupied' : 'm3',
                      'Active EOM Guess Unoccupied' : 'm4',
                      'EOM Multiplicity' : 'mult',
                      'EOM Maximum Iterations' : 'itEOM',
                      'EOM Convergence' : 'conver',
                      'EOM Number of Roots' : 'nroot'}
    with open(projectName+'.gjf','w') as gjffile:
        gjffile.write('%mem=800mw\n')
        gjffile.write('UCC[uhf')
        for key,value in entranceString.items():
            gjffile.write('  '+str(value)+'='+str(defaultValues[key]))
        gjffile.write(']')


def print_memory_usage():
    '''Displays the percentage of used RAM and available memory. Useful for 
    investigating the memory usages of various routines.'''
    import psutil
    import os
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    print(int(memory / (1024*1024) ), "MB")
    return

def clean_up(fid,n):
    for i in range(n):
        remove_files(fid+'-'+str(i+1)+'.npy')
    return


def remove_file(filePath):
    import os
    try:
        os.remove(filePath)
    except OSError:
        pass
    return
