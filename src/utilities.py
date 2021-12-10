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

def print_memory_usage():
    '''Displays the percentage of used RAM and available memory. Useful for 
    investigating the memory usages of various routines.'''
    import psutil
    print('Percentage of used RAM: {}'.format(psutil.virtual_memory().percent))
    print('Percentage of available memory: {}'.format(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total))
    return

def clean_up(fid,n):
    for i in range(n):
        remove_files(fid+'-'+str(i+1)+'.npy')
    return

def remove_files(fid):
    import os
    os.remove(fid)
    return
