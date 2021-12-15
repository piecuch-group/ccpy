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
    import os
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    print(int(memory / (1024*1024) ), "MB")
    return

def clean_up(fid,n):
    for i in range(n):
        remove_files(fid+'-'+str(i+1)+'.npy')
    return

def remove_files(fid):
    import os
    os.remove(fid)
    return
