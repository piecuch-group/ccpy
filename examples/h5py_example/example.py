import os
import psutil
import h5py
import numpy as np
import time


def current_memory():
    """Returns current memory usage in MB"""

    memory = psutil.Process(os.getpid()).memory_info().rss
    return memory / (1024 ** 2) 



def prange(start, end, step):

    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)


#@njit
def write_data(f, n):

    MAX_MEMORY = 4000 # MB
    BLKMIN = 4

    mem_now = current_memory()
    max_memory = max(0, MAX_MEMORY - mem_now)
    blksize = min(nvir, max(BLKMIN, int((max_memory * 0.9e6/8.0 - nocc**4)/unit)))

    for p0, p1 in prange(0, nvir, blksize):

    for i in range(n):
        f['mydataset'][i, :, :, :] = np.random.rand(100, 100, 100)

if __name__ == "__main__":

    f = h5py.File("mytestfile.hdf5", "w")
    f.create_dataset("mydataset", (100, 100, 100, 100), dtype=np.float64)

    write_data(f, 100)
    #t1 = time.time()
    #for i in range(100):
    #    f['mydataset'][i, :, :, :] = np.random.rand(100, 100, 100)
    #print('Time to write data =', time.time() - t1)
    f.close()


    








