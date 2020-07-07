import numpy as np
import sys

fname = sys.argv[1]
ncopies = 1

print("Opened file");
complete_data = np.load(fname)

print("Decompressing")
x = complete_data['x']
r,c = x.shape
shp = (ncopies*r,c)
print("Shape",shp)
# 259717, 3666

arr = np.memmap('totalall.mmap', dtype=np.float32, mode='w+',    
                shape=shp)

print("Saving")
for i in range(shp[0]):
    if (i%100000 == 0): print("Wrote",i)
    arr[i] = x[i % r]
    
del arr                                                        
print("Done...")            
