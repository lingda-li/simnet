import pandas as pd
import numpy as np
import sys
import copy
import math
ipfile= sys.argv[1]
if(sys.argv[2]==0):
    dt= np.float32
else:
    dt=np.uint
fileName= ipfile+ ".bin"
df= pd.read_csv(ipfile, sep=' ',dtype=dt, header=None,usecols=[0,1,2,3,4,5,6,7,8,9] )
#import ipdb;ipdb.set_trace()
fil= df.to_numpy()
fil.tofile(fileName)
#with open(fileName, mode='rb') as file: # b is important -> binary
#    fileContent = file.read()
data= np.fromfile(fileName,dtype=dt)
print(np.sum(fil.flatten()-data))
import ipdb;ipdb.set_trace()
