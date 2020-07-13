import os
import sys
import numpy as np

data_set_name = sys.argv[1]
batchnum = int(sys.argv[2])
batchsize = 32 * 1024 * 2

data = np.load(data_set_name + "/totalall.npz")
x = data['x']
x_test = x[batchnum*batchsize:(batchnum+1)*batchsize,]
np.savez_compressed(data_set_name + "/test.npz", x=x_test)
