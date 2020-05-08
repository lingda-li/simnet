import os
import numpy as np

data_set_name = "data_spec"
batchnum = 16 * 16 * 4
batchsize = 32 * 1024

data = np.load(data_set_name + "/totalall.npz")
x = data['x']
#y = data['y']

x_test = x[batchnum*batchsize:(batchnum+1)*batchsize,]
#y_test = y[batchnum*batchsize:(batchnum+1)*batchsize,]
#np.savez_compressed(data_set_name + "/test.npz", x=x_test, y=y_test)
np.savez_compressed(data_set_name + "/nntest.npz", x=x_test)
