import os
import sys
import numpy as np

data_set_name = sys.argv[1]
batchnum = 5200
total_size = 342959816
batchsize = 32 * 1024 * 2

x = np.memmap(data_set_name + "/totalall.mmap", dtype=np.float32,
              mode='r',shape=(total_size, 94*39))
x_test = x[batchnum*batchsize:(batchnum+1)*batchsize,]
np.savez_compressed(data_set_name + "/test.npz", x=x_test)
