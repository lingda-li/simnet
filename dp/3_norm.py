import sys
import time
import os
import numpy as np
import argparse
from ml.cfg import data_item_format, context_length, inst_length, data_set_name, data_file_name, total_inst_num, total_size

parser = argparse.ArgumentParser(description="Compute normalization factors")
args = parser.parse_args()

t1 = time.time()

all_mean = np.zeros(inst_length)
all_std = np.zeros(inst_length)

data = np.memmap(data_file_name + '.dat', dtype=data_item_format,
             mode='r', shape=total_inst_num*inst_length)
idx = np.memmap(data_file_name + '.idx', dtype=np.uint64,
             mode='r', shape=total_size+1)
print("Open", data_file_name + '.dat')
data = data.reshape((total_inst_num, inst_length))
for i in range(inst_length):
  all_mean[i] = np.mean(data[:, i])
  all_std[i] = np.std(data[:, i])
  print(all_mean[i], all_std[i])

print("Global mean is %s" % str(all_mean))
print("Global std is %s" % str(all_std))
print("Took %f to compute" % (time.time() - t1))

np.savez("%s/stats" % data_set_name, mean=all_mean, std=all_std)
np.savetxt("%s/mean.txt" % data_set_name, all_mean)
np.savetxt("%s/std.txt" % data_set_name, all_std)
