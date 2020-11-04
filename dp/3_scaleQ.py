import sys
import time
import os
import numpy as np
from multiprocessing import Pool
import argparse
from q_format import *

parser = argparse.ArgumentParser(description="Scale dataset stored as q files")
parser.add_argument('--save', action='store_true',default=False)
parser.add_argument('--dirName',type=str,default=".")
parser.add_argument('fnames', nargs='*')
args = parser.parse_args()

fnames = args.fnames

print("There are %d files. Saving? %s." % (len(fnames),str(args.save)))

t1 = time.time()

all_sum = np.zeros(inst_length)
all_sq_sum = np.zeros(inst_length)
total_len = 0
nlines = 0
target_lines = 1000000
bad_lines = 0
bad_contents = 0

for fname in fnames:
    print("read", fname, flush=True)
    with open(fname) as f:
        for line in f:
            try:
                vals = [int(s) for s in line.rstrip().split(' ')]
            except:
                bad_lines += 1
                continue

            try:
                assert len(vals) % inst_length == 0
                assert len(vals) <= inst_length*context_length
                data = np.array(vals)
                inst_num = int(len(vals) / inst_length)
            except:
                print("Bad content: ", len(vals), vals)
                bad_contents += 1
                continue

            for j in range(inst_num):
                all_sum += data[inst_length*j:inst_length*(j+1)]
                all_sq_sum += data[inst_length*j:inst_length*(j+1)]*data[inst_length*j:inst_length*(j+1)]
            total_len += context_length
            nlines += 1
            if nlines == target_lines:
                break


all_mean = all_sum/total_len
all_var = all_sq_sum/total_len - all_mean*all_mean
all_var[all_var == 0.0] = 1.0

print("Finished with", nlines, "entries,", bad_lines, "bad lines,", bad_contents, "bad contents.")
print('Total len (number of instances) is %d' % total_len)
print('Feature dimension is %d ' % len(all_mean))
print("Took %f to process" % (time.time() - t1))

print("Global mean is %s (Norm of the vector is %f)" % (str(all_mean), np.linalg.norm(all_mean)))
print("Global var is %s (Norm of the vector is %f Sum is %f)" % (str(all_var), np.linalg.norm(all_var), np.sum(all_var)))

if args.save:
    np.savez("%s/statsall" % args.dirName,all_mean=all_mean,all_var=all_var)
    np.savetxt("%s/mean.txt" % args.dirName,all_mean)
    np.savetxt("%s/var.txt" % args.dirName,all_var)
