import numpy as np
import sys
import time
import os
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description="Scale dataset stored as compress numpy files")
parser.add_argument('--save', action='store_true',default=False)
parser.add_argument('fnames',nargs='*')
args = parser.parse_args()

fnames = args.fnames

print("There are %d files. Saving? %s." % (len(fnames),str(args.save)))

t1 = time.time()

context_num = 94
inst_length = 39
all_sum = np.zeros(inst_length)
all_sq_sum = np.zeros(inst_length)
total_len = 0

for fname in fnames:
    print(fname)
    data = np.load(fname)
    x = data['x'].astype(np.float32)
    data = x
    #y = data['y'].astype(np.float32)
    #data = np.concatenate((y, x), axis=1)

    print(data.shape)
    for i in range(context_num):
        all_sum += np.sum(data[:, inst_length*i:inst_length*(i+1)], axis=0)
        all_sq_sum += np.sum(data[:, inst_length*i:inst_length*(i+1)]*data[:, inst_length*i:inst_length*(i+1)], axis=0)
        total_len += len(data)

    #print("Mean according to Numpy is %s" % str(np.mean(data,axis=0)))
    #print("Var according to Numpy is %s" % str(np.var(data,axis=0)))

print("Took %f to decompress" % (time.time() - t1))

all_mean = all_sum/total_len
all_var = all_sq_sum/total_len - all_mean*all_mean
all_var[all_var == 0.0] = 1.0

print('Total len (number of instances) is %d' % total_len)
print('Feature dimension is %d ' % len(all_mean))

print("Global mean is %s (Norm of the vector is %f)" % (str(all_mean), np.linalg.norm(all_mean)))
print("Global var is %s (Norm of the vector is %f Sum is %f)" % (str(all_var), np.linalg.norm(all_var), np.sum(all_var)))

if args.save:
    np.savez("%s/statsall" % os.path.dirname(fname),all_mean=all_mean,all_var=all_var)
    for fname in fnames:
        print("Standardizing %s " % fname)
        data = np.load(fname)
        x = data['x'].astype(np.float32)
        #y = data['y'].astype(np.float32)
        #l = data['l']
        #data = np.concatenate((y, x), axis=1)
        data = x
        for i in range(context_num):
            #data[:, inst_length*i:inst_length*(i+1)] -= all_mean
            data[:, inst_length*i:inst_length*(i+1)] /= np.sqrt(all_var)

        print("Saving...")
        #split_data = np.hsplit(data,[2])
        #np.savez_compressed("%s/normall_%s" % os.path.split(fname), x=split_data[1], y=split_data[0], l=l)
        np.savez_compressed("%s/normall_%s" % os.path.split(fname), x=data)
