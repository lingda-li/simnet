import numpy as np
import sys
import time
import os
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description="Scale dataset stored as compress numpy files")
#parser.add_argument('--total-rows', type=int, default=351348424)
parser.add_argument('--total-rows', type=int, default=342959816)
parser.add_argument('--cols', type=int, default=94*39)
parser.add_argument('fnames',nargs='*')
args = parser.parse_args()

r = args.total_rows
c = args.cols
shp = (r, c)
print(shp)
arr = np.memmap('data_spec_fc/totalall.mmap', dtype=np.float32, mode='w+',
                shape=shp)

fnames = args.fnames

print("There are %d files." % (len(fnames)))

t1 = time.time()

total_x = None
total_y = None
total_l = None

j = 0
for fname in fnames:
    print(fname)
    data = np.load(fname)
    x = data['x'].astype(np.float32)
    for i in range(len(x)):
        arr[j] = x[i]
        j = j+1

print("Total number: ", j)

del arr
print("Took %f to make files" % (time.time() - t1))
