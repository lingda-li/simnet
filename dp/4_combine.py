import numpy as np
import sys
import time
import os
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description="Scale dataset stored as compress numpy files")
parser.add_argument('fnames',nargs='*')
args = parser.parse_args()

fnames = args.fnames

print("There are %d files." % (len(fnames)))

t1 = time.time()

total_x = None
total_y = None
total_l = None

for fname in fnames:
    print(fname)
    data = np.load(fname)
    x = data['x'].astype(np.float32)
    if total_x is None:
        total_x = x
    else:
        total_x = np.vstack((total_x,x))

np.savez_compressed(os.path.join(os.path.dirname(sys.argv[1]), "totalall"), x=total_x)

print("Took %f to make files" % (time.time() - t1))
