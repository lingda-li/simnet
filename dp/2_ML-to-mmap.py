import sys
import os
import argparse
import numpy as np
from ml_format import *

parser = argparse.ArgumentParser(description="Make ML memmap dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
# 172161026 - 1 + 171035513 - 2
parser.add_argument('--total-rows', type=int, default=343196536)
parser.add_argument('--cols', type=int, default=context_length*inst_length)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

start = args.start
end = args.end
output = args.fname[0]
if len(args.fname) > 1:
    output = os.path.join(os.path.dirname(args.fname[0]), "all")
output += ".mmap"
r = args.total_rows
w = context_length*inst_length
shp = (r, w)

print("Make ML dataset ", output, ", start from", start, ", end with", end, ", shape is ", shp)

nlines = 0
nfilled = 0
file_idx = 0
bad_lines = 0
bad_content = 0
all_feats = np.memmap(output, dtype=np.float32, mode='w+', shape=shp)
all_sum = np.zeros(w)
all_sq_sum = np.zeros(w)

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read", fname, flush=True)
    with open(fname) as f:
        for line in f:
            if nlines < start:
                nlines += 1
                continue

            try:
                vals = [int(s) for s in line.rstrip().split(' ')]
            except:
                bad_lines += 1
                continue
            #print(vals)

            try:
                assert len(vals) % inst_length == 0
                assert len(vals) <= w
                all_feats[nfilled, 0:len(vals)] = np.array(vals)
                all_feats[nfilled, len(vals):w] = 0
                all_sum += all_feats[nfilled]
                all_sq_sum += all_feats[nfilled]*all_feats[nfilled]
                if nfilled == 0:
                    print(all_feats[nfilled].shape)
                    print(all_feats[nfilled], flush=True)
                nfilled += 1
                nlines += 1
                if end != 0 and nlines == end:
                    break
            except:
                print("Bad content: ", len(vals), vals)
                bad_content += 1
                continue

            if nfilled % 5000000 == 0:
                print("Have filed %d" % nfilled, flush=True)

del all_feats
np.savez(os.path.join(os.path.dirname(args.fname[0]), "stat_mmap"), all_sum=all_sum, all_sq_sum=all_sq_sum)
print("Finished with ", nfilled, "entries, ", nlines, "lines, ", bad_lines, "bad lines", bad_content, "bad contents.")
