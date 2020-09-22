#!/usr/bin/env python

import sys
import argparse
import numpy as np
from ml_format import *

iter_num = 1024 * 16 * 32 * 16

parser = argparse.ArgumentParser(description="Make ML dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

fname = args.fname[0]
start = args.start
end = args.end
print("Make ML dataset for", fname, ", start from", start, ", end with", end)

nlines = 0
nfilled = 0
file_idx = 0
bad_lines = 0
bad_content = 0
all_feats = np.zeros((iter_num, context_length*inst_length), dtype='i')

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
            assert len(vals) <= context_length*inst_length
            all_feats[nfilled, 0:len(vals)] = np.array(vals)
            #print(all_feats[nfilled])
            nfilled += 1
            nlines += 1
            if end != 0:
                if nlines == end:
                    break
        except:
            print("Bad content: ", vals)
            bad_content += 1
            continue

        if nfilled == iter_num:
            print("So far see %d" % nlines)
            np.savez_compressed(fname + ".t" + str(file_idx), x=all_feats)
            file_idx += 1
            nfilled = 0
            all_feats = np.zeros((iter_num, context_length*inst_length), dtype='i')

if nfilled > 0:
    print("The last one has %d" % nfilled)
    x = np.copy(all_feats[0:nfilled])
    np.savez_compressed(fname + ".t" + str(file_idx), x=x)

print("Finished with ", nlines, "lines, ", bad_lines, "bad lines", bad_content, "bad contents.")
