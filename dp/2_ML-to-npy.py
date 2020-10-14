import sys
import os
import argparse
import numpy as np
from ml_format import *

iter_num = 1024 * 16 * 32 * 16

parser = argparse.ArgumentParser(description="Make ML dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

start = args.start
end = args.end
output = args.fname[0]
if len(args.fname) > 1:
    output = os.path.join(os.path.dirname(args.fname[0]), "all")
print("Make ML dataset ", output, ", start from", start, ", end with", end)

nlines = 0
nfilled = 0
file_idx = 0
bad_lines = 0
bad_content = 0
all_feats = np.zeros((iter_num, context_length*inst_length), dtype='i')

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read ", fname)
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
                if end != 0 and nlines == end:
                    break
            except:
                print("Bad content: ", len(vals), vals)
                bad_content += 1
                continue

            if nfilled == iter_num:
                print("So far see %d" % nlines)
                np.savez_compressed(output + ".t" + str(file_idx), x=all_feats)
                file_idx += 1
                nfilled = 0
                all_feats = np.zeros((iter_num, context_length*inst_length), dtype='i')

if nfilled > 0:
    print("The last one has %d" % nfilled)
    x = np.copy(all_feats[0:nfilled])
    np.savez_compressed(output + ".t" + str(file_idx), x=x)

print("Finished with ", nlines, "lines, ", bad_lines, "bad lines", bad_content, "bad contents.")
