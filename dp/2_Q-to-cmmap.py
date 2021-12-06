import sys
import os
import argparse
import numpy as np
from ml.cfg import inst_length, context_length, data_item_format

parser = argparse.ArgumentParser(description="Make memmap dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('--total-entries', type=int, default=0)
parser.add_argument('--total-insts', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

start = args.start
end = args.end
output = args.fname[0]
if len(args.fname) > 1:
    output = os.path.join(os.path.dirname(args.fname[0]), "train")
idx_output = output + ".idx"
output += ".dat"
r = args.total_entries + 1
w = args.total_insts * inst_length

print("Make dataset", output, ", start from", start, ", end with", end, ", shape is ", r, w, flush=True)

nlines = 0
nfilled = 0
cur_idx = 0
bad_lines = 0
bad_content = 0
all_idx = np.memmap(idx_output, dtype=np.uint64, mode='w+', shape=r)
all_feats = np.memmap(output, dtype=data_item_format, mode='w+', shape=w)
all_idx[0] = 0

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read", fname, flush=True)
    with open(fname) as f:
        for line in f:
            if cur_idx >= r:
                print("Find more lines than the input shape.", flush=True)
                break
            if nlines < start:
                nlines += 1
                continue

            try:
                vals = [int(s) for s in line.rstrip().split(' ')]
            except:
                bad_lines += 1
                print("Bad line", flush=True)
                continue
            #print(vals)

            length = len(vals)
            try:
                assert length % inst_length == 0
                assert length <= context_length*inst_length
            except:
                print("Bad content:", length, vals, flush=True)
                bad_content += 1
                continue

            all_feats[nfilled:nfilled+length] = np.array(vals)
            if nfilled == 0:
                print("First sample:", length)
                print(all_feats[nfilled:nfilled+length], flush=True)
            nfilled += length
            nlines += 1
            cur_idx += 1
            all_idx[cur_idx] = nfilled
            if end != 0 and nlines == end:
                break

            if cur_idx % 5000000 == 0:
                all_feats.flush()
                all_idx.flush()
                assert all_idx[cur_idx] == nfilled
                print("Have filed", cur_idx, nfilled, flush=True)

all_feats.flush()
all_idx.flush()
print("Finished with ", nfilled, "entries, ", nlines, "lines, ", bad_lines, "bad lines", bad_content, "bad contents.")
