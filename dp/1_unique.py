import argparse
import os
import random
import numpy as np
import sys
from sortedcontainers import SortedList

parser = argparse.ArgumentParser(description="Unique dataset")
parser.add_argument('--size', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()
size = args.size * 1000000000
outputp = args.fname[0]
if len(args.fname) > 1:
    outputp = os.path.join(os.path.dirname(args.fname[0]), "all.21")
outputp += "u"

lines = SortedList()
nlines = 0
nuniqs = 0
idx = 0
cur_size = 0

def dump():
    global idx, lines, nlines, nuniqs, outputp, cur_size
    print("Ready to dump %d GB" % (cur_size / (1024*1024*1024)), flush=True)
    newlines = []
    for item in lines:
        newlines.append(item)
    #lines = SortedList()
    lines.clear()
    cur_size = 0
    for i in range(10):
        print("Shuffle %d" % i)
        random.shuffle(newlines)
    outputn = outputp
    if idx > 0:
        outputn += str(idx)
    print("Write to %s with %d unique out of %d total" % (outputn, nuniqs, nlines), flush=True)
    with open(outputn, 'w') as f:
        for item in newlines:
            f.write("%s\n" % item)
    newlines.clear()

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read ", fname, flush=True)
    with open(fname) as f:
        skip_count = 0
        for line in f:
            # skip first part.
            if skip_count < args.skip:
                skip_count += 1
                continue
            elif line.strip():
                nlines += 1
                data = line.rstrip('\n')
                if data not in lines:
                    lines.add(data)
                    cur_size += data.__sizeof__()
                    nuniqs += 1
                    #if size != 0 and nuniqs % size == 0:
                    if size != 0 and cur_size >= size:
                        dump()
                        idx += 1
                if (nlines % 1000000) == 0:
                    print("So far have %d unique (%d GB) out of %d total" % (nuniqs, cur_size / (1024*1024*1024), nlines), flush=True)

#if size == 0 or nuniqs % size > 0:
if len(lines) > 0:
    dump()

print("Finally have %d unique out of %d total" % (nuniqs, nlines), flush=True)
