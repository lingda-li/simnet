import argparse
import os
import random
import numpy as np
import sys
from sortedcontainers import SortedList

parser = argparse.ArgumentParser(description="Unique dataset")
parser.add_argument('--size', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()
size = args.size
outputp = args.fname[0]
if len(args.fname) > 1:
    outputp = os.path.join(os.path.dirname(args.fname[0]), "all.ML")
outputp += "U"

lines = SortedList()
nlines = 0
nuniqs = 0
idx = 0

def dump():
    global idx, lines, nlines, nuniqs, outputp
    newlines = []
    for item in lines:
        newlines.append(item)
    lines = SortedList()
    for i in range(10):
        print("Shuffle %d" % i)
        random.shuffle(newlines)
    outputn = outputp
    if idx > 0:
        outputn += str(idx)
    print("Write to %s with %d unique out of %d total" % (outputn, nuniqs, nlines))
    with open(outputn, 'w') as f:
        for item in newlines:
            f.write("%s\n" % item)

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read ", fname)
    first = False
    if len(args.fname) == 1:
        first = True
    with open(fname) as f:
        for line in f:
            # skip first line.
            if first:
                first = False
            elif line.strip():
                nlines += 1
                data = line.rstrip('\n')
                if data not in lines:
                    lines.add(data)
                    nuniqs += 1
                    if size != 0 and nuniqs % size == 0:
                        dump()
                        idx += 1
                if (nlines % 1000000) == 0:
                    print("So far have %d unique out of %d total" % (nuniqs, nlines))

if size == 0 or nuniqs % size > 0:
    dump()
