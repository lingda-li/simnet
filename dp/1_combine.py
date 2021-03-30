import argparse
import os
import random
import numpy as np
import sys
from sortedcontainers import SortedList

parser = argparse.ArgumentParser(description="Combine dataset")
parser.add_argument('--size', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()
size = args.size
outputp = args.fname[0]
if len(args.fname) > 1:
    outputp = os.path.join(os.path.dirname(args.fname[0]), "all.qq")
outputp += "u"

lines = SortedList()
nlines = 0
idx = 0

def dump():
    global idx, lines, nlines, outputp
    newlines = []
    for item in lines:
        newlines.append(item)
    #lines = SortedList()
    lines.clear()
    for i in range(10):
        print("Shuffle %d" % i)
        random.shuffle(newlines)
    outputn = outputp
    if idx > 0:
        outputn += str(idx)
    print("Write to %s with %d total" % (outputn, nlines), flush=True)
    with open(outputn, 'w') as f:
        for item in newlines:
            f.write("%s\n" % item)
    newlines.clear()

for i in range(len(args.fname)):
    fname = args.fname[i]
    print("read ", fname, flush=True)
    first = False
    if len(args.fname) == 1:
        first = True
    with open(fname) as f:
        for line in f:
            nlines += 1
            data = line.rstrip('\n')
            lines.add(data)
            if size != 0 and nlines % size == 0:
                dump()
                idx += 1
            if (nlines % 1000000) == 0:
                print("So far have %d total" % (nlines), flush=True)

if size == 0 or nlines % size > 0:
    dump()
