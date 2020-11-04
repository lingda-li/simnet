import argparse
import os
import random
import numpy as np
import sys
from sortedcontainers import SortedList

parser = argparse.ArgumentParser(description="Filter duplicated data")
parser.add_argument('fname', nargs='*')
args = parser.parse_args()
if len(args.fname) <= 1:
    print("Must have more than one arguments")
    sys.exit()

lines = SortedList()
nlines = 0
output = args.fname[0]
output += "u"

def dump():
    global lines, nlines, output
    newlines = []
    for item in lines:
        newlines.append(item)
    for i in range(10):
        print("Shuffle %d" % i)
        random.shuffle(newlines)
    print("Write to %s with %d unique" % (output, len(lines)))
    with open(output, 'w') as f:
        for item in newlines:
            f.write("%s\n" % item)

for i in range(len(args.fname)):
    fname = args.fname[i]
    if i == 0:
        print("read to be filtered ", fname)
        with open(fname) as f:
            for line in f:
                data = line.rstrip('\n')
                lines.add(data)
                nlines += 1
        print("Have %d lines initially" % nlines)
    else:
        print("read filter ", fname)
        if i == 1:
            nlines = 0
        with open(fname) as f:
            for line in f:
                data = line.rstrip('\n')
                lines.discard(data)
                nlines += 1
                if (nlines % 1000000) == 0:
                    print("Reduce to %d unique out of %d total" % (len(lines), nlines))

dump()
