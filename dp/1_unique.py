#!/usr/bin/env python

import random
import numpy as np
import sys
from sortedcontainers import SortedList


lines = SortedList()
nlines = 0


for i in range(1, len(sys.argv)):
    fname = sys.argv[i]
    print("read ", fname)
    first = False
    if len(sys.argv) == 1:
        first = True
    with open(fname) as f:
        for line in f:
            # skip first line.
            if first:
                first = False
            elif line.strip():
                nlines = nlines + 1
                data = line.rstrip('\n')
                if data not in lines:
                    lines.add(data)
                if ( (nlines % 500000) == 0):
                     print("So far have %d unique out of %d total" % (len(lines),
                                                                     nlines))

newlines = []
for item in lines:
    newlines.append(item)
for i in range(10):
    print("Shuffle %d" % i)
    random.shuffle(newlines)

if len(sys.argv) > 2:
    fname = "all.ml"
with open(fname + "u", 'w') as f:
    for item in newlines:
        f.write("%s\n" % item)
