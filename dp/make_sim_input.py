#!/usr/bin/env python

import numpy as np
import sys
from sortedcontainers import SortedList
from input_format import *

pre_scale = True
use_mean = False

fname = sys.argv[1]
num = int(sys.argv[2])

nlines = 1

if pre_scale:
    fs = np.load(sys.argv[3])

with open(fname) as f:
    for line in f:
        vals = [ int(s) for s in line.rstrip().split(' ') ]
        feat = make_feature_from_instr(vals)
        if use_mean:
            feat[4:] = (feat[4:] - fs['all_mean'][4:]) / np.sqrt(fs['all_var'][4:])
        else:
            feat[4:] = feat[4:] / np.sqrt(fs['all_var'][4:])
        print(' '.join(map(str, feat)))

        if nlines == num:
            exit()
        nlines += 1
