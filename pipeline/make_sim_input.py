#!/usr/bin/python

import numpy as np
import sys
from sortedcontainers import SortedList
from input_format_0 import *

pre_scale = True
use_mean = False

fname = sys.argv[1]
num = int(sys.argv[2])

nlines = 1

if pre_scale:
    fs = np.load("data_spec/statsall.npz")

with open(fname) as f:
    for line in f:
        vals = [ int(s) for s in line.rstrip().split(' ') ]
        feat = make_feature_from_instr(vals)
        if use_mean:
            feat[2:] = (feat[2:] - fs['all_mean'][2:]) / np.sqrt(fs['all_var'][2:])
        else:
            feat[2:] = feat[2:] / np.sqrt(fs['all_var'][2:])
        print(' '.join(map(str, feat)))

        if nlines == num:
            exit()
        nlines += 1
