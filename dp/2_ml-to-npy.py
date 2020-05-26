#!/usr/bin/env python

import numpy as np
import sys
from input_format import *

context_length = 94
iter_num = 1024 * 16 * 32 * 16
inst_length = 39

def make_feature_from_context(vals): # Vals is all instructions concatenated.
    feat = []
    instrs = split_instr(vals)
    inst_num = len(instrs)
    assert inst_num <= context_length
    for i in range(context_length):
        if i < inst_num:
            #print(instrs[i])
            feat_from_instr = make_feature_from_instr(instrs[i])
            #print('feat_from_instr len is' , len(feat_from_instr))
            #print(feat_from_instr)
            #print(len(feat_from_instr))
            assert len(feat_from_instr) == inst_length
            feat += feat_from_instr
        else:
            feat += zeros(inst_length)
    return feat, inst_num # inst_length*context_length


fname = sys.argv[1]

nlines = 1
idx = 0
bad_lines = 0
bad_content = 0

all_feats = []
lengths = []
with open(fname) as f:
    for line in f:
        try:
            vals = [int(s) for s in line.rstrip().split(' ')]
        except:
            bad_lines += 1
            continue
        #print(vals)
        try:
            feat, length = make_feature_from_context(vals)
        except:
            bad_content += 1
            print(vals)
            print(bad_content, bad_lines)
            continue
        #print('feat_from_cxt len is' , len(feat))
        #print(feat)
        all_feats.append(feat)
        lengths.append(length)
        if nlines == 1:
            print(feat)

        if ((nlines % iter_num) == 0):
            print("So far have %d" % nlines)
            x = np.array(all_feats)
            np.savez_compressed(fname + ".t" + str(idx), x=x)
            all_feats = []
            lengths = []
            idx = idx + 1
        nlines = nlines + 1

if all_feats:
    print("The last one has %d" % (nlines - 1))
    x = np.array(all_feats)
    np.savez_compressed(fname + ".t" + str(idx), x=x)
