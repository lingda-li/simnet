#!/usr/bin/env python

import numpy as np
import sys
from sortedcontainers import SortedList
from rr_format import *

nlines = 0
bad_lines = 0
bad_content = 0
size = 0

maxval = None
minval = None

all_context_lengths = SortedList()
all_time_ins = SortedList()
#all_time_outs = SortedList()
all_instr_types = SortedList()

for i in range(1, len(sys.argv)):
    fname = sys.argv[i]
    print("read", fname, flush=True)
    with open(fname) as f:
        for line in f:
            print("Have checked", nlines, flush=True)
            try:
                vals = [int(s) for s in line.rstrip().split(' ')]
            except:
                bad_lines += 1
                continue

            if len((vals)) % inst_length != 0:
                bad_content += 1
                continue

            ctxt_len = len((vals)) / inst_length
            size += ctxt_len
            if ctxt_len not in all_context_lengths:
                all_context_lengths.add(ctxt_len)

            if maxval == None or maxval < max(vals):
                maxval = max(vals)

            if minval == None or minval > min(vals):
                minval = min(vals)

            time_in = vals[field_fetch_lat]
            #time_out = vals[field_out_lat]
            instr_type = vals[field_op]

            if time_in not in all_time_ins:
                all_time_ins.add(time_in)

            #if time_out not in all_time_outs:
            #    all_time_outs.add(time_out)

            if instr_type not in all_instr_types:
                all_instr_types.add(instr_type)

            nlines += 1
            if nlines % 5000000 == 0:
                print("Have checked", nlines, flush=True)


print("Max val:", maxval, "Min val:", minval)
print("Total instruction #:", size)
print("Good lines:", nlines, "Bad lines:", bad_lines, "Bad content:", bad_content)
print("Vals seen for 'Context length':", all_context_lengths, "Len is %d" % len(all_context_lengths))
print("Vals seen for 'Time in':", all_time_ins,"Len is %d" % len(all_time_ins))
#print("Vals seen for 'Time out':", all_time_outs,"Len is %d" % len(all_time_outs))
print("Vals seen for 'Instruction type':",all_instr_types, "Len is %d" % len(all_instr_types))
