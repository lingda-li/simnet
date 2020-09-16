#!/usr/bin/env python

import numpy as np
import sys
from sortedcontainers import SortedList
from ml_format import *

nlines = 0
bad_lines = 0

all_context_lengths = SortedList()
all_time_ins = SortedList()
all_time_outs = SortedList()
all_time_stores = SortedList()
all_instr_types = SortedList()
all_pcs = SortedList()
all_depth = SortedList()
all_fetch_depth = SortedList()

for i in range(1, len(sys.argv)):
    fname = sys.argv[i]
    print("read ", fname)
    with open(fname) as f:
        for line in f:
            try:
                vals = [int(s) for s in line.rstrip().split(' ')]
            except:
                bad_lines += 1
                continue

            ctxt_len = len(split_instr(vals))
            if ctxt_len not in all_context_lengths:
                all_context_lengths.add(ctxt_len)

            time_in = vals[field_fetch_lat]
            time_out = vals[field_completion_lat]
            time_store = vals[field_store_lat]

            instr_type = vals[field_op]
            pc = vals[field_fetch_pcoff]
            depth = vals[field_data_depth]
            fetch_depth = vals[field_fetch_depth]

            if time_in not in all_time_ins:
                all_time_ins.add(time_in)

            if time_out not in all_time_outs:
                all_time_outs.add(time_out)

            if time_store not in all_time_stores:
                all_time_stores.add(time_store)

            if instr_type not in all_instr_types:
                all_instr_types.add(instr_type)

            if pc not in all_pcs:
                all_pcs.add(pc)

            if depth not in all_depth:
                all_depth.add(depth)

            if fetch_depth not in all_fetch_depth:
                all_fetch_depth.add(fetch_depth)

            nlines = nlines + 1



print("Vals seen for 'Context length':",all_context_lengths, "Len is %d" % len(all_context_lengths))
print("Vals seen for 'Time out':",all_time_outs,"Len is %d" % len(all_time_outs))
print("Vals seen for 'Time in':",all_time_ins,"Len is %d" % len(all_time_ins))
print("Vals seen for 'Instruction type':",all_instr_types, "Len is %d" % len(all_instr_types))
print("Vals seen for '# of source registers':",all_src_reg_counts, "Len is %d" % len(all_src_reg_counts))
print("Vals seen for 'Source register type':",all_src_reg_types, "Len is %d" % len(all_src_reg_types))
print("Vals seen for 'Source register index':",all_src_reg_indices, "Len is %d" % len(all_src_reg_indices))
print("Vals seen for '# of dest registers':",all_dst_reg_counts, "Len is %d" % len(all_dst_reg_counts))
print("Vals seen for 'Dest register type':",all_dst_reg_types, "Len is %d" % len(all_dst_reg_types))
print("Vals seen for 'Dest register index':",all_dst_reg_indices, "Len is %d" % len(all_dst_reg_indices))
print("Vals seen for 'Register':",all_regs, "Len is %d" % len(all_regs))
print("Vals seen for 'PC':",all_pcs, "Len is %d" % len(all_pcs))
print("Good lines: ", nlines, "Bad lines: ", bad_lines)
