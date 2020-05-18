#!/usr/bin/python

import numpy as np
import sys
from sortedcontainers import SortedList
from input_format import *

nlines = 1
bad_lines = 0

all_regs = SortedList()
all_instr_types = SortedList()
all_src_reg_counts = SortedList()
all_src_reg_types = SortedList()
all_src_reg_indices = SortedList()
all_dst_reg_counts = SortedList()
all_dst_reg_types = SortedList()
all_dst_reg_indices = SortedList()
all_context_lengths = SortedList()
all_time_outs = SortedList()
all_time_ins = SortedList()
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

            instr_type = get_instr_type(vals)
            time_out = get_time_out(vals)
            time_in = get_time_in(vals)
            pc = get_fetch_pcoffset(vals)
            depth = get_data_depth(vals)
            fetch_depth = get_fetch_depth(vals)

            src_reg_count = get_n_src_regs(vals)
            src_reg_types = get_src_reg_types(vals)
            src_reg_indices = get_src_reg_indices(vals)
            dst_reg_count = get_n_dst_regs(vals)
            dst_reg_types = get_dst_reg_types(vals)
            dst_reg_indices = get_dst_reg_indices(vals)

            if instr_type not in all_instr_types:
                all_instr_types.add(instr_type)

            if pc not in all_pcs:
                all_pcs.add(pc)

            if time_out not in all_time_outs:
                all_time_outs.add(time_out)

            if time_in not in all_time_ins:
                all_time_ins.add(time_in)

            if depth not in all_depth:
                all_depth.add(depth)

            if fetch_depth not in all_fetch_depth:
                all_fetch_depth.add(fetch_depth)

            if src_reg_count not in all_src_reg_counts:
                all_src_reg_counts.add(src_reg_count)

            for srt in src_reg_types:
                if srt not in all_src_reg_types:
                    all_src_reg_types.add(srt)

            for idx in src_reg_indices:
                if idx not in all_src_reg_indices:
                    all_src_reg_indices.add(idx)

            if dst_reg_count not in all_dst_reg_counts:
                all_dst_reg_counts.add(dst_reg_count)

            for drt in dst_reg_types:
                if drt not in all_dst_reg_types:
                    all_dst_reg_types.add(drt)

            for idx in dst_reg_indices:
                if idx not in all_dst_reg_indices:
                    all_dst_reg_indices.add(idx)

            assert len(src_reg_types) == len(src_reg_indices)
            assert len(dst_reg_types) == len(dst_reg_indices)

            for i in range(len(src_reg_types)):
                feat = [src_reg_types[i], src_reg_indices[i]]
                if feat not in all_regs:
                    all_regs.add(feat)

            for i in range(len(dst_reg_types)):
                feat = [dst_reg_types[i], dst_reg_indices[i]]
                if feat not in all_regs:
                    all_regs.add(feat)

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
print("Bad lines: ", bad_lines)
