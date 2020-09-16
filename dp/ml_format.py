#!/usr/bin/env python

context_length = 94
inst_length = 53

# fields.
field_fetch_cla = 0
field_fetch_lat = 1
field_completion_cla = 2
field_completion_lat = 3
field_store_cla = 4
field_store_lat = 5

field_op = 6
field_micro = 7
field_mispred = 8
field_cctrl = 9
field_ucctrl = 10
field_dctrl = 11
field_squash_af = 12
field_serial_af = 13
field_serial_be = 14
field_atom = 15
field_storec = 16
field_membar = 17
field_quiesce = 18
field_nonspeculative = 19

field_srcreg_begin = 20
field_srcreg_end   = 27
field_dstreg_begin = 28
field_dstreg_end   = 33

field_fetch_depth  = 34
field_fetch_linec  = 35
field_fetch_pcoff  = 36
field_fetch_walk_begin = 37
field_fetch_walk_end   = 39
field_fetch_pagec   = 40
field_fetch_wb_begin = 41
field_fetch_wb_end   = 42

field_data_depth  = 43
field_data_addrc  = 44
field_data_linec  = 45
field_data_walk_begin = 46
field_data_walk_end   = 48
field_data_pagec   = 49
field_data_wb_begin = 50
field_data_wb_end   = 52

def get_instr(vals):
    first_part = vals[:inst_length]
    second_part = vals[inst_length:]
    return [first_part,second_part]

def split_instr(vals):
    ret = []
    assert len(vals) % inst_length == 0
    while len(vals) > 0:
        fst,vals = get_instr(vals)
        #print(fst)
        ret.append(fst)
    return ret
