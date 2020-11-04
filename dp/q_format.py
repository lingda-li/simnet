#!/usr/bin/env python

context_length = 111
inst_length = 51

# fields.
field_fetch_cla = 0
field_fetch_lat = 1
field_out_cla = 2
field_out_lat = 3

field_op =              4
field_micro =           5
field_mispred =         6
field_cctrl =           7
field_ucctrl =          8
field_dctrl =           9
field_squash_af =      10
field_serial_af =      11
field_serial_be =      12
field_atom =           13
field_storec =         14
field_membar =         15
field_quiesce =        16
field_nonspeculative = 17

field_srcreg_begin = 18
field_srcreg_end   = 25
field_dstreg_begin = 26
field_dstreg_end   = 31

field_fetch_depth  = 32
field_fetch_linec  = 33
field_fetch_pcoff  = 34
field_fetch_walk_begin = 35
field_fetch_walk_end   = 37
field_fetch_pagec   =  38
field_fetch_wb_begin = 39
field_fetch_wb_end   = 40

field_data_depth  = 41
field_data_addrc  = 42
field_data_linec  = 43
field_data_walk_begin = 44
field_data_walk_end   = 46
field_data_pagec   =  47
field_data_wb_begin = 48
field_data_wb_end   = 50

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
