context_length = 210
inst_length = 50

# fields.
field_fetch_lat = 0
field_completion_lat = 1
field_store_lat = 2

field_op               = field_store_lat + 1
field_insq             = field_op + 1
field_micro            = field_insq + 1
field_mispred          = field_micro + 1
field_cctrl            = field_mispred + 1
field_ucctrl           = field_cctrl + 1
field_dctrl            = field_ucctrl + 1
field_squash_af        = field_dctrl + 1
field_serial_af        = field_squash_af + 1
field_serial_be        = field_serial_af + 1
field_atom             = field_serial_be + 1
field_storec           = field_atom + 1
field_membar           = field_storec + 1
field_quiesce          = field_membar + 1
field_nonspeculative   = field_quiesce + 1

field_srcreg_begin     = field_nonspeculative + 1
field_srcreg_end       = field_srcreg_begin + 7
field_dstreg_begin     = field_srcreg_end + 1
field_dstreg_end       = field_dstreg_begin + 5

field_fetch_depth      = field_dstreg_end + 1
field_fetch_linec      = field_fetch_depth + 1
field_fetch_walk_begin = field_fetch_linec + 1
field_fetch_walk_end   = field_fetch_walk_begin + 2
field_fetch_pagec      = field_fetch_walk_end + 1
field_fetch_wb_begin   = field_fetch_pagec + 1
field_fetch_wb_end     = field_fetch_wb_begin + 1

field_data_depth       = field_fetch_wb_end + 1
field_data_addrc       = field_data_depth + 1
field_data_linec       = field_data_addrc + 1
field_data_walk_begin  = field_data_linec + 1
field_data_walk_end    = field_data_walk_begin + 2
field_data_pagec       = field_data_walk_end + 1
field_data_wb_begin    = field_data_pagec + 1
field_data_wb_end      = field_data_wb_begin + 2

assert field_data_wb_end == inst_length - 1

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
