#!/usr/bin/python

src_reg_num = 8
dst_reg_num = 6

default_val = 0
classify = True

instr_type_mapping = {
  0 : 0,
  1 : 1,
  2 : 2,
  3 : 3,
  4 : 4,
  5 : 5,
  6 : 6,
  7 : 7,
  8 : 8,
  9 : 9,
  10: 10,
  11: 11,
  12: 12,
  13: 13,
  14: 14,
  15: 15,
  16: 16,
  17: 17,
  18: 18,
  19: 19,
  20: 20,
  24: 21,
  25: 22,
  26: 23,
  28: 24,
  30: 25,
  31: 26,
  32: 27,
  33: 28,
  38: 29,
  39: 30,
  40: 31,
 -1 : 32,
 -2 : 33,
 -3 : 34,
 -4 : 35,
 -5 : 36,
 -6 : 37,
 -7 : 38,
 -8 : 39,
 -39: 40,
 -40: 41
}

reg_mapper = {
  0: {
    0  : 0,
    1  : 1,
    2  : 2,
    3  : 3,
    4  : 4,
    5  : 5,
    6  : 6,
    7  : 7,
    8  : 8,
    9  : 9,
    10 : 10,
    11 : 11,
    12 : 12,
    13 : 13,
    14 : 14,
    15 : 15,
    16 : 16,
    17 : 17,
    18 : 18,
    19 : 19,
    20 : 20,
    21 : 21,
    22 : 22,
    23 : 23,
    24 : 24,
    25 : 25,
    26 : 26,
    27 : 27,
    28 : 28,
    29 : 29,
    30 : 30,
    31 : 31,
    34 : 32,
    35 : 33,
    43 : 34,
  },
  2: {
    0  : 35,
    1  : 36,
    2  : 37,
    3  : 38,
    4  : 39,
    5  : 40,
    6  : 41,
    7  : 42,
    8  : 43,
    9  : 44,
    10 : 45,
    11 : 46,
    12 : 47,
    13 : 48,
    14 : 49,
    15 : 50,
    16 : 51,
    17 : 52,
    18 : 53,
    19 : 54,
    20 : 55,
    21 : 56,
    22 : 57,
    23 : 58,
    24 : 59,
    25 : 60,
    26 : 61,
    27 : 62,
    28 : 63,
    29 : 64,
    30 : 65,
    31 : 66,
    32 : 67,
    33 : 68,
    34 : 69,
    35 : 70,
    40 : 71,
    41 : 72,
    42 : 73,
  },
  4: {
    0  : 74,
    1  : 75,
    2  : 76,
    3  : 77,
    4  : 78,
    5  : 79,
    6  : 80,
    7  : 81,
    8  : 82,
    9  : 83,
    10 : 84,
    11 : 85,
    12 : 86,
    13 : 87,
    14 : 88,
    15 : 89,
    16 : 90,
  },
  5: {
    0  : 91,
    1  : 92,
    2  : 93,
    3  : 94,
    4  : 95,
    5  : 96,
  },
  6: {
    0   : 97,
    1   : 98,
    17  : 99,
    20  : 100,
    29  : 101,
    394 : 102,
    395 : 103,
    400 : 104,
    403 : 105,
    405 : 106,
    410 : 107,
    412 : 108,
    414 : 109,
    424 : 110,
    425 : 111,
    426 : 112,
    430 : 113,
    431 : 114,
    432 : 115,
    447 : 116,
    456 : 117,
    472 : 118,
    485 : 119,
    486 : 120,
    488 : 121,
    490 : 122,
    549 : 123,
    550 : 124,
    551 : 125,
    557 : 126,
    561 : 127,
    562 : 128,
    605 : 129,
  }
}


def get_time_in(vals):
    return vals[0]

def get_time_out(vals):
    return vals[1]

def get_instr_type(vals):
    return vals[2]

def get_instr_micro(vals):
    return vals[3]

def get_instr_mispredict(vals):
    return vals[4]

def get_n_src_regs(vals):
    return vals[5]

def get_n_dst_regs(vals):
    return vals[6 + 2*get_n_src_regs(vals)]

def get_src_reg_types(vals):
    ret = []
    n_src = get_n_src_regs(vals)

    if n_src == 0:
        return []
    for i in range(6, 6+2*n_src, 2):
        ret.append(vals[i])
    return ret

def get_src_reg_indices(vals):
    ret = []
    n_src = get_n_src_regs(vals)

    if n_src == 0:
        return []
    for i in range(6, 6+2*n_src, 2):
        ret.append(vals[i+1])
    return ret

def get_dst_reg_types(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    if n_dst == 0:
        return []
    for i in range(7+2*n_src, 7+2*n_src+2*n_dst, 2):
        ret.append(vals[i])
    return ret

def get_dst_reg_indices(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    if n_dst == 0:
        return []
    for i in range(7+2*n_src, 7+2*n_src+2*n_dst, 2):
        ret.append(vals[i+1])
    return ret

def get_fetch_depth(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[7+2*n_src+2*n_dst]

def get_fetch_linec(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[8+2*n_src+2*n_dst]

def get_fetch_pcoffset(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[9+2*n_src+2*n_dst]

def get_fetch_walk_depth(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    for i in range(10+2*n_src+2*n_dst, 13+2*n_src+2*n_dst):
        ret.append(vals[i] + 1)
    return ret

def get_fetch_pagec(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    for i in range(13+2*n_src+2*n_dst, 16+2*n_src+2*n_dst):
        ret.append(vals[i])
    return ret

def get_data_depth(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[16+2*n_src+2*n_dst] + 1

def get_data_dep(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[17+2*n_src+2*n_dst]

def get_data_linedep(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    return vals[18+2*n_src+2*n_dst]

def get_data_walk_depth(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    for i in range(19+2*n_src+2*n_dst, 22+2*n_src+2*n_dst):
        ret.append(vals[i] + 1)
    return ret

def get_data_pagec(vals):
    ret = []
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    for i in range(22+2*n_src+2*n_dst, 25+2*n_src+2*n_dst):
        ret.append(vals[i])
    return ret

def get_instr(vals):
    n_src = get_n_src_regs(vals)
    n_dst = get_n_dst_regs(vals)
    last_feat = 25+2*n_src+2*n_dst
    first_part = vals[:last_feat]
    second_part = vals[last_feat:]
    return [first_part,second_part]

def split_instr(vals):
    ret = []
    while len(vals) > 0:
        fst,vals = get_instr(vals)
        #print(fst)
        ret.append(fst)
    return ret

def zeros(length):
    return [default_val]*length

def register_feature(reg_types,reg_indices,reg_num):
    reg_feature = []
    reg_count = len(reg_indices)
    assert len(reg_indices) == len(reg_types)
    assert reg_count <= reg_num
    for i in range(reg_num):
        if i < reg_count:
            reg_feature += [reg_mapper[reg_types[i]][reg_indices[i]] + 1]
        else:
            reg_feature += [default_val]

    return reg_feature

def make_feature_from_instr(vals): # Vals is one instruction.
    op = get_instr_type(vals)
    micro = get_instr_micro(vals)
    mp = get_instr_mispredict(vals)
    i_d = get_fetch_depth(vals)
    i_linec = get_fetch_linec(vals)
    pc_offset = get_fetch_pcoffset(vals)
    i_wd = get_fetch_walk_depth(vals)
    i_pagec = get_fetch_pagec(vals)
    d_d = get_data_depth(vals)
    d_addrc = get_data_dep(vals)
    d_linec = get_data_linedep(vals)
    d_wd = get_data_walk_depth(vals)
    d_pagec = get_data_pagec(vals)

    src_reg_count = get_n_src_regs(vals)
    src_reg_types = get_src_reg_types(vals)
    src_reg_indices = get_src_reg_indices(vals)
    dst_reg_count = get_n_dst_regs(vals)
    dst_reg_types = get_dst_reg_types(vals)
    dst_reg_indices = get_dst_reg_indices(vals)

    instr_type_feature = [instr_type_mapping[op] + 1] + [micro] + [mp] + [i_d] + [i_linec] + [pc_offset] + i_wd + i_pagec + [d_d] + [d_addrc] + [d_linec] + d_wd + d_pagec

    src_reg_feature = register_feature(src_reg_types, src_reg_indices, src_reg_num)
    dst_reg_feature = register_feature(dst_reg_types, dst_reg_indices, dst_reg_num)

    time_in = get_time_in(vals)
    time_out = get_time_out(vals)
    if classify:
        if time_in >= 0 and time_in <= 8:
            fetch_class_feature = time_in
        else:
            fetch_class_feature = 9
        if time_out >= 6 and time_out <= 14:
            out_class_feature = time_out - 6
        else:
            out_class_feature = 9
        feat = [fetch_class_feature] + [time_in] + [out_class_feature] + [time_out] + instr_type_feature + src_reg_feature + dst_reg_feature
    else:
        feat = [time_in] + [time_out] + instr_type_feature + src_reg_feature + dst_reg_feature

    return feat
