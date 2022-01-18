import numpy as np

# Data set configuration.
data_set_dir = 'data_spec_q/'
datasets = [
  (data_set_dir + '503.bwaves_r.seq.mmap', 4531375),
  (data_set_dir + '508.namd_r.seq.mmap', 4396538),
  (data_set_dir + '500.perlbench_r.seq.mmap', 4643818),
  (data_set_dir + '502.gcc_r.seq.mmap', 464695)
]

data_item_format = np.uint16
ori_batch_size = 4096
# total batch number is 877,276.625
testbatchnum = 840000
validbatchnum = 800000
validbatchsize = 256

test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 16) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

seq_length = 256
input_length = 47
tgt_length = 3
input_start = tgt_length
inst_length = input_start + input_length

num_classes = 0
