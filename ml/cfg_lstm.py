import numpy as np

# Data set configuration.
data_set_dir = 'data_spec_test/'
datasets = [
  (data_set_dir + '507.cactuBSSN_r.seq.mmap', 4044489),
  (data_set_dir + '508.namd_r.seq.mmap', 4317362),
  (data_set_dir + '500.perlbench_r.seq.mmap', 4635372),
  (data_set_dir + '502.gcc_r.seq.mmap', 63360)
]

data_item_format = np.uint16
ori_batch_size = 4096
# total batch number is 816,286.4375
testbatchnum = 800000
validbatchnum = 720000
validbatchsize = 64

test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

seq_length = 256
input_length = 33
tgt_length = 3
input_start = tgt_length
inst_length = input_start + input_length

num_classes = 0
