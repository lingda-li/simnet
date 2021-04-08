import numpy as np

# Data set configuration.
data_set_name = "data_spec_pc"

data_file_name = data_set_name + "/all.qqu.mmap"
data_item_format = np.uint16
total_size = 11192484
# total batch number is 170.78
testbatchnum = 152
validbatchnum = 160
validbatchsize = 8

ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

context_length = 111
inst_length = 50

num_classes = 10
min_complete_lat = 6
min_store_lat = 10
