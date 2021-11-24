import numpy as np

context_length = 111
inst_length = 11 + 33

num_classes = 10
min_complete_lat = 6
min_store_lat = 10

# Data set configuration.
data_set_name = "data_spec_q"
data_file_name = data_set_name + "/all0.qqu.mmap"
data_item_format = np.uint16
total_size = 71433975
# total batch number is 1089.996
testbatchnum = 1080
validbatchnum = 1000
validbatchsize = 40

#data_file_name = data_set_name + "/all.qqu.mmap"
#data_item_format = np.uint16
#total_size = 223711663
## total batch number is 3413.57
#testbatchnum = 3412 # 2720 is the number for 10% test
#validbatchnum = 3300
#validbatchsize = 80

ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size
