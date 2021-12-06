import numpy as np

input_start = 11
#target_length = input_start
#target_length = 3
target_length = 9
inst_length = 11 + 33
input_length = inst_length - 2
context_length = 111

# Data set configuration.
data_set_name = "data_spec_test"
data_file_name = data_set_name + "/all.21u"
data_item_format = np.uint16
total_size = 32978671
total_inst_num = 1520961770
# total batch number is 503.21
testbatchnum = 500
validbatchnum = 448
validbatchsize = 24
num_classes = 0

ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

field_fetch_lat = 0
field_completion_lat = 1
field_store_lat = 2
field_op = 11

min_complete_lat = 6
min_store_lat = 10
