import numpy as np

# Data set configuration.
#data_set_name = "data_spec_robreg/p128r120"
#data_file_name = data_set_name + "/all.qqu"
data_item_format = np.uint16
#total_size = 81939571
#total_inst_num = 4304420921
# total batch number is 1089.996
testbatchnum = 1080
validbatchnum = 1000
validbatchsize = 40

ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size * 3
test_end = (testbatchnum + 1) * ori_batch_size * 3
valid_start = validbatchnum * ori_batch_size * 3
valid_end = (validbatchnum + validbatchsize) * ori_batch_size * 3

context_length = 191
inst_length = 50

num_classes = 10
#num_classes = 100
min_complete_lat = 6
min_store_lat = 10
