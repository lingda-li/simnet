# Data set configuration.
data_set_name = "data_spec_postk"
data_file_name = data_set_name + "/all0.qqu.mmap"
total_size = 58946327
# total batch number is 899.45
testbatchnum = 898
validbatchnum = 810

ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + 80) * ori_batch_size

context_length = 210
inst_length = 50

num_classes = 10
min_complete_lat = 6
min_store_lat = 9
