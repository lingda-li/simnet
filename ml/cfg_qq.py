# Data set configuration.
data_set_name = "data_spec_q"
data_file_name = data_set_name + "/all.qqu.mmap"
total_size = 71433975

testbatchnum = 1080
ori_batch_size = 1024 * 64
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size

context_length = 111
inst_length = 50

num_classes = 10