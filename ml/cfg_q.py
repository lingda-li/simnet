run_mode = "train"
#run_mode = "debug"

cuda_devices = "0,1,2,3,4,5,6,7"
data_set_name = "data_spec_q"
epoch_num = 100
large_model = True
#large_model = False
large_model_scale_factor = 4
is_save_model = True
save_interval = 20

ori_batch_size = 32 * 1024 * 2
print_threshold = 16

# Data set configuration.
#total_size = 342959816
#testbatchnum = 5200
total_size = 330203367
testbatchnum = 5008
train_max_num = 5000
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size

#context_length = 94
#inst_length = 33
#inst_length = 39
#inst_length = 43
context_length = 111
inst_length = 51

# Do not suppose to change things below.
data_file_name = data_set_name + "/totalall.mmap"
data_stat_name = data_set_name + "/statsall.npz"

if large_model:
    testbatchnum *= large_model_scale_factor
    print_threshold *= large_model_scale_factor

if run_mode == "debug":
    cuda_devices = "1"
    epoch_num = 1
    ori_batch_size = 1024 * 8
    is_save_model = False
