cuda_devices = "0,1,2,3,4,5,6,7"
data_set_name = "data_spec_q"
#epoch_num = 1
epoch_num = 100
large_model = True

ori_batch_size = 32 * 1024 * 2
print_threshold = 16
#total_size = 342959816
#testbatchnum = 5200
total_size = 330203367
testbatchnum = 5008
train_max_num = 5000
large_model_scale_factor = 4
