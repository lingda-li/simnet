import sys
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import pickle
from sklearn import preprocessing
from datetime import datetime
np.random.seed(0)
from cfg import *
from utils import *
from models import *

if len(sys.argv) < 3:
    print("Illegal arguments.")
    sys.exit()
    
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
batchnum = int(sys.argv[1])
modelnum = len(sys.argv) - 2

if batchnum > train_max_num:
    print("Too large training set.")
    sys.exit()
elif train_max_num % batchnum != 0:
    print("Warning: training set not aligned.")
stride = math.floor(train_max_num / batchnum)
    
if large_model:
    batchsize = ori_batch_size // large_model_scale_factor
    testbatchnum *= large_model_scale_factor
    batchnum *= large_model_scale_factor
    print_threshold *= large_model_scale_factor
else:
    batchsize = ori_batch_size

f = np.memmap(data_set_name + "/totalall.mmap", dtype=np.float32,
              mode='r',shape=(total_size, context_length*inst_length))
fs = np.load(data_set_name + "/statsall.npz")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), " GPUs, ", device)
print("Train", modelnum, "models with", batchnum*batchsize, ", test with", int(0.5*batchsize))

x_test, y_test, y_test_cla = get_data(f, fs, testbatchnum*batchsize, int((testbatchnum+0.5)*batchsize), device)
loss = nn.MSELoss()
loss_cla = nn.CrossEntropyLoss()


model_name = []
simnet = []
optimizer = []
values0 = []
values1 = []
values2 = []
test_values0 = []
test_values1 = []
test_values2 = []
for i in range(modelnum):
    values0.append([])
    values1.append([])
    values2.append([])
    test_values0.append([])
    test_values1.append([])
    test_values2.append([])


# Init models.
for i in range(modelnum):
    cur_model = sys.argv[i + 2]
    model_name.append(generate_model_name(cur_model + " " + datetime.now().strftime("%m%d%y")))
    cur_model = eval(cur_model)
    print("Model", i, ":")
    profile_model(cur_model)
    cur_model = torch.jit.trace(cur_model, torch.rand(1, context_length * inst_length))
    print(cur_model.code, flush=True)
    simnet.append(cur_model)
    if torch.cuda.device_count() > 1:
        simnet[i] = nn.DataParallel(simnet[i])
    simnet[i].to(device)
    optimizer.append(torch.optim.Adam(simnet[i].parameters()))


for e in range(epoch_num):
    # Train 1 epoch.
    print(e, ":", flush=True, end=' ')
    startt = time.time()
    loss_0 = []
    loss_1 = []
    loss_2 = []
    for i in range(modelnum):
        loss_0.append(0.0)
        loss_1.append(0.0)
        loss_2.append(0.0)
    for didx in range(batchnum):
        if didx % print_threshold == 0:
            print('.', flush=True, end='')
        x, y, y_cla = get_data(f, fs, didx*stride*batchsize, (didx*stride+1)*batchsize, device)
        for i in range(modelnum):
            loss_0[i], loss_1[i], loss_2[i] = train_com(simnet[i], x, y, y_cla,
                                                        loss_0[i], loss_1[i], loss_2[i],
                                                        loss, loss_cla, optimizer[i])
    # Finish 1 epoch.
    endt = time.time()
    print(":", endt - startt)
    for i in range(modelnum):
        loss_0[i] /= batchnum
        values0[i].append(loss_0[i])
        loss_1[i] /= batchnum
        values1[i].append(loss_1[i])
        loss_2[i] /= batchnum
        values2[i].append(loss_2[i])
        print(i, ":", loss_0[i], loss_1[i], loss_2[i])
        tloss_0, tloss_1, tloss_2 = test_com(simnet[i], x_test, y_test, y_test_cla, loss, loss_cla)
        test_values0[i].append(tloss_0)
        test_values1[i].append(tloss_1)
        test_values2[i].append(tloss_2)
        print(i, ":", tloss_0, tloss_1, tloss_2)
    if e % save_interval == save_interval - 1:
        for i in range(modelnum):
            #save_model(simnet[i], model_name[i], device)
            save_ts_model(simnet[i], model_name[i])


for i in range(modelnum):
    print("Model", i, "training loss 0:", end=' ')
    print_arr(values0[i])
    print("Model", i, "training loss 1:", end=' ')
    print_arr(values1[i])
    print("Model", i, "training loss 2:", end=' ')
    print_arr(values2[i])
    print("Model", i, "testing loss 0:", end=' ')
    print_arr(test_values0[i])
    print("Model", i, "testing loss 1:", end=' ')
    print_arr(test_values1[i])
    print("Model", i, "testing loss 2:", end=' ')
    print_arr(test_values2[i])
