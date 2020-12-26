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
np.random.seed(0)
from utils import *
from models import *

if len(sys.argv) < 4:
    print("Illegal arguments.")
    sys.exit()
    
#epoch_num = 1
epoch_num = 100
data_set_name = sys.argv[1]
batchnum = int(sys.argv[2])
modelnum = len(sys.argv) - 3
batchsize = 32 * 1024 * 2
print_threshold = 16
out_fetch = False
out_comp = False
#total_size = 342959816
#testbatchnum = 5200
total_size = 330203367
testbatchnum = 5008
if 5000 % batchnum != 0:
  print("Warning: not aligned batch number")
stride = math.floor(5000 / batchnum)

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

for i in range(modelnum):
    model_name.append(sys.argv[i + 3])
    if i == 0:
        simnet.append(CNN3_F_P(22, 64, 2, 64, 2, 0, 2, 128, 2, 1, 2, 256, 2, 0, 400))
    else:
        simnet.append(CNN7_F(22, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 2, 512, 2, 0, 2, 1024, 2, 1, 2, 2048, 2, 0, 2, 4096, 2, 0, 400))
    if torch.cuda.device_count() > 1:
        simnet[i] = nn.DataParallel(simnet[i])
    simnet[i].to(device)
    optimizer.append(torch.optim.Adam(simnet[i].parameters()))

for e in range(epoch_num):
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
    if torch.cuda.device_count() > 1:
        torch.save(simnet[i].module, 'models/' + model_name[i])
    else:
        torch.save(simnet[i], 'models/' + model_name[i])
