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
from models import *

#epoch_num = 1
epoch_num = 100
saved_model_name = sys.argv[1]
data_set_name = sys.argv[2]
batchnum = int(sys.argv[3])
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

def get_lat(arr, low, high):
    x = np.copy(arr[low:high,])
    y1 = np.copy(x[:,1:2])
    y2 = np.copy(x[:,3:4])
    y = np.concatenate((y1, y2), axis=1)
    x[:,0:4] = 0
    #for i in range(1, context_length):
    #  x[:,inst_length*i+2] = 0
    x = torch.from_numpy(x.astype('f'))
    y = torch.from_numpy(y.astype('f'))
    return x, y

f = np.memmap(data_set_name + "/totalall_nn.mmap", dtype=np.float32,
              mode='r',shape=(total_size, context_length*inst_length))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), " GPUs, ", device)
print("Train with ", batchnum*batchsize, ", test with", int(0.5*batchsize))

x_test, y_test = get_lat(f, testbatchnum*batchsize, int((testbatchnum+0.5)*batchsize))
x_test_g = x_test.to(device)
y_test_g = y_test.to(device)

loss = nn.MSELoss()
#simnet = CNN3_F_P(2, 128, 2, 128, 2, 0, 2, 128, 2, 1, 2, 256, 2, 0, 400)
simnet = CNN7_F(2, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 2, 512, 2, 0, 2, 1024, 2, 1, 2, 2048, 2, 0, 2, 4096, 2, 0, 400)
if torch.cuda.device_count() > 1:
    simnet = nn.DataParallel(simnet)
simnet.to(device)
optimizer = torch.optim.Adam(simnet.parameters())
values = []
test_values = []

for i in range(epoch_num):
    print(i, ":", flush=True, end=' ')
    startt = time.time()
    for didx in range(batchnum):
        if didx % print_threshold == 0:
            print('.', flush=True, end='')
        #x, y = get_lat(f, didx*batchsize, (didx+1)*batchsize)
        x, y = get_lat(f, didx*stride*batchsize, (didx*stride+1)*batchsize)
        x = x.to(device)
        y = y.to(device)

        output = simnet(x)
        value = loss(output,y)
        values.append(value.cpu().data)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()

    output = simnet(x_test_g)
    value = loss(output,y_test_g)
    endt = time.time()
    print(":", endt - startt)
    test_values.append(value.cpu().data)
    print(test_values)


print(values)
print(test_values)
if saved_model_name != "":
    if torch.cuda.device_count() > 1:
        torch.save(simnet.module, 'models/' + saved_model_name)
    else:
        torch.save(simnet, 'models/' + saved_model_name)
