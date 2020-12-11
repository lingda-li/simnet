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

def get_data(arr, s, low, high):
    x = np.copy(arr[low:high,])
    y1 = np.copy(x[:,1:2])
    y2 = np.copy(x[:,3:4])
    y = np.concatenate((y1, y2), axis=1)
    y1 = np.copy(x[:,0:1])
    y1 *= np.sqrt(s['all_var'][0])
    y2 = np.copy(x[:,2:3])
    y2 *= np.sqrt(s['all_var'][2])
    y_cla = np.concatenate((y1, y2), axis=1)
    y_cla = np.rint(y_cla)
    #assert(y_cla.all() >= 0 and y_cla.all() <= 9)
    x[:,0:4] = 0
    x = torch.from_numpy(x.astype('f'))
    y = torch.from_numpy(y.astype('f'))
    y_cla = torch.from_numpy(y_cla.astype(int))
    return x, y, y_cla

f = np.memmap(data_set_name + "/totalall.mmap", dtype=np.float32,
              mode='r',shape=(total_size, context_length*inst_length))
fs = np.load(data_set_name + "/statsall.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), " GPUs, ", device)
print("Train with ", batchnum*batchsize, ", test with", int(0.5*batchsize))

x_test, y_test, y_test_cla = get_data(f, fs, testbatchnum*batchsize, int((testbatchnum+0.5)*batchsize))
x_test_g = x_test.to(device)
y_test_g = y_test.to(device)
y_test_cla_g = y_test_cla.to(device)

loss = nn.MSELoss()
loss_cla = nn.CrossEntropyLoss()
#simnet = CNN3_F(20, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 400)
simnet = CNN3_F_P(22, 64, 2, 64, 2, 0, 2, 128, 2, 1, 2, 256, 2, 0, 400)
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
        x, y, y_cla = get_data(f, fs, didx*stride*batchsize, (didx*stride+1)*batchsize)
        x = x.to(device)
        y = y.to(device)
        y_cla = y_cla.to(device)

        output = simnet(x)
        value = loss(output[:,0:2],y)
        values.append(value.cpu().data)
        value0 = loss_cla(output[:,2:12],y_cla[:,0:1].view(-1))
        values.append(value0.cpu().data)
        value1 = loss_cla(output[:,12:22],y_cla[:,1:2].view(-1))
        values.append(value1.cpu().data)
        optimizer.zero_grad()
        value.backward(retain_graph=True)
        value0.backward(retain_graph=True)
        value1.backward()
        optimizer.step()

    output = simnet(x_test_g)
    value = loss(output[:,0:2],y_test_g)
    value0 = loss_cla(output[:,2:12],y_test_cla_g[:,0:1].view(-1))
    value1 = loss_cla(output[:,12:22],y_test_cla_g[:,1:2].view(-1))
    endt = time.time()
    print(":", endt - startt)
    test_values.append(value.cpu().data)
    test_values.append(value0.cpu().data)
    test_values.append(value1.cpu().data)
    print(test_values)


print(values)
print(test_values)
if saved_model_name != "":
    if torch.cuda.device_count() > 1:
        torch.save(simnet.module, 'models/' + saved_model_name)
    else:
        torch.save(simnet, 'models/' + saved_model_name)
