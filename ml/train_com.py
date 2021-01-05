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
#simnet = CNN3_F_P_COM(64, 2, 64, 2, 0, 2, 128, 2, 1, 2, 256, 2, 0, 400)
simnet = CNN3_F_P(22, 64, 2, 64, 2, 0, 2, 128, 2, 1, 2, 256, 2, 0, 400)
if torch.cuda.device_count() > 1:
    simnet = nn.DataParallel(simnet)
simnet.to(device)
optimizer = torch.optim.Adam(simnet.parameters())
values0 = []
values1 = []
values2 = []
test_values0 = []
test_values1 = []
test_values2 = []

for i in range(epoch_num):
    print(i, ":", flush=True, end=' ')
    startt = time.time()
    epoch_loss_0 = 0.0
    epoch_loss_1 = 0.0
    epoch_loss_2 = 0.0
    for didx in range(batchnum):
        if didx % print_threshold == 0:
            print('.', flush=True, end='')
        x, y, y_cla = get_data(f, fs, didx*stride*batchsize, (didx*stride+1)*batchsize)
        x = x.to(device)
        y = y.to(device)
        y_cla = y_cla.to(device)

        output = simnet(x)
        #output, fc, rc = simnet(x)
        value0 = loss(output[:,0:2],y)
        epoch_loss_0 += value0.cpu().item()
        value1 = loss_cla(output[:,2:12],y_cla[:,0:1].view(-1))
        epoch_loss_1 += value1.cpu().item()
        value2 = loss_cla(output[:,12:22],y_cla[:,1:2].view(-1))
        epoch_loss_2 += value2.cpu().item()
        optimizer.zero_grad()
        #value0.backward(retain_graph=True)
        #value1.backward(retain_graph=True)
        #value2.backward()
        all_loss = 20 * value0 + value1 + value2
        all_loss.backward()
        optimizer.step()

    endt = time.time()
    print(":", endt - startt)
    epoch_loss_0 /= batchnum
    values0.append(epoch_loss_0)
    epoch_loss_1 /= batchnum
    values1.append(epoch_loss_1)
    epoch_loss_2 /= batchnum
    values2.append(epoch_loss_2)
    print(epoch_loss_0, epoch_loss_1, epoch_loss_2)
    output = simnet(x_test_g)
    #output, fc, rc = simnet(x_test_g)
    value0 = loss(output[:,0:2],y_test_g)
    value1 = loss_cla(output[:,2:12],y_test_cla_g[:,0:1].view(-1))
    value2 = loss_cla(output[:,12:22],y_test_cla_g[:,1:2].view(-1))
    print(value0.cpu().item(), value1.cpu().item(), value2.cpu().item())
    test_values0.append(value0.cpu().item())
    test_values1.append(value1.cpu().item())
    test_values2.append(value2.cpu().item())


print("Training loss 0: ", values0)
print("Training loss 1: ", values1)
print("Training loss 2: ", values2)
print("Testing loss 0", test_values0)
print("Testing loss 1", test_values1)
print("Testing loss 2", test_values2)
if saved_model_name != "":
    if torch.cuda.device_count() > 1:
        torch.save(simnet.module, 'models/' + saved_model_name)
    else:
        torch.save(simnet, 'models/' + saved_model_name)
