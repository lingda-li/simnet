import sys
import os
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

f = np.load(data_set_name + "/totalall.npz")
fs = np.load(data_set_name + "/statsall.npz")
x = f['x']

num_instances, feature_dim = x.shape
print(num_instances, feature_dim)

y = np.copy(x[:,0:1])
y *= np.sqrt(fs['all_var'][0])
y2 = np.copy(x[:,2:3])
y2 *= np.sqrt(fs['all_var'][2])
y = np.concatenate((y, y2), axis=1)
y = np.rint(y)
assert(y.all() >= 0 and y.all() <= 9)
x[:,0:4] = 0
print(x.shape)
print(y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), " GPUs, ", device)

x = x[0:int((batchnum+0.5)*batchsize),]
y = y[0:int((batchnum+0.5)*batchsize),]
x = torch.from_numpy(x.astype('f'))
y = torch.from_numpy(y.astype(int))
x_test = x[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
y_test = y[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
print("Train with ", batchnum*batchsize, ", test with", 0.5*batchsize)

loss = nn.CrossEntropyLoss()
simnet = CNN3_P(20, 64, 5, 64, 5, 64, 5, 256, 400)
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
        x_train_now = x[didx*batchsize:(didx+1)*batchsize,]
        y_train_now = y[didx*batchsize:(didx+1)*batchsize,]
        x_train_now = x_train_now.to(device)
        y_train_now = y_train_now.to(device)

        output = simnet(x_train_now)
        value0 = loss(output[:,0:10],y_train_now[:,0:1].view(-1))
        values.append(value0.cpu().data)
        value1 = loss(output[:,10:20],y_train_now[:,1:2].view(-1))
        values.append(value1.cpu().data)
        optimizer.zero_grad()
        value0.backward(retain_graph=True)
        value1.backward()
        optimizer.step()

    x_test_g = x_test.to(device)
    y_test_g = y_test.to(device)
    output = simnet(x_test_g)
    value0 = loss(output[:,0:10],y_test_g[:,0:1].view(-1))
    value1 = loss(output[:,10:20],y_test_g[:,1:2].view(-1))
    endt = time.time()
    print(":", endt - startt)
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
