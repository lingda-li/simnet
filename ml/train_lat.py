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
#saved_model_name = ""
saved_model_name = "spec_cnn_3p_latonly_l64_64_042020"
data_set_name = "data_spec"
batchnum = 16 * 16 * 4
batchsize = 32 * 1024
print_threshold = 16
out_fetch = False
out_comp = False

f = np.load(data_set_name + "/totalall.npz")
fs = np.load(data_set_name + "/statsall.npz")
x = f['x']

num_instances, feature_dim = x.shape
print(num_instances, feature_dim)

y = np.copy(x[:,1:2])
y2 = np.copy(x[:,3:4])
y = np.concatenate((y, y2), axis=1)
x[:,0:4] = 0
print(x.shape)
print(y.shape)

x = torch.from_numpy(x.astype('f'))
y = torch.from_numpy(y.astype('f'))
x_test = x[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
y_test = y[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
print("Train with ", batchnum*batchsize, ", test with", 0.5*batchsize)

loss = nn.MSELoss()
simnet = CNN3_P(2, 64, 5, 64, 5, 64, 5, 256, 400)
optimizer = torch.optim.Adam(simnet.parameters())
values = []
test_values = []

device = torch.device("cuda:12" if torch.cuda.is_available() else "cpu")
print(device)
simnet.to(device)

for i in range(epoch_num):
    print(i, ":", flush=True, end=' ')
    startt = time.time()
    for didx in range(batchnum):
        if didx % print_threshold == 0:
            print('.', flush=True, end='')
        x_train_now = x[didx*batchsize:(didx+1)*batchsize,]
        y_train_now = y[didx*batchsize:(didx+1)*batchsize,]
        x_train_now = x_train_now.to(device)
        #y_train_now = y_train_now.view(-1)
        y_train_now = y_train_now.to(device)

        output = simnet(x_train_now)
        value = loss(output,y_train_now)
        values.append(value.cpu().data)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()

    x_test_g = x_test.to(device)
    #y_test = y_test.view(-1)
    y_test_g = y_test.to(device)
    output = simnet(x_test_g)
    value = loss(output,y_test_g)
    endt = time.time()
    print(":", endt - startt)
    test_values.append(value.cpu().data)
    print(test_values)


print(values)
print(test_values)
if saved_model_name != "":
  torch.save(simnet, 'models/' + saved_model_name)
