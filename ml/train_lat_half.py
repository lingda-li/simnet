import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from torch.cuda import amp
import torch.nn.functional as F
from itertools import product
import pickle
from torch import optim
from sklearn import preprocessing
np.random.seed(0)
from models import *
import copy
import statistics
from model_helper import prep_param_lists, model_grads_to_master_grads, master_params_to_model_params
#epoch_num = 1
epoch_num = 20
saved_model_name = sys.argv[1]
data_set_name = sys.argv[2]
batchnum = int(sys.argv[3])
batchsize = 16 * 1024 
print_threshold = 16
out_fetch = False
out_comp = False
half = True
scale = False
scale_factor = 128

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), " GPUs, ", device)


x = x[0:int((batchnum+0.5)*batchsize),]
y = y[0:int((batchnum+0.5)*batchsize),]
x = torch.from_numpy(x.astype('f'))
y = torch.from_numpy(y.astype('f'))
x_test = x[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
y_test = y[batchnum*batchsize:int((batchnum+0.5)*batchsize),]
print("Train with ", batchnum*batchsize, ", test with", 0.5*batchsize)

scaler = GradScaler()
loss = nn.MSELoss()
# simnet = CNN3(2 ,5, 64, 5, 64, 5, 256, 400)
simnet = CNN3_P(2, 64, 5, 64, 5, 64, 5, 256, 400)
if half:
    simnet.half()
# if torch.cuda.device_count() > 1:
    # simnet = nn.DataParallel(simnet)

# simnet.to(device)
simnet = simnet.cuda()
if half:
    model_params, master_params = prep_param_lists(simnet)
    optimizer = torch.optim.Adam(master_params)
    # optimizer = torch.optim.Adam(simnet.parameters(),eps=1e-04)
else:
    optimizer = torch.optim.Adam(simnet.parameters(),eps=1e-04)
values = []
test_values = []
time_avg = []

for i in range(epoch_num):
    print(i, "Iteration :", flush=True, end=' ')
    startt = time.time()
    for didx in range(batchnum):
        if didx % print_threshold == 0:
            print('.', flush=True, end='') 
        x_train_now = x[didx*batchsize:(didx+1)*batchsize,]
        y_train_now = y[didx*batchsize:(didx+1)*batchsize,]
        x_train_now = x_train_now.to(device)
        y_train_now = y_train_now.to(device)
        with torch.autograd.set_detect_anomaly(True):
            if half:
                x_train_now= x_train_now.half()
                y_train_now= y_train_now.half()
            output = simnet(x_train_now)
            value = loss(output.float(),y_train_now.float())
            # if scale:
            #     scaled_loss = scale_factor * value.float()
            values.append(value.cpu().data)
            simnet.zero_grad()
            value.backward()
            if half:
                model_grads_to_master_grads(model_params, master_params)
            # if scale:
            #     for param in master_params:
            #         param.grad.data.mul_(1./scale_factor)
            optimizer.step()
            if half:
                master_params_to_model_params(model_params, master_params)

        x_test_g = x_test.to(device)
        if half:
            x_test_g = x_test_g.half()
        y_test_g = y_test.to(device)
        if half:
            y_test_g = y_test_g.half()
        
        output = simnet(x_test_g)
        value = loss(output,y_test_g)
        print(value)
    endt = time.time()
    if i!=0:
        time_avg.append(endt-startt)
    print("Time:", endt - startt)
    test_values.append(value.cpu().data)


# print(values)
print(test_values)
print("Average time:")
print(np.mean(time_avg))
if saved_model_name != "":
    # if torch.cuda.device_count() > 1:
        # torch.save(simnet.module, 'models/' + saved_model_name)
    # else:
    torch.save(simnet, 'models/' + saved_model_name)
