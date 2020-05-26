import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import matplotlib
import pickle
from sklearn import preprocessing
from IPython.core.display import display, HTML
matplotlib.rcParams.update({'font.size': 16})
np.random.seed(0)
from models import *

loaded_model_name = "specdc_cnn_3_latonly_l64_64_052120_cpu"
data_set_name = "data"
inst_type = -2
#inst_type = -1
#inst_type = 25
#inst_type = 26
batchnum = 16 * 16 * 2
batchsize = 32 * 1024 * 2
pre_scale = True
use_mean = False
#use_mean = True
out_fetch = False
out_comp = False
use_cuda = False

if pre_scale:
  fs = np.load(data_set_name + "/statsall.npz")

def get_inst_type(vals, n):
  idx = inst_length * n
  if pre_scale:
    if use_mean:
      return np.rint(vals[4 + idx] * np.sqrt(fs['all_var'][4]) + fs['all_mean'][4])
    else:
      return np.rint(vals[4 + idx] * np.sqrt(fs['all_var'][4]))
  else:
    assert(not(use_mean))
    return vals[4 + idx]

def get_inst(vals, n):
  if use_mean:
    return np.rint(vals[inst_length*n:inst_length*(n+1)] * np.sqrt(fs['all_var']) + fs['all_mean'])
  else:
    return np.rint(vals[inst_length*n:inst_length*(n+1)] * np.sqrt(fs['all_var']))

f = np.load(data_set_name + "/test.npz")
x = f['x']

y = np.copy(x[:,1:2])
y2 = np.copy(x[:,3:4])
y = np.concatenate((y, y2), axis=1)
x[:,0:4] = 0
print(x.shape)
print(y.shape)

## clear depth
#for i in range(context_length):
#  x[:,inst_length*i+7] = 0

x_test = torch.from_numpy(x.astype('f'))
y_test = torch.from_numpy(y.astype('f'))

if use_cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  simnet = torch.load('models/' + loaded_model_name, map_location='cuda')
else:
  simnet = torch.load('models/' + loaded_model_name, map_location='cpu')
output = simnet(x_test)
simnet.eval()
loss = nn.MSELoss()
if use_cuda:
  y_test = y_test.to(device)
  torch.save(simnet.module, 'models/' + loaded_model_name + '_cpu')
value = loss(output,y_test)
print(value.data)
if use_cuda:
  output = output.cpu().detach().numpy()
  target = y_test.cpu().detach().numpy()
else:
  output = output.detach().numpy()
  target = y_test.detach().numpy()


print(target)
print(output)
if pre_scale:
  if out_fetch:
    output = output * np.sqrt(fs['all_var'][1:2])
    target = target * np.sqrt(fs['all_var'][1:2])
  elif out_comp:
    output = output * np.sqrt(fs['all_var'][3:4])
    target = target * np.sqrt(fs['all_var'][3:4])
  else:
    factor = np.concatenate((np.sqrt(fs['all_var'][1:2]), np.sqrt(fs['all_var'][3:4])), axis=0)
    print(factor)
    output = output * factor
    target = target * factor
  if use_mean:
    if out_fetch:
      output += fs['all_mean'][0:1]
      target += fs['all_mean'][0:1]
    elif out_comp:
      output += fs['all_mean'][1:2]
      target += fs['all_mean'][1:2]
    else:
      mean = np.concatenate((fs['all_mean'][1:2], fs['all_mean'][3:4]), axis=0)
      output += mean
      target += mean
print(target)
print(output)
output = np.rint(output)
target = np.rint(target)
print(target)
print(output)

#plt.figure(figsize=(42,10))
#plt.plot(target,'.',label='True latency')
#plt.plot(output,'.',label='Prediction')
#plt.gca().set_yticks(np.linspace(start=min(output),stop=max(output),num=5))
#plt.ylabel("Execution time")
#plt.legend()
#plt.show()


#target_perm = target[:,0].argsort()
#plt.figure(figsize=(20,8))
#plt.plot(target[target_perm],'.',label="True latency")
#plt.plot(output[target_perm],'+',label="Prediction")
#plt.ylabel("Execution cycle #")
#plt.legend()
#plt.show()


allerrs = target - output
print(allerrs)
for i in range(2):
  errs = allerrs[:,i:i+1]
  errs = errs.ravel()
  errs[errs < 0] = -errs[errs < 0]
  print(errs)
  print(errs.size)

  if inst_type >= -1:
    for i in range(errs.size):
      cur_inst_type = get_inst_type(x[i], 0)
      if not(use_mean):
        cur_inst_type -= 1
      #print(cur_inst_type)
      assert cur_inst_type >= 0 and cur_inst_type < 37
      if inst_type >= 0 and cur_inst_type != inst_type:
        errs[i] = -1
      elif inst_type == -1 and (cur_inst_type == 25 or cur_inst_type == 26):
        errs[i] = -1
    print(errs)

  print(np.average(errs[errs != -1]))

  his = np.histogram(errs, bins=range(-1, 100))
  print(errs[errs != -1].size / errs.size)
  print(his[0] / errs[errs != -1].size)
  #print(his)
