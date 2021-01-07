import sys
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
#from IPython.core.display import display, HTML
matplotlib.rcParams.update({'font.size': 16})
np.random.seed(0)
from models import *

inst_type = -2
#inst_type = -1
#inst_type = 25
#inst_type = 26

use_mean = False
#use_mean = True
out_fetch = False
out_comp = False
#use_cuda = True
use_cuda = False

def get_inst_type(vals, n):
  idx = inst_length * n
  if use_mean:
    return np.rint(vals[4 + idx] * np.sqrt(fs['all_var'][4]) + fs['all_mean'][4])
  else:
    return np.rint(vals[4 + idx] * np.sqrt(fs['all_var'][4]))

def get_inst(vals, n):
  if use_mean:
    return np.rint(vals[inst_length*n:inst_length*(n+1)] * np.sqrt(fs['all_var']) + fs['all_mean'])
  else:
    return np.rint(vals[inst_length*n:inst_length*(n+1)] * np.sqrt(fs['all_var']))

if len(sys.argv) == 4:
  print("Use seperate models")
  lat_model_name = sys.argv[1]
  class_model_name = sys.argv[2]
  data_set_name = sys.argv[3]
  combined = False
elif len(sys.argv) == 3:
  print("Use combined model")
  lat_model_name = sys.argv[1]
  data_set_name = sys.argv[2]
  combined = True
else:
  print("Illegal arguments")
  exit()

f = np.load(data_set_name + "/test.npz")
fs = np.load(data_set_name + "/statsall.npz")
x = f['x']

y = np.copy(x[:,0:1])
y *= np.sqrt(fs['all_var'][0])
y2 = np.copy(x[:,2:3])
y2 *= np.sqrt(fs['all_var'][2])
y = np.concatenate((y, y2), axis=1)
y = np.rint(y)
assert(y.all() >= 0 and y.all() <= 9)
print(y)
lat_y = np.copy(x[:,1:2])
lat_y2 = np.copy(x[:,3:4])
lat_y = np.concatenate((lat_y, lat_y2), axis=1)
print(lat_y)
x[:,0:4] = 0
print("shape:", x.shape, y.shape)

x_test = torch.from_numpy(x.astype('f'))

if use_cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lat_simnet = torch.load('models/' + lat_model_name, map_location='cuda')
else:
  lat_simnet = torch.load('models/' + lat_model_name, map_location='cpu')
lat_simnet.eval()
lat_output_all = lat_simnet(x_test)
lat_output_all_np = lat_output_all.detach().numpy()
print("latency output:", lat_output_all)

if not(combined):
  if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simnet = torch.load('models/' + class_model_name, map_location='cuda')
  else:
    simnet = torch.load('models/' + class_model_name, map_location='cpu')
  output_all = simnet(x_test)
  simnet.eval()
  print("class output:", output_all)

for i in range(2):
  y_test = torch.from_numpy(y.astype(int))
  lat_y_test = torch.from_numpy(lat_y.astype('f'))
  y_test = y_test.view(-1)
  if combined:
    output = torch.argmax(lat_output_all[:,10*i+2:10*i+12], dim=1)
  else:
    output = torch.argmax(output_all[:,10*i:10*i+10], dim=1)
  output = output.detach().numpy()
  target = np.squeeze(y[:,i:i+1])
  lat_output = np.squeeze(lat_output_all_np[:,i:i+1])
  lat_target = np.squeeze(lat_y[:,i:i+1])
  print("class output:", output)
  print("class target:", target)
  print("latency output:", lat_output)
  print("latency target:", lat_target)

  lat_output *= np.sqrt(fs['all_var'][2*i+1])
  lat_target *= np.sqrt(fs['all_var'][2*i+1])
  lat_output = np.rint(lat_output)
  lat_target = np.rint(lat_target)
  print("norm latency output:", lat_output)
  print("norm latency target:", lat_target)
  print("latency output shape:", lat_output.shape)

  lat_output = np.where(output < 9, output + 6*i, lat_output)
  print("combined output:", lat_output)

  errs = lat_target - lat_output
  print(errs)
  errs = errs.ravel()
  errs[errs < 0] = -errs[errs < 0]
  #errs[target == 9] = -1
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

  print("cycle avg err:", np.average(errs[errs != -1]))
  print("err std deviation:", np.std(errs[errs != -1]))
  flat_target = lat_target.ravel()
  print("err persentage:", np.sum(errs[errs != -1]) / np.sum(flat_target[errs != -1]))

  his = np.histogram(errs, bins=range(-1, 100))
  print("data percentage:", errs[errs != -1].size / errs.size)
  print(his[0] / errs[errs != -1].size)
  #print(his)
