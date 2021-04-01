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
import pandas as pd
import matplotlib.pyplot as plt
import plotext.plot as plx
import terminalplot as tp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from xgboost import XGBRegressor

data_set_name = sys.argv[1]
batchnum = int(sys.argv[2])
batchsize = 32 * 1024 * 2
#batchsize = 32
total_size = 330203367
testbatchnum = 5008
num = 1
stride=5000
def get_lat(arr, low, high):
         #import ipdb; ipdb.set_trace()
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


if 5000 % batchnum != 0:
      print("Warning: not aligned batch number")
      stride = math.floor(5000 / batchnum)

loss = nn.MSELoss()
f = np.memmap(data_set_name + "/totalall.mmap", dtype=np.float32,
                      mode='r',shape=(total_size, context_length*inst_length))
fs = np.load(data_set_name + "/statsall.npz")


#original test data
faa = np.load(data_set_name + "/test.npz")
x = faa['x']

y = np.copy(x[:,1:2])
y2 = np.copy(x[:,3:4])
y = np.concatenate((y, y2), axis=1)
x[:,0:4] = 0
print("test data shape: ", x.shape, y.shape)
x_test = torch.from_numpy(x.astype('f'))
y_test = torch.from_numpy(y.astype('f'))
x_ = x_test.view(-1, inst_length, context_length)
x_test_single = x_[:,:,0]
#round is used for classifier
fetch_val_test= y_test[:,0].round().numpy()
complete_val_test= y_test[:,1].round().numpy()
store_val_test= y_test[:,2].round().numpy()


#train data
x, y = get_lat(f, 0*batchsize, (5)*batchsize)
x__ = x.view(-1, inst_length, context_length)
x_train_single = x__[:,:,0]
fetch_val= y[:,0].round().numpy()
complete_val= y[:,1].round().numpy()
store_val=  y[:,1].round().numpy()
df= pd.read_csv('labels.txt')


#XGBoost Classfier Model

model_fetch= XGBClassifier( learning_rate =0.1, n_estimators=32, max_depth=5)
model_complete= XGBClassifier( learning_rate =0.1, n_estimators=32, max_depth=5)
model_store= XGBClassifier( learning_rate =0.1, n_estimators=32, max_depth=5)


#XGBoost Regression Model
#model_fetch= XGBRegressor(learning_rate =0.1, n_estimators=32, max_depth=5)
#model_complete= XGBRegressor(learning_rate =0.1, n_estimators=32, max_depth=5)
#model_store= XGBRegressor(learning_rate =0.1, n_estimators=32, max_depth=5)

# fit the fetch_lat
model_fetch.fit(x.numpy(), fetch_val)
#fit the complete_lat
model_complete.fit(x.numpy(), complete_val)
#fit the store_lat
model_store.fit(x.numpy(), store_val)
#model.fit(x.numpy(), store_val)

#test 
y_pred_fetch= model.predict(x_test.numpy())
y_pred_complete= model.predict(x_test.numpy())
y_pred_store= model.predict(x_test.numpy())

#loss
fetch_loss= loss(torch.from_numpy(y_pred_fetch),torch.from_numpy(fetch_val_test))
fetch_loss= loss(torch.from_numpy(y_pred_complete),torch.from_numpy(complete_val_test))
fetch_loss= loss(torch.from_numpy(y_pred_store),torch.from_numpy(store_val_test))


#compute feature importance for model fetch
aa= torch.from_numpy(model_fetch.feature_importances_)
tes, ind = aa.topk(30)
print(tes)
print(ind)
print("XGBClassifier loss: ", pred)
for i in range(30):
        print("Context: %d, Index: %d Feature %s (%f)" % (ind[i]/51,ind[i], df.columns[ind[i]%51], tes[i]))
