import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import pickle
from torch import optim
from sklearn import preprocessing
np.random.seed(0)
from models_n import *
import copy
import statistics

custom = True

saved_model_name = sys.argv[1]
simnet = torch.load(saved_model_name, map_location='cpu')

shape = []

if custom:
    convp_w = simnet.convp.weight.detach().numpy().flatten()
    shape.append(len(convp_w))
    convp_b = simnet.convp.bias.detach().numpy().flatten()
    shape.append(len(convp_b))
conv1_w = simnet.conv1.weight.detach().numpy().flatten()
shape.append(len(conv1_w))
conv1_b = simnet.conv1.bias.detach().numpy().flatten()
shape.append(len(conv1_b))
conv2_w = simnet.conv2.weight.detach().numpy().flatten()
shape.append(len(conv2_w))
conv2_b = simnet.conv2.bias.detach().numpy().flatten()
shape.append(len(conv2_b))
conv3_w = simnet.conv3.weight.detach().numpy().flatten()
shape.append(len(conv3_w))
conv3_b = simnet.conv3.bias.detach().numpy().flatten()
shape.append(len(conv3_b))
fc1_w = simnet.fc1.weight.detach().numpy().flatten()
shape.append(len(fc1_w))
fc1_b = simnet.fc1.bias.detach().numpy().flatten()
shape.append(len(fc1_b))
fc2_w = simnet.fc2.weight.detach().numpy().flatten()
shape.append(len(fc2_w))
fc2_b = simnet.fc2.bias.detach().numpy().flatten()
shape.append(len(fc2_b))
dims = np.array(shape)

if custom:
    with open("../params/CNN3_P/dims.bin", "wb+") as f:
        dims.tofile(f)
        
    with open("../params/CNN3_P/convp_w.bin", "wb+") as f:
        convp_w.tofile(f)

    with open("../params/CNN3_P/convp_b.bin", "wb+") as f:
        convp_b.tofile(f)

    with open("../params/CNN3_P/conv1_w.bin", "wb+") as f:
        conv1_w.tofile(f)

    with open("../params/CNN3_P/conv1_b.bin", "wb+") as f:
        conv1_b.tofile(f)

    with open("../params/CNN3_P/conv2_w.bin", "wb+") as f:
        conv2_w.tofile(f)

    with open("../params/CNN3_P/conv2_b.bin", "wb+") as f:
        conv2_b.tofile(f)

    with open("../params/CNN3_P/conv3_w.bin", "wb+") as f:
        conv3_w.tofile(f)

    with open("../params/CNN3_P/conv3_b.bin", "wb+") as f:
        conv3_b.tofile(f)

    with open("../params/CNN3_P/conv1_w.bin", "wb+") as f:
        conv1_w.tofile(f)

    with open("../params/CNN3_P/fc1_w.bin", "wb+") as f:
        fc1_w.tofile(f)

    with open("../params/CNN3_P/fc1_b.bin", "wb+") as f:
        fc1_b.tofile(f)

    with open("../params/CNN3_P/fc2_w.bin", "wb+") as f:
        fc2_w.tofile(f)

    with open("../params/CNN3_P/fc2_b.bin", "wb+") as f:
        fc2_b.tofile(f)


else:
    dims = np.array(shape)
    with open("../params/CNN3/dims.bin", "wb+") as f:
        dims.tofile(f)

    with open("../params/CNN3/conv1_w.bin", "wb+") as f:
        conv1_w.tofile(f)

    with open("../params/CNN3/conv1_b.bin", "wb+") as f:
        conv1_b.tofile(f)

    with open("../params/CNN3/conv2_w.bin", "wb+") as f:
        conv2_w.tofile(f)

    with open("../params/CNN3/conv2_b.bin", "wb+") as f:
        conv2_b.tofile(f)

    with open("../params/CNN3/conv3_w.bin", "wb+") as f:
        conv3_w.tofile(f)

    with open("../params/CNN3/conv3_b.bin", "wb+") as f:
        conv3_b.tofile(f)

    with open("../params/CNN3/conv1_w.bin", "wb+") as f:
        conv1_w.tofile(f)

    with open("../params/CNN3/fc1_w.bin", "wb+") as f:
        fc1_w.tofile(f)

    with open("../params/CNN3/fc1_b.bin", "wb+") as f:
        fc1_b.tofile(f)

    with open("../params/CNN3/fc2_w.bin", "wb+") as f:
        fc2_w.tofile(f)

    with open("../params/CNN3/fc2_b.bin", "wb+") as f:
        fc2_b.tofile(f)

    f.close()