import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from itertools import product
import pickle
np.random.seed(0)
import glob
import os

#context_length = 96
#inst_length = 10
#context_length = 93
#inst_length = 17
context_length = 94
inst_length = 39

files = glob.glob("models/*.pt")
files.sort(key=os.path.getmtime)

for loaded_model_name in files:
    simnet = torch.load(loaded_model_name, map_location='cpu')
    simnet.eval()

    traced_script_module = torch.jit.trace(simnet, torch.rand(1, context_length * inst_length))
    output = traced_script_module(torch.ones(1, context_length * inst_length))
    print(type(simnet),output)
    traced_script_module.save("../data_spec/converted_models/%s.pt" % os.path.basename(loaded_model_name))
