import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from itertools import product
import pickle
import glob
import os
from models import context_length, inst_length
np.random.seed(0)

if len(sys.argv) < 3:
    print("Usage: %s input_dir output_dir" % sys.argv[0])
    sys.exit()
    
input_dir = sys.argv[1]
output_dir = sys.argv[2]

files = glob.glob("%s/*.pt" % input_dir)
files.sort(key=os.path.getmtime)

for loaded_model_name in files:
    simnet = torch.load(loaded_model_name, map_location='cpu')
    simnet.eval()

    traced_script_module = torch.jit.trace(simnet, torch.rand(1, context_length * inst_length))
    output = traced_script_module(torch.ones(1, context_length * inst_length))
    print(type(simnet),output)
    traced_script_module.save("%s/%s.pt" % (output_dir,os.path.basename(loaded_model_name)))
