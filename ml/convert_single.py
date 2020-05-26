import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import pickle
np.random.seed(0)
from models import *

#context_length = 96
#inst_length = 10
#context_length = 93
#inst_length = 17
context_length = 94
inst_length = 39

loaded_model_name = "specdc_cnn_3_latonly_l64_64_052120_cpu"

simnet = torch.load('models/' + loaded_model_name, map_location='cpu')
simnet.eval()

traced_script_module = torch.jit.trace(simnet, torch.rand(1, context_length * inst_length))
output = traced_script_module(torch.ones(1, context_length * inst_length))
print(output)
traced_script_module.save('models/' + loaded_model_name + '.pt')
