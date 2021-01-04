import math
import numpy as np
import sys
import torch
import torch.nn as nn
from models import *

loaded_model_name = sys.argv[1]

simnet = torch.load(loaded_model_name, map_location='cpu')
simnet.eval()

traced_script_module = torch.jit.trace(simnet, torch.rand(1, context_length * inst_length))
output = traced_script_module(torch.ones(1, context_length * inst_length))
print(output)
traced_script_module.save(loaded_model_name + '.pt')
