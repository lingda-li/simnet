import torch
import torch.onnx
from models import *

model= CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
inp= torch.rand(16384,5661).cuda()

torch.onnx.export(model, inp, "simnet_untrained.onnx", verbose=True, export_params=True)
