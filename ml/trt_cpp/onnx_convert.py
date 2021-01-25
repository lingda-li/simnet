import torch
import torch.onnx
from models import *

torch.set_default_dtype(torch.float32)
batch_size= 32768
model= CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
model=model.eval()
inp= torch.rand(batch_size,5661).cuda()
with torch.no_grad():
    sa=torch.onnx.export(model, inp, "onnx_models/simnet_untrained_32k_n_tp.onnx", verbose=True, export_params=True,
        input_names= ['input'],
        output_names=['output']
       )

#import ipdb; ipdb.set_trace()
