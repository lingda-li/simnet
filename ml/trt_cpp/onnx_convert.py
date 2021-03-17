import torch
import sys
import torch.onnx
from models import *

torch.set_default_dtype(torch.float32)
loaded_model_name = sys.argv[1]
simnet = torch.load(loaded_model_name, map_location='cuda')
batch_size= int(sys.argv[2])
#model= CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
model=simnet.eval()
inp= torch.ones(batch_size,5661).cuda()

with torch.no_grad():
    sa=torch.onnx.export(model, inp, "onnx_models/simnet_cnn7_com_" + sys.argv[2] + ".onnx", verbose=True, export_params=True,
        input_names= ['input'],
        output_names=['output']
       )
out= model(inp)
print(out)
import ipdb; ipdb.set_trace()
