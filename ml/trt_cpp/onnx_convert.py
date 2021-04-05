import torch
import sys
import torch.onnx
from models import *
context=111
instr=50
torch.set_default_dtype(torch.float32)
loaded_model_name = sys.argv[1]
model= CNN7_F(33,2,256,2,1,2,512,2,0,2,512,2,0,2,1024,2,0,2,1024,2,1,2,2048,2,0,2,2048,2,0,1024)
print(model.conv1.weight[0][0])
simnet = torch.load(loaded_model_name, map_location='cuda')
model.load_state_dict(simnet['model_state_dict'])
batch_size= int(sys.argv[2])
model=model.cuda().eval()
print(model.conv1.weight[0][0])
inp= torch.ones(batch_size,context*instr).cuda()
#import ipdb; ipdb.set_trace()
with torch.no_grad():
    sa=torch.onnx.export(model, inp, "onnx_models/sim_qq_cnn7_com_" + sys.argv[2] + ".onnx", verbose=True, export_params=True,
        input_names= ['input'],
        output_names=['output']
       )
out= model(inp)
print(out)
import ipdb; ipdb.set_trace()
