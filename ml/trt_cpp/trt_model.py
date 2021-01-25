import torch
import time
#import torch2trt
from torch2trt import torch2trt
from models import *
from tensorrt import *

# create some regular pytorch model...
model = CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
model= model.eval()
batchsize= 32768
# create example data
x = torch.ones((batchsize, 5661), device=torch.device('cuda:0'))

# convert to TensorRT feeding sample data as input
#model_trt = torch2trt(model, [x], max_batch_size=65536)
'''
for i in range(5):
    x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
    st_time2= time.time()
    y=model(x)
    #y_trt= model_trt(x)
    en_time2= time.time()
    print(" Torch: %f"%(en_time2-st_time2))
    #print("TRT: %f"%(en_time1-st_time1))
    #import ipdb; ipdb.set_trace()
'''
model_trt = torch2trt(model, [x], max_batch_size=32768,fp16_mode=True,max_workspace_size=1<<30)
#import ipdb; ipdb.set_trace()
torch.cuda.empty_cache()
print("Full precision")
for i in range(5):
    x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
    st_time1= time.time()
    y_trt= model_trt(x)
    #y= model(x)
    en_time1= time.time()
    print(" TRT: %f"%(en_time1-st_time1))
#print(torch.max(torch.abs(y - y_trt)))

import ipdb; ipdb.set_trace()

"Serialize the tensorRT model"
serialized_engine= model_trt.serialize()
with trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(serialized_engine)
with open("siment_trt_engine", "wb") as f:
    f.write(engine.serialize())

"Read serilized engine"
with open("sample.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())



exit(0)


'''
print("FP16 mode")
model_trt = torch2trt(model, [x], max_batch_size=32768, fp16_mode=True,max_workspace_size=1<<30)
torch.cuda.empty_cache()
for i in range(5):
    x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
    st_time1= time.time()
    y_trt= model_trt(x)
    #y= model(x)
    en_time1= time.time()
    print(" TRT: %f"%(en_time1-st_time1))
print(torch.max(torch.abs(y - y_trt)))

model_trt = torch2trt(model, [x], max_batch_size=32768, int8_mode=True,max_workspace_size=1<<30)
torch.cuda.empty_cache()
print("Int8 mode")
for i in range(5):
    x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
    st_time1= time.time()
    y_trt= model_trt(x)
    #y= model(x)
    en_time1= time.time()
    print(" TRT: %f"%(en_time1-st_time1))
print(torch.max(torch.abs(y - y_trt)))
'''
