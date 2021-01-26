import torch
import time
import numpy as np
#import torch2trt
from torch2trt import torch2trt
from models import *
from tensorrt import *
import tensorrt as trt
import pycuda.driver as cuda 

# create some regular pytorch model...
model = CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
model= model.eval()
#batch= np.power(range(15),2)
batch= [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16382,32768,65536]
for j in range(len(batch)):    
    batchsize= batch[j]
# create example data
    x = torch.ones((batchsize, 5661), device=torch.device('cuda:0'))
    #model_trt = torch2trt(model, [x], max_batch_size= batchsize)
#import ipdb; ipdb.set_trace()
    torch.cuda.empty_cache()
    print("Full precision")
    for i in range(2):
        x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
        st_time1= time.time()
        #y_trt= model_trt(x)
        torch.cuda.synchronize()
        #y= model(x)
        en_time1= time.time()
        print(" Batch: %d, TRT: %f, per inst:%f "%(batch[j], en_time1-st_time1, (en_time1-st_time1)*1000000/batch[j]))
#print(torch.max(torch.abs(y - y_trt)))

#import ipdb; ipdb.set_trace()
torch.cuda.empty_cache()

#model = CNN3(2 ,5, 64, 5, 64, 5, 256, 400).cuda()
#model= model.eval()
#batch= np.power(range(15),2)
batch= [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16382,32768,65536]
for j in range(len(batch)):
    batchsize= batch[j]
# create example data
    x = torch.ones((batchsize, 5661), device=torch.device('cuda:0'))
#import ipdb; ipdb.set_trace()
    torch.cuda.empty_cache()
    print("Full precision")
    for i in range(2):
        x = torch.rand((batchsize, 5661), device=torch.device('cuda:0'))
        st_time1= time.time()
        y_trt= model(x)
        torch.cuda.synchronize()
        #y= model(x)
        en_time1= time.time()
        print(" Batch: %d, Pytorch: %f, per inst:%f "%(batch[j], en_time1-st_time1, (en_time1-st_time1)*1000000/batch[j]))
#print(torch.max(torch.abs(y - y_trt)))






'''
TRT_LOGGER= trt.Logger()
with trt.Builder(TRT_LOGGER) as builder:
    engine = builder.build_cuda_engine(model_trt)
    with open('engine.trt', 'wb') as f:
        f.write(bytearray(engine.serialize()))
'''

with open("simnet.engine", "wb") as f:
    f.write(model_trt.engine.serialize())


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
