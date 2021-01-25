import onnx
import tensorrt as trt
import onnx_tensorrt.backend as backend
import numpy as np
import sys
import time
import os
from models import *
import pycuda.driver as cuda
import common
"Python loader"

TRT_LOGGER= trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    if os.path.exists(engine_file_path):
        print("Reading from a file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size=1
            builder.max_workspace_size= 1 << 28
            if not os.path.exists(onnx_file_path):
                print("ONNX not found")
                exit(0)
            with open(onnx_file_path, "rb") as model:
                print("parsing onnx file")
                parser.parse(model.read())
            print("Parsing completed. Building engine now...")
            engine= builder.build_cuda_engine(network)
            print("Completed creating engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
 
onnx_path= sys.argv[1]
engine_file_path= sys.argv[2]
model= onnx.load(onnx_path)
batchsize= int(sys.argv[3])
input_data = np.full((batchsize, 5661),1).astype(np.float32)
input_resolution= (batchsize, 5661)
output_resolution= (batchsize,2)

with get_engine(onnx_path, engine_file_path) as engine, engine.create_execution_context() as context:
    #import ipdb; ipdb.set_trace()
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_input= input_data.copy()
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    print("Dim 0: {}, Dim 1: {}".format(context.get_binding_shape(0),context.get_binding_shape(1)))
# Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes) 
    stream= cuda.Stream()
    for i in range(5):
        st= time.time()
        cuda.memcpy_htod(d_input, h_input)
        en= time.time()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
        en2= time.time()
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        print("Mem cpy: {}, Compute: {}, Total: {}".format(en-st,en2-en ,en2-st))
print(h_output.shape)
print(h_output)

"Library loader"
engine = backend.prepare(model, device='CUDA:0')
#import ipdb; ipdb.set_trace()
#input_data = np.random.random(size=(batchsize, 5661)).astype(np.float32)


for i in range(5):    
    st= time.time()
    output_data = engine.run(input_data)
    en= time.time()
    print(en-st, (en-st)/batchsize)
print("Lib out")
print(output_data)
#print(output_data.shape)
h_output= h_output.reshape(batchsize,2)
print("Py outout")
print(h_output)
print("difference")
print(output_data-h_output)
import ipdb; ipdb.set_trace()
