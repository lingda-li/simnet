import os
import sys
from models import *
from enet import E1DNet
context= 111
instr= 50
directory= '/home/spandey/final_models/'
#batches= [256,512,1024,2048,4096,8192,16384,32768]
batches= [32768]
#a = CNN_F(3,2,64,2,1,1024)
b = CNN3_F(3,2,128,2,1,2,256,2,0,2,256,2,0,1024)
b_d = CNN3_F_D(3,2,128,2,1,2,256,2,0,2,256,2,0,1024)
c = CNN5_F(3,2,192,2,1,2,384,2,0,2,384,2,0,2,768,2,0,2,768,2,1,1024)
d = CNN7_F(3,2,256,2,1,2,512,2,0,2,512,2,0,2,1024,2,0,2,1024,2,1,2,2048,2,0,2,2048,2,0,1024)
e = CNN7_F(33,2,256,2,1,2,512,2,0,2,512,2,0,2,1024,2,0,2,1024,2,1,2,2048,2,0,2,2048,2,0,1024)
f = E1DNet.from_input('e1d-b0',bargs=['r1_k2_s2_e2_i512_o128_se0.25','r1_k2_s2_e2_i128_o196_se0.25','r1_k2_s2_e2_i196_o256_se0.25','r1_k2_s2_e2_i256_o384_se0.25','r1_k2_s2_e2_i384_o512_se0.25','r1_k2_s2_e2_i512_o768_se0.25','r1_k2_s2_e2_i768_o1024_se0.25'],num_classes=33)


model_list= {'CNN3': b,
               'CNN5': c,
               'CNN7': d, 
               'CNN7COM': e,
               'E1DNet': f
               }

onnx_location= 'new_onnx_models/'
trt_location= '/home/spandey/new_tensorrt_models/'
for filename in os.listdir(directory):
    f= os.path.join(directory,filename)
    if os.path.isfile(f):
        loaded_model_name = filename
        name=loaded_model_name.split("_")
        model= model_list[name[0]]
        print(f)
        #model= CNN7_F(33,2,256,2,1,2,512,2,0,2,512,2,0,2,1024,2,0,2,1024,2,1,2,2048,2,0,2,2048,2,0,1024)
        simnet = torch.load(f, map_location='cuda')
        #model.load_state_dict(simnet['model_state_dict'])
        model.load_state_dict(simnet)
        model=model.cuda().eval()
        if not (os.path.exists(onnx_location + name[0])):
            os.makedirs(onnx_location + name[0])
        #if not (os.path.exists(trt_location + name[0])):
            #os.makedirs(trt_location + name[0])
        for batch_size in batches:
            inp= torch.ones(batch_size,context*instr).cuda()
            #inp= torch.ones(batch_size,7168).cuda()
            #import ipdb; ipdb.set_trace()
            save_name= name[0]+ "_" + str(batch_size)
            onnx_save_name= onnx_location + name[0] +'/'+ save_name + "_no_T.onnx"
            print(onnx_save_name)
            with torch.no_grad():
                sa=torch.onnx.export(model, inp, onnx_save_name, verbose=False, export_params=True, input_names= ['input'],output_names=['output']) 
            #import ipdb; ipdb.set_trace()
            # trt loader
            executable= './build/trt_static'
            trt_save_name= trt_location + "_T.engine"
             #./build/trt_static onnx_models/sim_qq_cnn7_com_4096.onnx 4096 tensorrt_models/test_matra 1
            os.system(executable +' '+onnx_save_name+' ' +str(batch_size) + ' '+trt_location + save_name + '_no_T.engine 0' )
            os.system(executable +' '+onnx_save_name+' ' +str(batch_size) + ' '+trt_location + save_name + '_no_T_half.engine 1' )
    
