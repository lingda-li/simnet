# simnet
This folder contains the optimized version of SimNet. To run the simulation, we first need trace files from the benchmarks. Directions to generate the trace file can be found at the simnet root folder. After we have the trace, we need to generate the TensorRT inference engine for the trained models. trt_cpp/ folder contains the script required for generating the TensorRT model. 

## Generating the TensorRT engine
TensorRT models can be generated using the `convert_script.py` script. Basically, the script converts all of the trained libtorch model from SimNet (present in folder final_models/) to ONNX model and to TensorRT models (new_tensorrt_models/). `code_static.cpp` is the actual scripts which converts individual libtorch model to TensorRT model. Before running `convert_script.py`, execute `run_code.sh` to build executables for `code_static`. `models.py` describes the model used by SimNet. The configuration of TensorRT inference engine (half precision, 2:4 pruning) can be changed from the same `code_static.cpp` file. Note, the batchsize of TensorRT model should match the batchsize of simulator.   

## Compiling the simulator
The simulator can be build by running `make` in the sim_qq_GPU which builds both CPU and GPU version of the simulator in the build folder. `trt_n_simulator` is the executable which includes all the optimizations and used for experiments. 


## Running the simulator
The arguments to run the simulator are listed below: 

`
./build/trt_n_simulator <trace> <aux_trace> <inference_engine> <Total instructions to simulation from trace> <Batchsize per GPU> <number of GPUs> <Warmup instruction>
`

Here, 
both <trace> and <aux trace> represent the benchmarks pre-processed data in binary format. 
The repo includes 557.xz_r.qq100m.tr.bin and 557.xz_r.qq100m.tra.bin which consists of 1000 instructions as an example. 
<Warmup instructions> specifies how many instructions to simulate for warmup from each partition. 

For e.g., 
./build/trt_n_simulator 557.xz_r.qq100m.tr.bin 557.xz_r.qq100m.tra.bin ~/new_tensorrt_models/CNN3_32768_half.engine 1000 4 1 5 
This will run simulation on 557.xz_r benchmark with a CNN3 model having a batchsize of 4 on a single GPU. Here, The warmup length is 5. 
