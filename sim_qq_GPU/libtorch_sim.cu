#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
#include "wtime.h"
#include "herror.h"
#include "sim.cuh"
#include<torch/script.h>
using namespace std;
#define NO_MEAN
#define GPU
#define WARP
//#define DEBUG

//#define Total_Trace 1024

Tick Num = 0;



__global__ void
preprocess(ROB *rob_d,SQ *sq_d, Inst *insts, float *default_val, float *inputPtr, int *status, int Total_Trace)
{
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int warpID = TID / WARPSIZE;
  int warpTID = TID % WARPSIZE;
  //int W= threadIdx.x/WARPSIZE;
  int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
  int index, Total;
  ROB *rob;
  SQ *sq;
  float *input_Ptr;
#ifdef WARP
  index = warpID;
  Total = TotalWarp;
#else
  index = blockIdx.x;
  Total = gridDim.x;
#endif
  while (index < Total_Trace)
  {
    rob= &rob_d[index];
    sq= &sq_d[index];
    Inst *newInst;
    //Inst __shared__ *temp[4];
    Tick curTick = rob->curTick;
    Tick lastFetchTick = rob->lastFetchTick;
    input_Ptr= inputPtr + ML_SIZE * index;  
    //int old_head= rob->head;

if (warpTID == 0){  
      //printf("\n\n");
      if (status[index]==1)
      {
	      //int sq_retired = sq->retire_until(curTick);      
              //int retired = rob->retire_until(curTick, sq); 
	      //printf("SQ retired: %d, ROB Retired: %d\n", sq_retired, retired);
              //printf("Retire until: %ld, Retired: %d\n",curTick, retired);
      }
	 newInst = rob->add();
	 memcpy(newInst, &insts[index], sizeof(Inst)); 
	 printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    }
    __syncwarp();
    //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    if (curTick != lastFetchTick)
    {
     if(warpTID==0){printf("ROB update\n");}
     __syncwarp();
      rob->update_fetch_cycle(curTick - lastFetchTick);
      __syncwarp();
     if(warpTID==0){printf("SQ update\n");}
     __syncwarp();
      sq->update_fetch_cycle(curTick - lastFetchTick);  
      __syncwarp();
    } 
    __syncwarp();
    if(warpTID==0){printf("ROB: head: %d, tail: %d \n", rob->head, rob->tail);}
     if(warpTID==0){printf("SQ: head: %d, tail: %d \n", sq->head, sq->tail);}
     __syncwarp();
    int rob_num= rob->make_input_data(input_Ptr, curTick, insts[index]);
    __syncwarp();
    int sq_num= sq->make_input_data(input_Ptr + rob_num * TD_SIZE, curTick, insts[index]);
    __syncwarp();
    int num= rob_num + sq_num;
    // copy default values
    if(num < CONTEXTSIZE && warpTID==0)
    {
       memcpy(input_Ptr+num*TD_SIZE, default_val +num*TD_SIZE, sizeof(float)*(CONTEXTSIZE-num)*TD_SIZE);
    }   
#ifdef DEBUG
    if (warpTID == 0)
    {
      //printf("Input_Ptr\n");
      //dis(input_Ptr, TD_SIZE, 6);
    }
#endif
    __syncwarp();
    index += Total;
  }
}



float *read_numbers(char *fname, int sz)
{
  float *ret = new float[sz];
  ifstream in(fname);
  //printf("Trying to read from %s\n", fname);
  for (int i = 0; i < sz; i++)
    in >> ret[i];
  return ret;
}


void display(float *data, int size, int rows)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%.2f\t", data[i * size + j]);
      /*
      if (data[i * size + j]!=0){
          printf("%.2f  ", data[i * size + j]);}
      else{printf("   ");}*/
    }
    printf("\n");
  }
}


int main(int argc, char *argv[])
{
  printf("args count: %d\n", argc);
#ifdef CLASSIFY
  if (argc != 7)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst> <Total trace>" << endl;
    return 0;
  }
#else
  if (argc != 6)
  {
    cerr << "Usage: ./simulator_qq <trace> <aux trace> <lat module> <Total trace> <#Insts>" << endl;
#endif
  return 0;
}
//int arg_idx = 4;
//float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);

for (int i = 0; i < TD_SIZE; i++) {
    default_val[i] = 0;
  }
  for (int i = TD_SIZE; i < ML_SIZE; i++)
    default_val[i] = default_val[i % TD_SIZE];


cout<< argv[3] << endl;
torch::jit::script::Module lat_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    lat_module = torch::jit::load(argv[3]);
#ifdef GPU
    lat_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return 0;
  }
//lat_module.save("libtorch.pt");
//return 0;
//cout<<endl;
int Total_Trace = atoi(argv[4]);
int Instructions = atoi(argv[5]);
cout<< "Total_Trace: "<< Total_Trace << ", Instructions: "<< Instructions << endl;
//std::string model_path(argv[3]);
at::Tensor input = torch::ones({Total_Trace, ML_SIZE});
float *inp= input.data_ptr<float>();
//cout<<"Input dims: "<< input_dims << ", output dims: "<<output_dims << endl;
float *inputPtr, *output;
H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float) * ML_SIZE * Total_Trace));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * Total_Trace * 33));
//cout<< "Input dim: "<< ML_SIZE * Total_Trace << endl;
float *trace;
Tick *aux_trace;
trace = (float *)malloc(TRACE_DIM * Instructions * sizeof(float));
aux_trace = (Tick *)malloc(AUX_TRACE_DIM * Instructions * sizeof(Tick));
read_trace_mem(argv[1], argv[2], trace, aux_trace, Instructions);
int Batch_size = Instructions / Total_Trace;
//cout << " Iterations: " << Batch_size << endl;
//cout<<"Parameters read..\n";
omp_set_num_threads(1);
double measured_time = 0.0;
Tick Case0 = 0;
Tick Case1 = 0;
Tick Case2 = 0;
Tick Case3 = 0;
Tick Case4 = 0;
Tick Case5 = 0;
int *fetched_inst_num = new int[Total_Trace];
int *fetched = new int[Total_Trace];
int *ROB_flag = new int[Total_Trace];
float *trace_all[Total_Trace];
Tick *aux_trace_all[Total_Trace];
//printf("variable init\n");
#pragma omp parallel for
for (int i = 0; i < Total_Trace; i++)
{
  int offset = i * Batch_size;
  trace_all[i] = trace + offset * TRACE_DIM;
  aux_trace_all[i] = aux_trace + offset * AUX_TRACE_DIM;
}
// printf("Allocated. \n");
//return 0;
float *default_val_d;
Tick *curTick, *lastFetchTick;
int *status;
H_ERR(cudaMalloc((void **)&curTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&lastFetchTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&status, sizeof(int) * Total_Trace));
cudaMemset(curTick, 0, Total_Trace);
cudaMemset(lastFetchTick, 0, Total_Trace);
cudaMemset(status, 1, Total_Trace);
struct SQ *sq= new SQ[Total_Trace];
struct ROB *rob= new ROB[Total_Trace];
struct Inst *inst= new Inst[Total_Trace];
struct ROB *rob_d; 
struct SQ *sq_d;
struct Inst *inst_d;
H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB)*Total_Trace));
H_ERR(cudaMalloc((void **)&sq_d, sizeof(SQ)*Total_Trace));
H_ERR(cudaMalloc((void **)&inst_d, sizeof(Inst)*Total_Trace));
// For factor, mean and default values
H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
struct timeval total_start, total_end;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0,pre=0, tr=0,inf=0,upd=0;
FILE *pFile;
pFile= fopen ("libcustom.bin", "wb");
while (iteration < Batch_size){
  //if((iteration % 500)==0)
  {cout << "\nIteration: " << iteration << endl;}
  double st = wtime();
#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    if (!inst[i].read_sim_mem(trace_all[i], aux_trace_all[i],i))
    {cout << "Error\n";}
    trace_all[i] += TRACE_DIM; aux_trace_all[i] += AUX_TRACE_DIM;
    //printf("Trace: %d, read\n",i);
    }
 //printf("Inst read\n"); 
  double check1 = wtime();
  red+= (check1-st);
    H_ERR(cudaMemcpy(inst_d, inst, sizeof(Inst) * Total_Trace, cudaMemcpyHostToDevice));
    double check2 = wtime();
  tr+= (check2-check1);
  //cout<<"Data transferred\n";
  preprocess<<<4096, 64>>>(rob_d,sq_d,inst_d, default_val_d, inputPtr, status, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  //cout<<"Preprocess done \n"<<endl; 
  double check3= wtime();
    H_ERR(cudaMemcpy(inp,inputPtr, sizeof(float) * ML_SIZE*Total_Trace, cudaMemcpyDeviceToHost));
  fwrite(inp, sizeof(float), ML_SIZE, pFile);
  //printf("Input:\n");
  //display(inp, 51,4);
  pre+= (check3-check2);
  check3 = wtime();
  //pre+= (check3-check2);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input.cuda());  
  at::Tensor outputs = lat_module.forward(inputs).toTensor();
  cudaStreamSynchronize(0);
  //cout<<outputs<<endl;
  double check4= wtime();
  inf+= (check4-check3);
  cout<<"Output size: "<< outputs.sizes()[0]<<endl;
  //cout<<"Inference done\n";
  int out_shape= outputs.sizes()[1];
  H_ERR(cudaMemcpy(output, outputs.data_ptr<float>(), sizeof(float) * Total_Trace*33, cudaMemcpyHostToDevice));
  update<<<4096,64>>>(rob_d,sq_d, output, status, Total_Trace, out_shape);
  H_ERR(cudaDeviceSynchronize());
  //cout<<"Update done\n";
  double check5=wtime();
  upd+=(check5-check4);
  iteration++;
}
fclose(pFile);
printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
double end_ = wtime();

gettimeofday(&total_end, NULL);
result<<<1, 1>>>(rob_d, Total_Trace, Instructions);
H_ERR(cudaDeviceSynchronize());
double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
//cout << "Total time: " << (end_ - start_) << endl;
#ifdef RUN_TRUTH
cout << "Truth"
     << "\n";
#endif
return 0;
     cout << Instructions << " instructions finish by " << (curTick - 1) << "\n";
cout << "Time: " << total_time << "\n";
cout << "MIPS: " << Instructions / total_time / 1000000.0 << "\n";
cout << "USPI: " << total_time * 1000000.0 / Instructions << "\n";
cout << "Measured Time: " << measured_time / Instructions << "\n";
cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  //cout << "Lat Model: " << argv[3] << "\n";
#endif
return 0;
}
