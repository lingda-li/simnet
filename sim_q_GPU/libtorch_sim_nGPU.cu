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
//#include "trt.cuh"
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
preprocess(ROB *rob_d, Inst *insts, float *factor, float *mean, float *default_val, float *inputPtr, int *status, int Total_Trace)
{
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int warpID = TID / WARPSIZE;
  int warpTID = TID % WARPSIZE;
  int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
  int index, Total;
  ROB *rob;
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
#ifdef DEBUG
    if(threadIdx.x==0){ printf("Before memcpy: Head: %d,Tail: %d, len: %d,\n", rob->head,rob->tail, rob->len);}
#endif
    Tick curTick = rob->curTick;
    Tick lastFetchTick = rob->lastFetchTick;
    input_Ptr= inputPtr + ML_SIZE * index;  
    //int old_head= rob->head;

if (warpTID == 0)
    {
      if (status[index]==1)
      {      
              int retired = rob->retire_until(curTick); 
#ifdef DEBUG
              printf("Retire until: %ld, Retired: %d\n",curTick, retired);
#endif
      }
    }
	__syncwarp();
    if(warpTID==0){
	     Inst *newInst = rob->add();
	    //printf("Rob pointer before: %p, new Inst: %p, head: %d\n",rob,newInst,rob->dec(rob->tail));
	    memcpy(newInst, &insts[index], sizeof(Inst));
    	    //inst_copy(&rob->insts[rob->tail],&insts[index]);  
#ifdef DEBUG
	    printf("Rob pointer after: %p, new Inst: %p, head: %d\n",rob,newInst,rob->dec(rob->tail));
#endif
    }
    __syncwarp();
    //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    if (curTick != lastFetchTick)
    {
      rob->update_fetch_cycle(curTick - lastFetchTick, factor);
    } 
    __syncwarp();
    rob->make_input_data(input_Ptr, curTick, factor, default_val);
#ifdef DEBUG
    if (warpTID == 0)
    {
      printf("Input_Ptr\n");
      dis(input_Ptr, TD_SIZE, 6);
    }
#endif
    __syncwarp();
    index += Total;
  }
}


void display(float *data, int size, int rows)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%.2f\t", data[i * size + j]);
    }
    printf("\n");
  }
}

void display(unsigned long *data, int size, int rows)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%.f\t", (float)data[i * size + j]);
    }
    printf("\n");
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


int main(int argc, char *argv[])
{
  printf("args count: %d\n", argc);
#ifdef CLASSIFY
  if (argc != 9)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst> <nGPUs> <ROB_per_GPU>" << endl;
    return 0;
  }
#else
  if (argc != 8)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <variances> <#Insts> <ROB_per_GPU> <nGPUs>" << endl;
#endif
  return 0;
}
int arg_idx = 4;
float *varPtr = read_numbers(argv[4], TD_SIZE);
for (int i = 0; i < TD_SIZE; i++)
{
#ifdef NO_MEAN
  mean[i] = -0.0;
#endif
  factor[i] = sqrtf(varPtr[i]);
  default_val[i] = -mean[i] / factor[i];
  //cout<<default_val[i]<<" ";
}
  const int nGPUs= atoi(argv[7]);
  const int ROB_per_GPU= atoi(argv[6]);
  int count[2];
  H_ERR(cudaGetDeviceCount(count));
  if (count[0] < nGPUs) {
   cerr << "GPUs not enough" << endl;
   return 0;
  }
printf("GPU loaded\n");
at::Tensor input[nGPUs];
torch::jit::script::Module lat_module[nGPUs];
#pragma omp parallel for
    for (int i = 0; i < nGPUs; i++) {
      try{
      lat_module[i] = torch::jit::load(argv[3]);
      input[i] = torch::ones({ROB_per_GPU, ML_SIZE});
      string dev = "cuda:";
      string id = to_string(i);
      dev = dev + id;
      const std::string device_string = dev;
      lat_module[i].to(device_string);
    }
   catch (const c10::Error &e) {
    cerr << "error loading the model\n";
    //return 0;
  }
}


//cout<<endl;
//int Total_Trace = atoi(argv[arg_idx++]);
const int Total_Trace= nGPUs * ROB_per_GPU;
const int Instructions = atoi(argv[5]);
cout<< "Total_Trace: "<< Total_Trace << ", Instructions: "<< Instructions << endl;
std::string model_path(argv[3]);
//at::Tensor input = torch::ones({Total_Trace, ML_SIZE});
//cout<<"Input dims: "<< input_dims << ", output dims: "<<output_dims << endl;
//cout<< "Input dim: "<< ML_SIZE * Total_Trace << endl;
float *trace;
Tick *aux_trace;
trace = (float *)malloc(TRACE_DIM * Instructions * sizeof(float));
aux_trace = (Tick *)malloc(AUX_TRACE_DIM * Instructions * sizeof(Tick));
read_trace_mem(argv[1], argv[2], trace, aux_trace, Instructions);
int Batch_size = Instructions / Total_Trace;
cout << " Iterations: " << Batch_size << endl;
//cout<<"Parameters read..\n";
omp_set_num_threads(96);
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
printf("variable init\n");
#pragma omp parallel for
for (int i = 0; i < Total_Trace; i++)
{
  int offset = i * Batch_size;
  trace_all[i] = trace + offset * TRACE_DIM;
  aux_trace_all[i] = aux_trace + offset * AUX_TRACE_DIM;
}

int *status[nGPUs];
//struct ROB *rob[nGPUs];
struct Inst *inst[nGPUs];
struct ROB *rob_d[nGPUs];
struct Inst *inst_d[nGPUs];
float *factor_d[nGPUs], *default_val_d[nGPUs], *mean_d[nGPUs];
float *inputPtr[nGPUs], *output[nGPUs];

#pragma omp parallel for
    for (int i = 0; i < nGPUs; i++) {
      inst[i]= new Inst[ROB_per_GPU];
      H_ERR(cudaSetDevice(i));
      H_ERR(cudaMalloc((void **)&status[i], sizeof(int) * ROB_per_GPU));
      H_ERR(cudaMalloc((void **)&rob_d[i], sizeof(ROB)*ROB_per_GPU));
      H_ERR(cudaMalloc((void **)&inst_d[i], sizeof(Inst)*ROB_per_GPU));
      H_ERR(cudaMalloc((void **)&inputPtr[i], sizeof(float) * ML_SIZE * ROB_per_GPU));
      H_ERR(cudaMalloc((void **)&output[i], sizeof(float) * ROB_per_GPU * 22));
      // For factor, mean and default values
      H_ERR(cudaMalloc((void **)&factor_d[i], sizeof(float) * (TD_SIZE)));
      H_ERR(cudaMalloc((void **)&mean_d[i], sizeof(float) * (TD_SIZE)));
      H_ERR(cudaMalloc((void **)&default_val_d[i], sizeof(float) * (TD_SIZE)));
      H_ERR(cudaMemcpy(factor_d[i], &factor, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
      H_ERR(cudaMemcpy(default_val_d[i], &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
      H_ERR(cudaMemcpy(mean_d[i], &mean, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
    }

struct timeval total_start, total_end;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0,pre=0, tr=0,inf=0,upd=0;
FILE *pFile;
pFile= fopen ("libcustom.bin", "wb");
while (iteration < Batch_size){
  if((iteration % 1)==0){cout << "Iteration: " << iteration << endl;}
  double st = wtime();
#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    int GPU_ID= i/ROB_per_GPU;
    int index= i % ROB_per_GPU; 
    if (!inst[GPU_ID][index].read_sim_mem(trace_all[i], aux_trace_all[i],i))
    {cout << "Error\n";}
    trace_all[i] += TRACE_DIM; aux_trace_all[i] += AUX_TRACE_DIM;
    //printf("Trace: %d, read\n",i);
    } 
  double check1 = wtime();
  red+= (check1-st);
  //cout<< "Instructions read\n";
  #pragma omp parallel for
    for (int i = 0; i < nGPUs; i++) {
    //cout<< "GPU: "<<i<< " "<<wtime()<<endl;
	    H_ERR(cudaSetDevice(i));
    //cout<<"set device: \n";
    H_ERR(cudaMemcpy(inst_d[i], inst[i], sizeof(Inst) * ROB_per_GPU, cudaMemcpyHostToDevice));
    double check2 = wtime();
    //cout << "copied\n";
    tr+= (check2-check1);
    float *inp= input[i].data_ptr<float>();
    preprocess<<<4096, 64>>>(rob_d[i],inst_d[i], factor_d[i], mean_d[i], default_val_d[i], inputPtr[i], status[i], ROB_per_GPU);
    H_ERR(cudaDeviceSynchronize());
    //cout << "Preprocess done\n";
    double check3= wtime();
      H_ERR(cudaMemcpy(inp,inputPtr[i], sizeof(float) * ML_SIZE*ROB_per_GPU, cudaMemcpyDeviceToHost));
    // fwrite(inp, sizeof(float), ML_SIZE, pFile);
    pre+= (check3-check2);
    check3 = wtime();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input[i].cuda());  
    at::Tensor outputs = lat_module[i].forward(inputs).toTensor();
    cudaStreamSynchronize(0);
    //cout<<outputs<<endl;
    double check4= wtime();
    inf+= (check4-check3);
    //cout<<"Inference done\n";
    H_ERR(cudaMemcpy(output[i], outputs.data_ptr<float>(), sizeof(float) * ROB_per_GPU*22, cudaMemcpyHostToDevice));
    update<<<4096,64>>>(rob_d[i], output[i], factor_d[i], mean_d[i], status[i], ROB_per_GPU);
    H_ERR(cudaDeviceSynchronize());
  }
  //cout<<"Update done\n";
  double check5=wtime();
  //upd+=(check5-check4);
  iteration++;
}
fclose(pFile);
printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
double end_ = wtime();

gettimeofday(&total_end, NULL);
for(int i=0; i<nGPUs; i++){
	 H_ERR(cudaSetDevice(i));
	result<<<1, 1>>>(rob_d[i], ROB_per_GPU, Instructions);
	H_ERR(cudaDeviceSynchronize());
}
double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
//cout << "Total time: " << (end_ - start_) << endl;
#ifdef RUN_TRUTH
cout << "Truth"
     << "\n";
#endif
     //cout << Instructions << " instructions finish by " << (curTick - 1) << "\n";
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
