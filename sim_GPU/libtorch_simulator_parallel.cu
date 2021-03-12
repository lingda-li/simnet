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



 __device__ void
  dis(float *data, int size, int rows)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < size; j++)
      {
        printf("%.3f  ", data[i * size + j]);
      }
      printf("\n");
    }
  }


__device__ void inst_copy(Inst *dest, Inst *source)
{
	dest->trueFetchClass= source->trueFetchClass;
	dest->trueFetchTick= source->trueFetchTick;
	dest->trueCompleteClass= source->trueCompleteClass;
	dest->trueCompleteTick= source->trueCompleteTick;
	dest->pc= source->pc;
	dest->isAddr= source->isAddr;
	dest->addr= source->addr;
	dest->addrEnd= source->addrEnd;
	for(int i=0;i<3; i++){
		dest->iwalkAddr[i]= source->iwalkAddr[i];
		dest->dwalkAddr[i]= source->dwalkAddr[i];
	}
	for(int i=0;i<TD_SIZE;i++){
		dest->train_data[i]= source->train_data[i];
	}
}

__global__ void
preprocess(ROB *rob_d, Inst *insts, float *factor, float *mean, float *default_val, float *inputPtr, Tick *curTick_d, Tick *lastFetchTick_d, int *status, int Total_Trace)
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
    Tick curTick = curTick_d[index];
    Tick lastFetchTick = lastFetchTick_d[index];
    input_Ptr= inputPtr + ML_SIZE * index;  
    int old_head= rob->head;

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
      rob->update_fetch_cycle(curTick - lastFetchTick, curTick, factor);
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

__device__ Tick
max(float x, Tick y)
{
  if (x > y)
  {
    return x;
  }
  else
  {
    return y;
  }
}

__global__ void
result(Tick *curTick, int Total_Trace, int instructions)
{
  Tick sum = 0;
  for (int i = 0; i < Total_Trace; i++)
  {
    sum += curTick[i];
  }
  printf("~~~~~~~~~%d instructions finish by %ld ~~~~~~~~~\n", instructions, sum);
}

__global__ void
update(ROB *rob_d, float *output, float *factor, float *mean, Tick *curTick, Tick *lastFetchTick, int *status, int Total_Trace)
{
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int index = TID;
  ROB *rob;
  while (index < Total_Trace)
  {
    int offset = index * 2;
    Tick nextFetchTick = 0;
    rob= &rob_d[index];
    int tail= rob->dec(rob->tail);
    int rob_offset = ROBSIZE * INST_SIZE * index;
    int context_offset = rob->dec(rob->tail) * INST_SIZE;
    //float *rob_pointer = insts + rob_offset + context_offset;
    //printf("%ld, %.3f, %.3f\n ",curTick[index],output[offset+0],output[offset+1]);
    float fetch_lat = output[offset + 0] * factor[1] + mean[1];
    float finish_lat = output[offset + 1] * factor[3] + mean[3];
    int int_fetch_lat = round(fetch_lat);
    int int_finish_lat = round(finish_lat);
    #ifdef DEBUG
    printf("%ld, %.3f, %d, %.3f, %d\n",curTick[index], output[offset+0], int_fetch_lat, output[offset+1], int_finish_lat);
    #endif
    if (int_fetch_lat < 0)
    {int_fetch_lat = 0;}
    if (int_finish_lat < MIN_COMP_LAT)
      int_finish_lat = MIN_COMP_LAT;
    rob->insts[tail].train_data[0] = (-int_fetch_lat - mean[0]) / factor[0];
    rob->insts[tail].train_data[1] = (-int_fetch_lat - mean[1]) / factor[1];
    rob->insts[tail].train_data[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2]; 
    if (rob->insts[tail].train_data[2] >= 9 / factor[2])
    {
      rob->insts[tail].train_data[2] = 9 / factor[2];
    }
    rob->insts[tail].train_data[3] = (int_finish_lat - mean[3]) / factor[3];
#ifdef DEBUG
    printf("Index: %d, offset: %d, Fetch: %.4f, Finish: %.4f, Rob0: %.2f, Rob1: %.2f, Rob2: %.2f, Rob3: %.2f\n", index, rob->tail, output[offset + 0], output[offset + 1], rob_pointer[0], rob_pointer[1], rob_pointer[2], rob_pointer[3]);
#endif
    rob->insts[tail].tickNum = int_finish_lat;
    rob->insts[tail].completeTick = curTick[index] + int_finish_lat + int_fetch_lat;    
    lastFetchTick[index] = curTick[index];
    if (int_fetch_lat)
    {
	    status[index]=1;
      nextFetchTick = curTick[index] + int_fetch_lat;
    	//printf("Break with int fetch\n");
    }
    else{status[index]=0; index += (gridDim.x * blockDim.x);continue;}
    if ((rob->is_full() || rob->saturated) && int_fetch_lat)
    {
      curTick[index] = max(rob[tail].getHeadTick(), nextFetchTick);
      //printf("getting max\n");
    }
    else if (int_fetch_lat)
    {
      curTick[index] = nextFetchTick;
      //printf("fastforward ot fetch\n");
    }
    else if (rob->saturated || rob->is_full())
    {
      curTick[index] = rob[tail].getHead()->completeTick;
      //printf("fastforward to retire\n");
    }
    //printf("Head: %d, Len at the end: %d\n",rob->head,rob->len);
    //printf("curTick: %ld, completeTick: %.2f, nextfetchTick: %ld, lastFetchTick: %ld \n",curTick[index],rob_pointer[rob_offset+COMPLETETICK],nextFetchTick,lastFetchTick[index]);
    index += (gridDim.x * blockDim.x);
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

int read_trace_mem(char trace_file[], char aux_trace_file[], float *trace, Tick *aux_trace, int instructions)
{
  FILE *trace_f = fopen(trace_file, "rb");
  if (!trace_f)
  {
    printf("Unable to read trace binary.");
    return 1;
  }
  int r = fread(trace, sizeof(float), TRACE_DIM * instructions, trace_f);
  printf("read :%d values for trace.\n", r);
  //display(trace,TRACE_DIM,2);

  FILE *aux_trace_f = fopen(aux_trace_file, "rb");
  if (!aux_trace_f)
  {
    printf("Unable to aux_trace binary.");
    return 1;
  }
  int k = fread(aux_trace, sizeof(Tick), AUX_TRACE_DIM * instructions, aux_trace_f);
  printf("read :%d values for aux_trace.\n", k);
  //display(aux_trace,AUX_TRACE_DIM,2);
  return true;
}

int main(int argc, char *argv[])
{
  printf("args count: %d\n", argc);
#ifdef CLASSIFY
  if (argc != 8)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst> <Total trace>" << endl;
    return 0;
  }
#else
  if (argc != 7)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <variances> <Total trace> <#Insts>" << endl;
#endif
  return 0;
}
int arg_idx = 4;
float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);
for (int i = 0; i < TD_SIZE; i++)
{
#ifdef NO_MEAN
  mean[i] = -0.0;
#endif
  factor[i] = sqrtf(varPtr[i]);
  default_val[i] = -mean[i] / factor[i];
  //cout<<default_val[i]<<" ";
}

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

//cout<<endl;
int Total_Trace = atoi(argv[arg_idx++]);
int Instructions = atoi(argv[arg_idx++]);
std::string model_path(argv[3]);
at::Tensor input = torch::ones({Total_Trace, ML_SIZE});
float *inp= input.data_ptr<float>();
//cout<<"Input dims: "<< input_dims << ", output dims: "<<output_dims << endl;

float *inputPtr, *output;
H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float) * ML_SIZE * Total_Trace));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * Total_Trace * 2));
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
float *factor_d, *default_val_d, *mean_d;
float *train_data;
Tick *curTick, *lastFetchTick;
int *status;
H_ERR(cudaMalloc((void **)&curTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&lastFetchTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&status, sizeof(int) * Total_Trace));
cudaMemset(curTick, 0, Total_Trace);
cudaMemset(lastFetchTick, 0, Total_Trace);
//cudaHostAlloc((void **)&train_data, Total_Trace *INST_SIZE * sizeof(float),
  //            cudaHostAllocDefault);
struct ROB *rob= new ROB[Total_Trace];
struct Inst *inst= new Inst[Total_Trace];
struct ROB *rob_d;
struct Inst *inst_d;
H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB)*Total_Trace));
H_ERR(cudaMalloc((void **)&inst_d, sizeof(Inst)*Total_Trace));
// For factor, mean and default values
H_ERR(cudaMalloc((void **)&factor_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&mean_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMemcpy(factor_d, &factor, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
struct timeval start, end, total_start, total_end;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0,pre=0, tr=0,inf=0,upd=0;
FILE *pFile;
pFile= fopen ("libcustom.bin", "wb");
while (iteration < Batch_size){
  if((iteration % 50)==0){cout << "Iteration: " << iteration << endl;}
  double st = wtime();
#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    if (!inst[i].read_sim_mem(trace_all[i], aux_trace_all[i],i))
    {cout << "Error\n";}
    trace_all[i] += TRACE_DIM; aux_trace_all[i] += AUX_TRACE_DIM;
    } 
  double check1 = wtime();
  red+= (check1-st);
    H_ERR(cudaMemcpy(inst_d, inst, sizeof(Inst) * Total_Trace, cudaMemcpyHostToDevice));
    double check2 = wtime();
  tr+= (check2-check1);
  preprocess<<<4096, 64>>>(rob_d,inst_d, factor_d, mean_d, default_val_d, inputPtr, curTick, lastFetchTick, status, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(inp,inputPtr, sizeof(float) * ML_SIZE*Total_Trace, cudaMemcpyDeviceToHost));
  fwrite(inp, sizeof(float), ML_SIZE, pFile);
  //printf("Input:\n");
  //display(inp, 51,4);
  double check3 = wtime();
  pre+= (check3-check2);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input.cuda());  
  at::Tensor outputs = lat_module.forward(inputs).toTensor();
  cudaStreamSynchronize(0);
  double check4= wtime();
  inf+= (check4-check3);
  //cout<<"Inference done\n";
  H_ERR(cudaMemcpy(output, outputs.data_ptr<float>(), sizeof(float) * Total_Trace*2, cudaMemcpyHostToDevice));
  update<<<4096,32>>>(rob_d, output, factor_d, mean_d, curTick, lastFetchTick, status, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  double check5=wtime();
  upd+=(check5-check4);
  iteration++;
}
fclose(pFile);
printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
double end_ = wtime();

gettimeofday(&total_end, NULL);
result<<<1, 1>>>(curTick, Total_Trace, Instructions);
H_ERR(cudaDeviceSynchronize());
double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
cout << "Total time: " << (end_ - start_) << endl;
#ifdef RUN_TRUTH
cout << "Truth"
     << "\n";
#endif
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
