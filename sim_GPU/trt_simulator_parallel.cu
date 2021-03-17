#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
#include "wtime.h"
#include "herror.h"
#include "trt.cuh"
#include "sim.cuh"
using namespace std;
#define NO_MEAN
#define GPU
#define WARP
//#define DEBUG

//#define Total_Trace 1024

Tick Num = 0;



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
    //int old_head= rob->head;

if (warpTID == 0)
    {
      if (status[index]==1)
      {         //printf("Before: Head: %d, len: %d,\n", rob->head, rob->len);
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
//cout<<endl;
int Total_Trace = atoi(argv[arg_idx++]);
int Instructions = atoi(argv[arg_idx++]);
std::string model_path(argv[3]);
TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
deseralizer(engine, context, model_path);
std::vector<void *> buffers(engine->getNbBindings());
std::vector<nvinfer1::Dims> input_dims;
std::vector<nvinfer1::Dims> output_dims;
for (size_t i = 0; i < engine->getNbBindings(); ++i)
{
  auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
  std::cout<<"Index: "<<i<<" Binding_size: "<<binding_size<< " Engine binding Dim 0: "<<engine->getBindingDimensions(i).d[0]<<" Dim 1: "<<engine->getBindingDimensions(i).d[1]<< "\n";
  //cudaMalloc(&buffers[i], binding_size);
  if (engine->bindingIsInput(i))
  {
    input_dims.emplace_back(engine->getBindingDimensions(i));
  }
  else
  {
    output_dims.emplace_back(engine->getBindingDimensions(i));
  }
}
if (input_dims.empty() || output_dims.empty())
{
  std::cerr << "Expect at least one input and one output for network\n";
  return -1;
}
//std::cout<<"Input dims: "<< input_dims << ", output dims: "<<output_dims << endl;
float *inputPtr, *output;
H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float) * ML_SIZE * Total_Trace));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * Total_Trace * 22));
buffers[0] = inputPtr;
buffers[1] = output;
//cout<< "Input dim: "<< ML_SIZE * Total_Trace << endl;
float *trace;
Tick *aux_trace;
trace = (float *)malloc(TRACE_DIM * Instructions * sizeof(float));
aux_trace = (Tick *)malloc(AUX_TRACE_DIM * Instructions * sizeof(Tick));
read_trace_mem(argv[1], argv[2], trace, aux_trace, Instructions);
int Batch_size = Instructions / Total_Trace;
//cout << " Iterations: " << Batch_size << endl;
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
//float *train_data;
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
//H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB)*Total_Trace));
// For factor, mean and default values
H_ERR(cudaMalloc((void **)&factor_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&mean_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMemcpy(factor_d, &factor, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
struct timeval total_start, total_end;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0,pre=0, tr=0,inf=0,upd=0;
#ifdef DEBUG
FILE *pFile;
pFile= fopen ("trtcustom.bin", "wb");
#endif
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
  cout<<"trace read\n";
  H_ERR(cudaMemcpy(inst_d, inst, sizeof(Inst) * Total_Trace, cudaMemcpyHostToDevice));
  double check2 = wtime();
  tr+= (check2-check1);
  preprocess<<<4096, 64>>>(rob_d,inst_d, factor_d, mean_d, default_val_d, inputPtr, curTick, lastFetchTick, status, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  cout<<"preprocess done\n";
#ifdef DEBUG
  float *inp;
  inp= (float*) malloc(ML_SIZE*Total_Trace*sizeof(float));
  H_ERR(cudaMemcpy(inp,inputPtr, sizeof(float) * ML_SIZE*Total_Trace, cudaMemcpyDeviceToHost));
  fwrite(inp, sizeof(float), ML_SIZE, pFile);
#endif
  double check3 = wtime();
  pre+= (check3-check2);
  //context->enqueue(Total_Trace, buffers.data(), 0, nullptr); 
  context->enqueueV2(buffers.data(), 0, nullptr);
  //context->executeV2(buffers.data());
  cudaStreamSynchronize(0);
  double check4= wtime();
  inf+= (check4-check3);
  cout<<"Inference done\n";

  update<<<4096,32>>>(rob_d, output, factor_d, mean_d, curTick, lastFetchTick, status, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  double check5=wtime();
  upd+=(check5-check4);
  iteration++;
}
#ifdef DEBUG
  fclose(pFile);
#endif
printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red/Instructions*1000000, tr/Instructions*1000000, pre/Instructions*1000000, inf/Instructions*1000000, upd/Instructions*1000000);
double end_ = wtime();
for (void *buf : buffers)
{
  cudaFree(buf);
}
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
