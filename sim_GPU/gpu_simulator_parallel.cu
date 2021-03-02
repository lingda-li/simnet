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

//#define Total_Trace 1024

Tick Num = 0;




__global__ void
preprocess(ROB_d *rob_d, float *insts, float *factor, float *mean, float *default_val, float *inputPtr, float *train_data, Tick *curTick_d, Tick *lastFetchTick_d, int Total_Trace)
{

  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int warpID = TID / WARPSIZE;
  int warpTID = TID % WARPSIZE;
  int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
  int index, Total;
  ROB *rob;
  float *rob_pointer;
#ifdef WARP
  index = warpID;
  Total = TotalWarp;
#else
  index = blockIdx.x;
  Total = gridDim.x;
#endif
  while (index < Total_Trace)
  {
    rob = &rob_d->rob[index];
    Tick curTick = curTick_d[index];
    Tick lastFetchTick = lastFetchTick_d[index];
    //if(warpTID==0) { printf("Read: Warp: %d, assigned: %d, next: %d\n",warpID, index, index + Total);}
    //push new instruction to respective ROB but not latency
    //if(warpTID==0) { printf("Read: Warp: %d, assigned: %d, next: %d\n",warpID, index, index + Total);}
    //int tail= rob->dec(tail);
    rob_pointer = insts + ROBSIZE * INST_SIZE * index;
    float *input_Ptr = inputPtr + ML_SIZE * index;
    int i = warpTID + 4;
    while (i < INST_SIZE)
    {
      rob_pointer[i + INST_SIZE * rob->tail] = train_data[i + warpID * INST_SIZE];
      //printf("t: %d, i: %d, offset: %d\n",TID,i,train_offset);
      i += WARPSIZE;
    }
    __syncwarp();
    //if(warpTID==0) { printf("Inpt: %d\n",warpID);}
    if (warpTID == 0)
    {
      if (rob->is_full())
      {
        printf("retired\n");
        int retired = rob->retire_until(curTick, insts);
      }
      rob->add();
      //printf("Tail: %d, Curtick: %ld, lastFetchTick: %ld\n", rob->tail, curTick, lastFetchTick);
    }
    //printf("Update: ROB: %d, thread: %d, head:%d, tail: %d, newIndex: %d\n", index, threadIdx.x, rob->head, rob->tail, (index + gridDim.x * blockDim.x));
    __syncwarp();
    //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    if (curTick != lastFetchTick)
    {
      //if(warpTID==0){printf("update fetch\n");}
      rob->update_fetch_cycle(curTick - lastFetchTick, curTick, factor, rob_pointer);
    }
    __syncwarp();
    //if(TID==0){printf("update completed\n"); }
    //rob = &rob_d->rob[index];
    //while(index<Total_Trace){
    //if(warpTID==0) { printf("Make input: Warp: %d, assigned: %d,offset: %d, next: %d\n",warpID, index,ML_SIZE*index, index + Total);}
    rob->make_input_data(input_Ptr, rob_pointer, curTick, factor, default_val);
    if (warpTID == 0)
    {
      //printf("Input_Ptr\n");
      //dis(input_Ptr, TD_SIZE, 4);
    }
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
result(Tick *curTick, int Total_Trace)
{
  Tick sum = 0;
  for (int i = 0; i < Total_Trace; i++)
  {
    sum += curTick[i];
  }
  printf("Total CurTick: %ld\n", sum);
}

__global__ void
update(ROB_d *rob_d, float *output, float *insts, float *factor, float *mean, Tick *curTick, Tick *lastFetchTick, int Total_Trace)
{
  //float output[]={ 2.1987 ,0.4428,  0.0245 , 0.2029, 0.0094 , 0.1621};
  //printf("Here\n");
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int index = TID;
  ROB *rob;
  while (index < Total_Trace)
  {
    int offset = index * 2;
    //printf("index: %d, thread: %d, offset: %d \n",index,TID,offset);
    Tick nextFetchTick = 0;
    rob = &rob_d->rob[index];
    int rob_offset = ROBSIZE * INST_SIZE * index;
    int context_offset = rob->dec(rob->tail) * INST_SIZE;
    float *rob_pointer = insts + rob_offset + context_offset;
    printf("Head: %d, Tail: %d\n", rob->head, rob->tail);
    //printf("Index: %d, offset: %d,Fetch: %.4f, Finish: %.4f\n ",index,rob->tail,output[offset+0],output[offset+1]);
    float fetch_lat = output[offset + 0] * factor[1] + mean[1];
    float finish_lat = output[offset + 1] * factor[3] + mean[3];
    int int_fetch_lat = round(fetch_lat);
    int int_finish_lat = round(finish_lat);
    if (int_fetch_lat < 0)
      int_fetch_lat = 0;
    if (int_finish_lat < MIN_COMP_LAT)
      int_finish_lat = MIN_COMP_LAT;
    rob_pointer[0] = (-int_fetch_lat - mean[0]) / factor[0];
    rob_pointer[1] = (-int_fetch_lat - mean[1]) / factor[1];
    rob_pointer[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
    if (rob_pointer[2] >= 9 / factor[2])
    {
      rob_pointer[2] = 9 / factor[2];
    }
    rob_pointer[3] = (int_finish_lat - mean[3]) / factor[3];
    printf("Index: %d, offset: %d, Fetch: %.4f, Finish: %.4f, Rob0: %.2f, Rob1: %.2f, Rob2: %.2f, Rob3: %.2f\n", index, rob->tail, output[offset + 0], output[offset + 1], rob_pointer[0], rob_pointer[1], rob_pointer[2], rob_pointer[3]);
    rob_pointer[COMPLETETICK] = curTick[index] + int_finish_lat + int_fetch_lat;
    lastFetchTick[index] = curTick[index];
    if (int_fetch_lat)
    {
      nextFetchTick = curTick[index] + int_fetch_lat;
    }
    if ((rob->is_full() || rob->saturated) && int_fetch_lat)
    {
      curTick[index] = max(rob_pointer[COMPLETETICK], nextFetchTick);
    }
    else if (int_fetch_lat)
    {
      curTick[index] = nextFetchTick;
    }
    else if (rob->saturated || rob->is_full())
    {
      curTick[index] = rob_pointer[COMPLETETICK];
    }

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
ROB_d *rob_d;
Tick Case0 = 0;
Tick Case1 = 0;
Tick Case2 = 0;
Tick Case3 = 0;
Tick Case4 = 0;
Tick Case5 = 0;
float *inputPtr, *output;
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
//train_data= (float*) malloc(Total_Trace*TD_SIZE*sizeof(float));
H_ERR(cudaMalloc((void **)&curTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&lastFetchTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * Total_Trace * 2));
cudaMemset(curTick, 0, Total_Trace);
cudaMemset(lastFetchTick, 0, Total_Trace);
cudaHostAlloc((void **)&train_data, Total_Trace *INST_SIZE * sizeof(float),
              cudaHostAllocDefault);

ROB_d rob = ROB_d(Total_Trace);
H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float) * ML_SIZE * Total_Trace));
//printf("Total mem: %d\n",ML_SIZE*Total_Trace);
H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB_d)));
//H_ERR(cudaMalloc((void **)&insts, sizeof(float)*Total_Trace*ROB_SIZE*INST_SIZE));
H_ERR(cudaMalloc((void **)&factor_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&mean_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * (TD_SIZE)*2));
H_ERR(cudaMemcpy(rob_d, &rob, sizeof(ROB_d), cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(factor_d, &factor, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
//H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float)*TD_SIZE*2, cudaMemcpyHostToDevice));
buffers[0] = inputPtr;
buffers[1] = output;
struct timeval start, end, total_start, total_end;
Inst Inst_;
float *train_data_d, *insts;
H_ERR(cudaMalloc((void **)&train_data_d, sizeof(float) * Total_Trace * INST_SIZE));
H_ERR(cudaMalloc((void **)&insts, sizeof(float) * Total_Trace * ROBSIZE * INST_SIZE));
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
while (iteration < Batch_size)
{
  cout << "Iteration: " << iteration << endl;
  double st = wtime();
#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    Inst newInst(train_data);
    if (!newInst.read_sim_mem(trace_all[i], aux_trace_all[i], train_data, i))
    {
      cout << "Inside 1st\n";
    }
    trace_all[i] += TRACE_DIM;
    aux_trace_all[i] += AUX_TRACE_DIM;
  }
  display(train_data, INST_SIZE, 1);
  double check1 = wtime();
  H_ERR(cudaMemcpy(train_data_d, train_data, sizeof(float) * Total_Trace * INST_SIZE, cudaMemcpyHostToDevice));

  double check2 = wtime();
  preprocess<<<1, 32>>>(rob_d, insts, factor_d, mean_d, default_val_d, inputPtr, train_data_d, curTick, lastFetchTick, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  double check3 = wtime();
  //context->enqueue(Total_Trace, buffers.data(), 0, nullptr);
  context->enqueueV2(buffers.data(), 0, nullptr);
  cudaStreamSynchronize(0);
  update<<<1, 32>>>(rob_d, output, insts, factor_d, mean_d, curTick, lastFetchTick, Total_Trace);
  H_ERR(cudaDeviceSynchronize());
  iteration++;
}
double end_ = wtime();
for (void *buf : buffers)
{
  cudaFree(buf);
}
gettimeofday(&total_end, NULL);
result<<<1, 1>>>(curTick, Total_Trace);
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
