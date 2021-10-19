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


/*
__global__ void
preprocess(ROB *rob_d,SQ *sq_d, Inst *insts, float *default_val, float *inputPtr, int *status, int Total_Trace, int *index_all)
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
    //Inst *newInst;
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
	 Inst *newInst = rob->add();
	 memcpy(newInst, &insts[index], sizeof(Inst)); 
	 //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    }
    __syncwarp();
    //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    if (curTick != lastFetchTick)
    {
     //if(warpTID==0){printf("ROB update\n");}
     __syncwarp();
      rob->update_fetch_cycle(curTick - lastFetchTick);
      __syncwarp();
     //if(warpTID==0){printf("SQ update\n");}
     __syncwarp();
      sq->update_fetch_cycle(curTick - lastFetchTick);  
      __syncwarp();
    } 
    __syncwarp();
    //if(warpTID==0){printf("ROB: head: %d, tail: %d \n", rob->head, rob->tail);}
     //if(warpTID==0){printf("SQ: head: %d, tail: %d \n", sq->head, sq->tail);}
     __syncwarp();
    int rob_num= rob->make_input_data(input_Ptr, curTick, insts[index]);
    __syncwarp();
    int sq_num= sq->make_input_data(input_Ptr + rob_num * TD_SIZE, curTick, insts[index]);
    __syncwarp();
    int num= rob_num + sq_num;
    // copy default values
    //if(warpTID==0){printf("%d, %lu, %lu, %d, %d\n", index_all[index], curTick, lastFetchTick, rob_num, sq_num);} __syncwarp();
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
*/


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
  //printf("args count: %d\n", argc);
#ifdef WARMUP
  if (argc != 8)
  {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <Total trace><# inst> <W (warmup)> <U>" << endl;
    return 0;
    //int W= atoi(argv[6]);
  }
#else
  if (argc != 8)
  {
    cerr << "Usage: ./simulator_qq <trace> <aux trace> <lat module> <Total trace> <#Insts> <W not used> <U>" << endl;
  return 0;
}
#endif
//int arg_idx = 4;
//float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);

for (int i = 0; i < TD_SIZE; i++) {
    default_val[i] = 0;
  }
  for (int i = TD_SIZE; i < ML_SIZE; i++)
    default_val[i] = default_val[i % TD_SIZE];


//cout<< argv[3] << endl;
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
const unsigned long long int Total_Trace = atoi(argv[4]);
const unsigned long long int Instructions = atoi(argv[5]);
//cout<< "Total_Trace: "<< Total_Trace << ", Instructions: "<< Instructions << endl;
//std::string model_path(argv[3]);
at::Tensor input = torch::ones({atoi(argv[4]), ML_SIZE});
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


int Batch_size = Instructions / Total_Trace;
//printf("Batchsize: %d\n",Batch_size);
if(Instructions%Total_Trace!=0){
        //printf("Prev bsize: %d, mew bsize: %d\n", Batch_size, Batch_size + 1);
        Batch_size= Batch_size +1;
        unsigned long long int new_instr=  (Batch_size+1)*Total_Trace;
        trace = (float *)malloc(TRACE_DIM * new_instr * sizeof(float));
        aux_trace = (Tick *)malloc(AUX_TRACE_DIM * new_instr* sizeof(Tick));
        unsigned long long int index= Instructions;
        for (index; index<new_instr; index++){
                memcpy(&trace[index * TRACE_DIM], zeros, sizeof(float)*TRACE_DIM);
                memcpy(&aux_trace[index * AUX_TRACE_DIM], zeros, sizeof(Tick)*AUX_TRACE_DIM);
                index+=1;
         }
}
else {
        trace = (float *)malloc(TRACE_DIM * Instructions * sizeof(float));
        aux_trace = (Tick *)malloc(AUX_TRACE_DIM * Instructions * sizeof(Tick));
}
read_trace_mem(argv[1], argv[2], trace, aux_trace, Instructions);
//omp_set_num_threads(1);
double measured_time = 0.0;
int *fetched_inst_num = new int[Total_Trace];
int *fetched = new int[Total_Trace];
int *ROB_flag = new int[Total_Trace];
float *trace_all[Total_Trace];
Tick *aux_trace_all[Total_Trace];
int index_all[Total_Trace];
int *index_all_gpu;
H_ERR(cudaMalloc((void **)&index_all_gpu, sizeof(int) * Total_Trace));
//printf("variable init\n");
int W=0;
#ifdef WARMUP
W= atoi(argv[6]);
#endif
int U= atoi(argv[7]);
//#pragma omp parallel for
for (int i = 0; i < Total_Trace; i++)
{
  long long int offset = (i * (Batch_size))-U;
  //cout<<"offset: " << offset << endl;
//cout<< "W: "<<W<<", Index: "<< i <<" ,start: "<< offset << " ,warmup: "<< offset-W<< " ,End: "<<(offset + Batch_size) << endl;
#ifdef WARMUP
    offset= offset - W;
    //cout<< "W: "<<W<<", Index: "<< i <<", Offset: "<< offset << endl; 
#endif
    if(offset<0){offset=0;}
    //cout<< "W: "<<W<<", Index: "<< i <<" ,start: "<< offset << " ,warmup: "<< offset-W<< " ,End: "<<(offset + Batch_size) << endl;
  //if(offset>Instructions)printf("Index: %d, offset: %d\n",i,offset);
  //assert(offset<=Instructions);
  //assert(offset>=0);
  index_all[i]= offset;
  //cout<< "W: "<<W<<", Index: "<< i <<",Start: "<< offset <<" ,warmup: "<<offset + W << "End: "<<offset + Batch_size + W << endl;
  trace_all[i]= trace + offset * TRACE_DIM;
  aux_trace_all[i]= aux_trace + offset * AUX_TRACE_DIM;
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
struct timeval check3, t;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0,pre=0, tr=0,inf=0,upd=0;
FILE *pFile,*outFile;
pFile= fopen ("libcustom.bin", "wb");
//outFile= fopen("pred.bin", "wb");
//printf("Simulation started.. \n");
//:return 0;
int total_iterations= Batch_size + W;
while (iteration < total_iterations){
  //if((iteration % 500)==0)
  //{cout << "\nIteration: " << iteration << endl;}
  double st = wtime();
  #pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    index_all[i]+=1;
    if (!inst[i].read_sim_mem(trace_all[i], aux_trace_all[i],index_all[i]))
    {cout << "Error\n";}
    trace_all[i] += TRACE_DIM; aux_trace_all[i] += AUX_TRACE_DIM;
  }
  double check1 = wtime();
  red+= (check1-st);
  H_ERR(cudaMemcpy(inst_d, inst, sizeof(Inst) * Total_Trace, cudaMemcpyHostToDevice));
  double check2 = wtime();
  tr+= (check2-check1);
  //cout<<"Data transferred\n";
  H_ERR(cudaMemcpy(index_all_gpu, index_all, sizeof(int) * Total_Trace, cudaMemcpyHostToDevice));
  gettimeofday(&t, NULL);
  preprocess<<<4096, 32>>>(rob_d,sq_d,inst_d, default_val_d, inputPtr, status, Total_Trace, index_all_gpu, iteration, W, Batch_size);
  H_ERR(cudaDeviceSynchronize());
  //cout<<"Preprocess done \n"<<endl; 
  //double check3= wtime();
  gettimeofday(&check3, NULL);
  H_ERR(cudaMemcpy(inp,inputPtr, sizeof(float) * ML_SIZE * Total_Trace, cudaMemcpyDeviceToHost));
  //fwrite(inp, sizeof(float), ML_SIZE*Total_Trace, pFile);
  //int *in= (int *) malloc(1);
  //in[0]=iteration; 
  //fwrite(in,sizeof(int),1,pFile);
  //printf("Input:\n");
  //display(inp, 51,4);
  pre+= (check3-t);
  //printf(",%f \n",(check3-t));
  check3 = wtime();
  //pre+= (check3-check2);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input.cuda());  
  at::Tensor outputs = lat_module.forward(inputs).toTensor();
  outputs=outputs.to(at::kCPU);
  cudaStreamSynchronize(0);
  //cout<<outputs<<endl;
  double check4= wtime();
  inf+= (check4-check3);
  //cout<<"Output size: "<< outputs.sizes()[0]<<endl;
  //cout<<"Inference done \n";
  int out_shape= outputs.sizes()[1];
  H_ERR(cudaMemcpy(output, outputs.data_ptr<float>(), sizeof(float)*Total_Trace*out_shape, cudaMemcpyHostToDevice));
  update<<<4096,32>>>(rob_d,sq_d, output, status, Total_Trace, out_shape, iteration, W, Batch_size, index_all_gpu);
  H_ERR(cudaDeviceSynchronize());
  //cout<<"Update done\n";
  double check5=wtime();
  upd+=(check5-check4);
  iteration++;
  //if(iteration==11) return 0;
}
double e = wtime();
double end_ = wtime();
double total_time = check3.tv_sec - t.tv_sec + (check3.tv_usec - t.tv_usec) / 1000000.0;
printf("Avg: Total: %f, Pre %f\n",(end_-start_)/Instructions,total_time/Instructions);
return 0;
fclose(pFile);
//fclose(outFile);
//printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
//double end_ = wtime();

gettimeofday(&total_end, NULL);
Tick *total_tick;
H_ERR(cudaMalloc((void **)&total_tick, sizeof(Tick)));
string p(argv[3]);
string d(argv[2]);
size_t found= p.find_last_of("/\\");
//size_t found1=p.find_last_of("_");
cout<<argv[0]<<",";
cout<<p.substr(found+1)<<",";
found= d.find_last_of("/\\");
cout<< d.substr(found+1) <<",";
printf("%llu,%llu,%d,%d,",Instructions,Total_Trace,W,iteration);
result<<<1, 1>>>(rob_d, Total_Trace, Instructions, total_tick);
H_ERR(cudaDeviceSynchronize());
//H_ERR(cudaMemcpy(&total_tick[i], total_tick_d[i], sizeof(Tick), cudaMemcpyDeviceToHost));
double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
cout <<total_time<< endl;

return 0;
cout << Instructions << " instructions finish by " << (curTick - 1) << "\n";
cout << "Time: " << total_time << "\n";
cout << "MIPS: " << Instructions / total_time / 1000000.0 << "\n";
cout << "USPI: " << total_time * 1000000.0 / Instructions << "\n";
cout << "Measured Time: " << measured_time / Instructions << "\n";
//cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  //cout << "Lat Model: " << argv[3] << "\n";
#endif
return 0;
}
