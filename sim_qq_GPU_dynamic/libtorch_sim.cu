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
#include <algorithm>
using namespace std;
#define NO_MEAN
#define GPU
#define WARP
//#define DEBUG

//#define Total_Trace 1024

Tick Num = 0;



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
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <Total trace><# inst> <partition_bin> <W (addtitional warmup)>" << endl;
    return 0;
    //int W= atoi(argv[6]);
  }
#else
  if (argc != 7)
  {
    cerr << "Usage: ./simulator_qq <trace> <aux trace> <lat module> <Total trace> <#Insts> <partition_bin>" << endl;
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
//Total_Trace= 7;
unsigned long long int Instructions = atoi(argv[5]);
//cout<< "Total_Trace: "<< Total_Trace << ", Instructions: "<< Instructions << endl;
//std::string model_path(argv[3]);
at::Tensor input = torch::ones({(int)Total_Trace, ML_SIZE});
float *inp= input.data_ptr<float>();
//cout<<"Input dims: "<< input_dims << ", output dims: "<<output_dims << endl;
float *inputPtr, *output;
int o_d; // dimension of ML output
#ifdef COMBINED
o_d= 33;
#else
o_d= 3;
#endif
H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float) * ML_SIZE * Total_Trace));
H_ERR(cudaMalloc((void **)&output, sizeof(float) * Total_Trace * o_d));
//cout<< "Input dim: "<< ML_SIZE * Total_Trace << endl;
int W=0;
#ifdef WARMUP
W= atoi(argv[7]);
#endif
int part_count= fsize(argv[6])/(sizeof(p_index)*2)-1; // counts total number of partitions from bin file
int *parts= (int *)malloc(sizeof(int)*part_count*2); // allocates memory for partition file
int *part_start= (int *)malloc(sizeof(int)*part_count); // partition start
int *part_end= (int *)malloc(sizeof(int)*part_count);   // partition end
//int *partition_index= (int *)malloc(sizeof(int)*part_count);   // index for actual partition; same with part_start if no warmup 
int *warmup_index= (int *)malloc(sizeof(int)*part_count);   // index from where subtraces should be reset
//int *partition_index_d,
int *current_index_d, *warmup_reset_index_d;
//H_ERR(cudaMalloc((void **)&partition_index_d, sizeof(int) * part_count)); // partition index for device
H_ERR(cudaMalloc((void **)&current_index_d, sizeof(int) * part_count)); // index counter for each subtrace
H_ERR(cudaMalloc((void **)&warmup_reset_index_d, sizeof(int) * part_count)); // subtrace reset index for GPU
partition(argv[6], Total_Trace, parts, part_start, part_end, warmup_index, W);  // -----> Replace with part_count to go through all instructions
Instructions= part_end[Total_Trace-1];
float *trace;
Tick *aux_trace;
trace = (float *)malloc(TRACE_DIM * Instructions * sizeof(float));
aux_trace = (Tick *)malloc(AUX_TRACE_DIM * Instructions * sizeof(Tick));
read_trace_mem(argv[1], argv[2], trace, aux_trace, Instructions);
H_ERR(cudaMemcpy(warmup_reset_index_d, warmup_index, sizeof(int) * part_count, cudaMemcpyHostToDevice));
//omp_set_num_threads(1);
double measured_time = 0.0;

int *fetched_inst_num = new int[Total_Trace];
int *fetched = new int[Total_Trace];
int *ROB_flag = new int[Total_Trace];
float *trace_all[Total_Trace];
Tick *aux_trace_all[Total_Trace];
int index_all[Total_Trace];
#pragma omp parallel for
for (int i = 0; i < Total_Trace; i++)
{
  unsigned long long int offset = part_start[i];
  index_all[i]= offset;
#ifdef DEBUG
  cout<< " Index: "<< i <<", Offset: "<< offset << endl;
#endif
  trace_all[i]= trace + offset * TRACE_DIM;
  aux_trace_all[i]= aux_trace + offset * AUX_TRACE_DIM;
  if(offset>Instructions) printf("i: %d, offset: %d, instr: %d\n",i,offset,Instructions);
  assert (offset<=Instructions);
  //printf("Trace: %d, offset: %d\n",i,part_start[i]);
}
// printf("Allocated. \n");
//return 0;
float *default_val_d;
Tick *curTick, *lastFetchTick;
int *inf_index, *status;
H_ERR(cudaMemcpy(current_index_d, index_all, sizeof(int) * Total_Trace, cudaMemcpyHostToDevice));
H_ERR(cudaMalloc((void **)&curTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&lastFetchTick, sizeof(Tick) * Total_Trace));
H_ERR(cudaMalloc((void **)&inf_index, sizeof(int) * 1));
H_ERR(cudaMalloc((void **)&status, sizeof(int)*Total_Trace));
cudaMemset(curTick, 0, Total_Trace);
cudaMemset(lastFetchTick, 0, Total_Trace);
cudaMemset(status, 1, Total_Trace);
struct SQ *sq= new SQ[Total_Trace];
struct ROB *rob= new ROB[Total_Trace];
struct Inst *inst= new Inst[Total_Trace];
struct ROB *rob_d; 
struct SQ *sq_d;
struct Inst *inst_d;
int *inf_id;
H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB)*Total_Trace));
H_ERR(cudaMalloc((void **)&sq_d, sizeof(SQ)*Total_Trace));
H_ERR(cudaMalloc((void **)&inst_d, sizeof(Inst)*Total_Trace));
H_ERR(cudaMalloc((void **)&inf_id, sizeof(int)*Total_Trace));
// For factor, mean and default values
H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float) * (TD_SIZE)));
H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float) * TD_SIZE, cudaMemcpyHostToDevice));
struct timeval total_start, total_end;
int iteration = 0;
gettimeofday(&total_start, NULL);
double start_ = wtime();
double red=0, pre=0, tr=0, inf=0, upd=0;
FILE *pFile,*outFile;
//pFile= fopen ("libcustom.bin", "wb");
//outFile= fopen("pred.bin", "wb");
int completed=0;
bool *active_d;
bool *active= (bool *)malloc(Total_Trace * sizeof(bool));
int *index_count_d;
H_ERR(cudaMalloc((void **)&index_count_d, sizeof(int) * Total_Trace));
H_ERR(cudaMalloc((void **)&active_d, sizeof(bool) * (Total_Trace)));
cudaMemset(active_d, true, Total_Trace*sizeof(bool));
memset(active,true,Total_Trace*sizeof(bool));
H_ERR(cudaDeviceSynchronize());
//int iteration=0;
while (completed!=1){
  //if((iteration % 500)==0
      	//{cout << "\nIteration: " << iteration++ << endl;}
 //cout<< "*******************************************************\n";
//if (iteration==11){break;} 
  double st = wtime();
  #pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
	  //printf("%d\n",i);
    //if(i==8951){printf("%d,%d,%d,%d\n",i,active[i],index_all[i],part_end[i]);}
	  if(index_all[i]<part_end[i]){
    index_all[i]+=1;
    if (!inst[i].read_sim_mem(trace_all[i], aux_trace_all[i],index_all[i]))
    {cout << "Error\n";}
    if(i==8951){printf("\nR: %d,%d,%d,%d [inside]\n",i,active[i],index_all[i],part_end[i]);}
    //index_all[i]+=1;
    //printf("%d\n",index_all[i]);
    trace_all[i] += TRACE_DIM; aux_trace_all[i] += AUX_TRACE_DIM;
    //printf("ID: %d,instr_index: %d\n",i,index_all[i]);
    //printf("A: ind: %d, end: %d\n ",index_all[i], part_end[i]);
    active[i]=true;
    }
    else{active[i]=false;}
  }
#ifdef DEBUG
  printf("Inst read\n"); 
#endif
  double check1 = wtime();
  red+= (check1-st);
    H_ERR(cudaMemcpy(inst_d, inst, sizeof(Inst) * Total_Trace, cudaMemcpyHostToDevice));
    double check2 = wtime();
  tr+= (check2-check1);
#ifdef DEBUG
  cout<<"Data transferred\n";
#endif
  H_ERR(cudaMemcpy(current_index_d, index_all, sizeof(int) * Total_Trace, cudaMemcpyHostToDevice));
  H_ERR(cudaMemcpy(active_d, active, sizeof(bool) * Total_Trace, cudaMemcpyHostToDevice));
 cudaMemset(inf_index, 0, sizeof(int));
H_ERR(cudaDeviceSynchronize());
 cout<<"Preprocess\n";
  preprocess<<<4096,64>>>(rob_d,sq_d,inst_d, default_val_d, inputPtr, status, Total_Trace, warmup_reset_index_d, active_d, inf_id, inf_index, current_index_d);
  H_ERR(cudaDeviceSynchronize());
#ifdef DEBUG
  cout<<"Preprocess done \n"<<endl; 
#endif
  double check3= wtime();
    H_ERR(cudaMemcpy(inp,inputPtr, sizeof(float) * ML_SIZE*Total_Trace, cudaMemcpyDeviceToHost));
  //fwrite(inp+ML_SIZE, sizeof(float), ML_SIZE, pFile);
  //printf("Input:\n");
  //display(inp, 51,4);
  pre+= (check3-check2);
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
int out_shape= outputs.sizes()[0];
#ifdef DEBUG
  cout<<"Inference done\n";
  int out_shape= outputs.sizes()[0];
  printf("Shape: %d, trace: %lu\n", out_shape, Total_Trace);
#endif
  H_ERR(cudaMemcpy(output, outputs.data_ptr<float>(), sizeof(float) * Total_Trace*o_d, cudaMemcpyHostToDevice));
  //printf("Shape: %d, trace: %d\n", out_shape, Total_Trace);
  //fwrite(outputs.data_ptr<float>(), sizeof(float), 3, outFile);
  //H_ERR(cudaMemcpy(index_all_gpu, index_all, sizeof(int) * Total_Trace, cudaMemcpyHostToDevice));
  update<<<4096,64>>>(rob_d,sq_d, output, status, Total_Trace, o_d, warmup_reset_index_d, current_index_d, active_d, inf_id, inf_index);
  H_ERR(cudaDeviceSynchronize());
#ifdef DEBUG
  cout<<"Update done\n";
#endif
  double check5=wtime();
  upd+=(check5-check4);
  // check for completion
 //bool allTrue = (std::end(active) == std::find(std::begin(active),std::end(active),false) );
  //return 0;
  completed=1;
  for(int j=0; j<Total_Trace; j++){
	  if(active[j]==1){completed=0;break;}
  }
  iteration++;
}
//fclose(pFile);
//fclose(outFile);
//printf("%.4f, %.4f, %.4f, %.4f, %.4f\n",red, tr, pre, inf, upd);
double end_ = wtime();

gettimeofday(&total_end, NULL);
Tick *total_tick;
H_ERR(cudaMalloc((void **)&total_tick, sizeof(Tick)));
result<<<1, 1>>>(rob_d, Total_Trace, Instructions, total_tick);
H_ERR(cudaDeviceSynchronize());
//H_ERR(cudaMemcpy(&total_tick[i], total_tick_d[i], sizeof(Tick), cudaMemcpyDeviceToHost));
double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
//cout << "Total time: " << (end_ - start_) << endl;
#ifdef RUN_TRUTH
cout << "Truth"
     << "\n";
#endif
//return 0;
cout << Instructions << " instructions finish by " << (iteration) << " iterations.\n";
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
