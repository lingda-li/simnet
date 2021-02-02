#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include "init.cuh"
//#include <torch/script.h> // One-stop header.

using namespace std;

//#define CLASSIFY
//#define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU
typedef long unsigned Tick;
typedef long unsigned Addr;
Tick Num = 0;

float factor[TD_SIZE];
float mean[TD_SIZE];
float default_val[TD_SIZE];

struct Inst {
  float train_data[TD_SIZE];
  Tick inTick;
  Tick completeTick;
  Tick tickNum;
  Tick trueFetchTick;
  Tick trueCompleteTick;
  int trueFetchClass;
  int trueCompleteClass;
  Addr pc;
  int isAddr;
  Addr addr;
  Addr addrEnd;
  Addr iwalkAddr[3];
  Addr dwalkAddr[3];
  // Read simulation data.
  bool read_sim_data(ifstream &trace, ifstream &aux_trace) {
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++) {
      trace >> train_data[i];
      train_data[i] /= factor[i];
    }
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = train_data[3] = 0.0;
    aux_trace >> isAddr >> addr >> addrEnd;
    for (int i = 0; i < 3; i++)
      aux_trace >> iwalkAddr[i];
    for (int i = 0; i < 3; i++)
      aux_trace >> dwalkAddr[i];
    assert(!trace.eof() && !aux_trace.eof());
    //cout << "in: ";
    //for (int i = 0; i < TD_SIZE; i++)
    //  cout << train_data[i] << " ";
    //cout << "\n";
    return true;
  }
};



__global__ void 
preprocess(ROB *rob, int fetched, int curTick, int lastFetchTick, float *inputPtr )
{
    Inst_d *newInst = rob->add();
    int retired = rob->retire_until(curTick); 
    fetched++;
    newInst->inTick = curTick;
    if (curTick != lastFetchTick) {
        rob->update_fetch_cycle(curTick - lastFetchTick, curTick);
    }
    rob->make_input_data(inputPtr, curTick);
}


float *read_numbers(char *fname, int sz) {
  float *ret = new float[sz];
  ifstream in(fname);
  printf("Trying to read from %s\n", fname);
  for(int i=0;i<sz;i++)
    in >> ret[i];
  return ret;
}

int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc != 7) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst>" << endl;
#else
  if (argc != 6) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <variances> <# inst>" << endl;
#endif
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream aux_trace(argv[2]);
  if (!aux_trace.is_open()) {
    cerr << "Cannot open auxiliary trace file.\n";
    return 0;
  }
  int arg_idx=4;
  float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);
  unsigned long long total_num = atol(argv[arg_idx++]);

  for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i];
    cout << default_val[i] << " ";
  }
  cout << "\n";
  //at::Tensor input = torch::ones({1, ML_SIZE});
  //float *inputPtr = input.data_ptr<float>();
  Inst *newInst;
  unsigned long long inst_num = 0;
  unsigned long long fetched_inst_num = 0;
  double measured_time = 0.0;
  Tick curTick = 0;
  Tick lastFetchTick = 0;
  bool eof = false;
  ROB *rob;
  Tick nextFetchTick = 0;
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  Tick Case4 = 0;
  Tick Case5 = 0;
  float *inputPtr;
  H_ERR(cudaMalloc((void **)&inputPtr, sizeof(int)*ML_SIZE));
  
  //H_ERR(cudaMalloc((void **)&factor, sizeof(int)*TD_SIZE));
  //H_ERR(cudaMalloc((void **)&mean, sizeof(int)*TD_SIZE));
  //H_ERR(cudaMalloc((void **)&default_val, sizeof(int)*TD_SIZE));
  //float factor[TD_SIZE];
  //float mean[TD_SIZE];
  //float default_val[TD_SIZE];
  
  //HRR(cudaMemcpy(sampler, &S, sizeof(Sampling), cudaMemcpyHostToDevice));
  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  bool is_empty=false;
  bool is_full=false;
  bool saturated=false;
  int retired=0;
  while(!eof || !is_empty) {
    // Retire instructions.
    inst_num += retired;
    int fetched = 0;
    int int_fetch_lat;
    while (fetched < FETCH_BANDWIDTH && !is_full && !eof) {
      
      if (!newInst->read_sim_data(trace, aux_trace)) {
        eof = true;
        //rob->tail = rob->dec(rob->tail);
        break;
      }
      //preprocess<<<1,1>>>(rob);
	float output[10];
      	measured_time += (end.tv_sec - start.tv_sec) * 1000000.0 + end.tv_usec - start.tv_usec;
      //cout << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << "\n";
#ifdef CLASSIFY
      int f_class, c_class;
      for (int i = 0; i < 2; i++) {
        float max = cla_output[0][10*i].item<float>();
        int idx = 0;
        for (int j = 1; j < 10; j++) {
          if (max < cla_output[0][10*i+j].item<float>()) {
            max = cla_output[0][10*i+j].item<float>();
            idx = j;
          }
        }
        if (i == 0)
          f_class = idx;
        else
          c_class = idx;
     }
#endif
      float fetch_lat = output[0] * factor[1] + mean[1];
      float finish_lat = output[0] * factor[3] + mean[3];
      int_fetch_lat = round(fetch_lat);
      int int_finish_lat = round(finish_lat);
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;
      if (int_finish_lat < MIN_COMP_LAT)
        int_finish_lat = MIN_COMP_LAT;
#ifdef CLASSIFY
      if (f_class <= 8)
        int_fetch_lat = f_class;
      if (c_class <= 8)
        int_finish_lat = c_class + MIN_COMP_LAT;
      //std::cout << curTick << ": ";
      //std::cout << " " << f_class << " " << fetch_lat << " " << int_fetch_lat << " " << newInst->trueFetchTick << " :";
      //std::cout << " " << c_class << " " << finish_lat << " " << int_finish_lat << " " << newInst->trueCompleteTick << '\n';
#endif
#ifdef DUMP_ML_INPUT
      int_finish_lat = newInst->trueCompleteTick;
      int_fetch_lat = newInst->trueFetchTick;
#endif
      newInst->train_data[0] = (-int_fetch_lat - mean[0]) / factor[0];
      newInst->train_data[1] = (-int_fetch_lat - mean[1]) / factor[1];
      newInst->train_data[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
      if (newInst->train_data[2] >= 9 / factor[2])
        newInst->train_data[2] = 9 / factor[2];
      newInst->train_data[3] = (int_finish_lat - mean[3]) / factor[3];
      newInst->tickNum = int_finish_lat;
      newInst->completeTick = curTick + int_finish_lat + int_fetch_lat;
      lastFetchTick = curTick;
      if (total_num && fetched_inst_num == total_num) {
        eof = true;
        break;
      }
      if (int_fetch_lat) {
        nextFetchTick = curTick + int_fetch_lat;
        break;
      }
    }
#ifdef DEBUG
    if (fetched)
      cout << curTick << " f " << fetched << "\n";
#endif
    if ((is_full || saturated) && int_fetch_lat) {
      // Fast forward curTick to the next cycle when it is able to fetch and retire instructions.
      //curTick = max(rob->getHead()->completeTick, nextFetchTick);
      if (is_full)
        Case1++;
      else
        Case2++;
    } else if (int_fetch_lat) {
      // Fast forward curTick to fetch instructions.
      curTick = nextFetchTick;
      Case0++;
    } else if (is_full || saturated) {
      // Fast forward curTick to retire instructions.
      //curTick = rob->getHead()->completeTick;
      if (is_full)
        Case3++;
      else
        Case4++;
    } else {
      curTick++;
      Case5++;
    }
    int_fetch_lat = 0;
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace.close();
  aux_trace.close();
#ifdef RUN_TRUTH
  cout << "Truth" << "\n";
#endif
  cout << inst_num << " instructions finish by " << (curTick - 1) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  cout << "Lat Model: " << argv[3] << "\n";
#endif
  return 0;
}

