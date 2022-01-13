#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <sys/time.h>

#include <torch/script.h> // One-stop header.

using namespace std;

//#define COMBINED
//#define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
//#define DUMP_IPC
#define DUMP_IPC_INTERVAL 1000
#define NO_MEAN
#define GPU

// Classic CPU dataset.
#define ROBSIZE 94
#define SQSIZE 17
#define MIN_COMP_LAT 6
#define MIN_ST_LAT 10
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 8

// PostK CPU dataset.
//#define ROBSIZE 185
//#define SQSIZE 25
//#define MIN_COMP_LAT 6
//#define MIN_ST_LAT 9
//#define FETCH_BANDWIDTH 8
//#define RETIRE_BANDWIDTH 4

#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TD_SIZE 50
#define CONTEXTSIZE (ROBSIZE + SQSIZE)
#define TICK_STEP 500.0
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define CLASS_NUM 10

#define INSQ_BIT 4
#define ATOMIC_BIT 13
#define SC_BIT 14
#define ILINEC_BIT 33
#define IPAGEC_BIT 37
#define DADDRC_BIT 41
#define DLINEC_BIT 42
#define DPAGEC_BIT 46

typedef long unsigned Tick;
typedef long unsigned Addr;

float default_val[ML_SIZE];

Addr getLine(Addr in) { return in & ~0x3f; }

#include "sim_module.h"

int main(int argc, char *argv[]) {
  if (argc < 5) {
    cerr << "Usage: ./simulator_qq_pa <module> <# inst> <trace> <aux trace> ..." << endl;
    return 0;
  }
  torch::jit::script::Module lat_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    lat_module = torch::jit::load(argv[1]);
    lat_module.eval();
#ifdef GPU
    lat_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    cerr << e.msg() << endl;
    return 0;
  }
  unsigned long long total_num = atol(argv[2]);

  for (int i = 0; i < TD_SIZE; i++) {
    default_val[i] = 0;
  }
  int trace_num = (argc - 3) / 2;
  assert((argc - 3) % 2 == 0);
  for (int i = TD_SIZE; i < ML_SIZE; i++)
    default_val[i] = default_val[i % TD_SIZE];
  at::Tensor input = torch::zeros({trace_num, ML_SIZE});
  float *inputPtr = input.data_ptr<float>();
  std::vector<torch::jit::IValue> inputs;
  at::Tensor outputTensor;
  SimModule mods[trace_num];
//#pragma omp parallel for
  for (int i = 0; i < trace_num; i++) {
    bool res = mods[i].init(argv[2*i+3], argv[2*i+4], argv[1], total_num);
    if (!res)
      return 0;
  }
#if defined(COMBINED)
  int output_stride = (CLASS_NUM+1)*3;
#else
  int output_stride = 3;
#endif

  struct timeval total_start, total_end;
  gettimeofday(&total_start, NULL);
  while(!mods[0].eof) {
#pragma omp parallel for num_threads(trace_num)
    for (int i = 0; i < trace_num; i++)
      mods[i].preprocess(inputPtr + ML_SIZE*i);
    if (mods[0].eof) {
      cout << "Finish early with shorter trace.\n";
      break;
    }
    inputs.clear();
#ifdef GPU
    inputs.push_back(input.cuda());
#else
    inputs.push_back(input);
#endif
    outputTensor = lat_module.forward(inputs).toTensor();
    outputTensor = outputTensor.to(at::kCPU);
    float *output = outputTensor.data_ptr<float>();
#pragma omp parallel for num_threads(trace_num)
    for (int i = 0; i < trace_num; i++)
      mods[i].postprocess(output + output_stride*i);
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  time_t now = time(0);
  unsigned long long inst_num = mods[0].fetched_inst_num * trace_num;
  cout << "Finish at " << ctime(&now);
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
#if defined(RUN_TRUTH)
  cout << "Truth" << "\n";
#elif defined(COMBINED)
  cout << "Combined Model: " << argv[1] << "\n";
#else
  cout << "Latency Model: " << argv[1] << "\n";
#endif
  for (int i = 0; i < trace_num; i++)
    mods[i].finish(argv[2*i+3], argv[2*i+4]);
  return 0;
}
