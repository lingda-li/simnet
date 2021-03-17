#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
#include <ctime>
#include <vector>
#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>

using namespace std;
//#define CLASSIFY
//#define DEBUG
//#define NGPU_DEBUG
//#define VERBOSE
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU
//#define PREFETCH
// #define ROB_ANALYSIS
// #define HALF
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TD_SIZE 51
#define CONTEXTSIZE 111
#define ROBSIZE 400
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6
#define ILINEC_BIT 33
#define IPAGEC_BIT 38
#define DADDRC_BIT 42
#define DLINEC_BIT 43
#define DPAGEC_BIT 47

typedef long unsigned Tick;
typedef long unsigned Addr;
Tick Num = 0;

float factor[TD_SIZE];
float mean[TD_SIZE];

float default_val[TD_SIZE];

Addr getLine(Addr in) { return in & ~0x3f; }

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
    pc = getLine(pc);
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

struct ROB {
  Inst insts[ROBSIZE + 1];
  int head = 0;
  int tail = 0;
  bool saturated = false;
  int inc(int input) {
    if (input == ROBSIZE)
      return 0;
    else
      return input + 1;
  }
  int dec(int input) {
    if (input == 0)
      return ROBSIZE;
    else
      return input - 1;
  }
  bool is_empty() { return head == tail; }
  bool is_full() { return head == inc(tail); }
  Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    return &insts[old_tail];
  }
  Inst *getHead() {
    return &insts[head];
  }
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  int retire_until(Tick tick) {
    int retired = 0;
    while (!is_empty() && insts[head].completeTick <= tick) {
      retire();
      retired++;
    }
    return retired;
  }

  void make_input_data(float *context, Tick tick) {
    assert(!is_empty());
    saturated = false;
    Addr pc = insts[dec(tail)].pc;
    int isAddr = insts[dec(tail)].isAddr;
    Addr addr = insts[dec(tail)].addr;
    Addr addrEnd = insts[dec(tail)].addrEnd;
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
      iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
      dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
    }
    std::copy(insts[dec(tail)].train_data,
              insts[dec(tail)].train_data + TD_SIZE, context);
    int num = 1;
    for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
      if (insts[i].completeTick <= tick)
        continue;
      if (num >= CONTEXTSIZE) {
        saturated = true;
        break;
      }
      // Update context instruction bits.
      insts[i].train_data[ILINEC_BIT] =
          insts[i].pc == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
      int conflict = 0;
      for (int j = 0; j < 3; j++) {
        if (insts[i].iwalkAddr[j] != 0 && insts[i].iwalkAddr[j] == iwalkAddr[j])
          conflict++;
      }
      insts[i].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
      insts[i].train_data[DADDRC_BIT] =
          (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr &&
           addr <= insts[i].addrEnd)
              ? 1.0 / factor[DADDRC_BIT]
              : 0.0;
      insts[i].train_data[DLINEC_BIT] =
          (isAddr && insts[i].isAddr &&
           (addr & ~0x3f) == (insts[i].addr & ~0x3f))
              ? 1.0 / factor[DLINEC_BIT]
              : 0.0;
      conflict = 0;
      if (isAddr && insts[i].isAddr)
        for (int j = 0; j < 3; j++) {
          if (insts[i].dwalkAddr[j] != 0 &&
              insts[i].dwalkAddr[j] == dwalkAddr[j])
            conflict++;
        }
      insts[i].train_data[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE,
                context + num * TD_SIZE);
      num++;
    }
    for (int i = num; i < CONTEXTSIZE; i++) {
      std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
    }
  }

  void update_fetch_cycle(Tick tick, Tick curTick) {
    assert(!is_empty());
    for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
      if (insts[i].completeTick <= curTick)
        continue;
      insts[i].train_data[0] += tick / factor[0];
      if (insts[i].train_data[0] >= 9 / factor[0])
        insts[i].train_data[0] = 9 / factor[0];
      insts[i].train_data[1] += tick / factor[1];
      assert(insts[i].train_data[0] >= 0.0);
      assert(insts[i].train_data[1] >= 0.0);
    }
  }
};

float *read_numbers(char *fname, int sz) {
  float *ret = new float[sz];
  ifstream in(fname);
  printf("Trying to read from %s\n", fname);
  for(int i=0;i<sz;i++)
    in >> ret[i];
  return ret;
}

int main(int argc, char *argv[])
{
#ifdef CLASSIFY
  if (argc != 10) {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <#Batchsize> "
            "<Total_instr> <#nGPU> <OpenMP threads> <variances> <class module>"
         << endl;
    return 0;
  }
#else
  if (argc != 9) {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <#Batchsize> "
            "<Total_instr> <#nGPU> <OpenMP threads> <variances>"
         << endl;
    return 0;
  }
#endif
  cout << "Arguments provided: " << argc << endl;
  ifstream trace_test(argv[1]);
  if (!trace_test.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream aux_trace_test(argv[2]);
  if (!aux_trace_test.is_open()) {
    cerr << "Cannot open auxiliary trace file.\n";
    return 0;
  }

  int Total_Trace = atoi(argv[4]);
  std::string line;
  int lines = 0;
  //while (std::getline(trace_test, line))
    //++lines;
  int Total_instr = atoi(argv[5]);
  int Batch_size = Total_instr / Total_Trace;
  cout << "Simulate " << Total_instr << " instructions (" << Total_Trace
       << " batches * " << Batch_size << ")\n";

  int nGPU = atoi(argv[6]);
  if ((int)torch::cuda::device_count() < nGPU) {
    cerr << "GPUs not enough" << endl;
    return 0;
  }
  torch::jit::script::Module lat_module[nGPU];
#ifdef CLASSIFY
  torch::jit::script::Module cla_module[nGPU];
  at::Tensor cla_output[nGPU];
#endif
  at::Tensor *input = new at::Tensor[nGPU];
  try {
// Deserialize the ScriptModule from a file using torch::jit::load().
#ifdef GPU
#pragma omp parallel for
    for (int i = 0; i < nGPU; i++) {
      lat_module[i] = torch::jit::load(argv[3]);
      input[i] = torch::ones({Total_Trace / nGPU, ML_SIZE});
      string dev = "cuda:";
      string id = to_string(i);
      dev = dev + id;
      const std::string device_string = dev;
      lat_module[i].to(device_string);
#ifdef CLASSIFY
      cla_module[i] = torch::jit::load(argv[8]);
      cla_module[i].to(device_string);
#endif
    }
#endif
  } catch (const c10::Error &e) {
    cerr << "error loading the model\n";
    return 0;
  }
  cout << "Parameters loaded..." << endl;

  float *varPtr = read_numbers(argv[8], TD_SIZE);
  cout << "vars: ";
  for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i];
    cout << default_val[i] << " ";
  }
  cout << endl;

  omp_set_num_threads(96);
  ifstream *trace = new ifstream[Total_Trace];
  ifstream *aux_trace = new ifstream[Total_Trace];
  Tick *curTick = new Tick[Total_Trace];
  Tick *nextFetchTick = new Tick[Total_Trace];
  Tick *lastFetchTick = new Tick[Total_Trace];
  // cout<<"Ticks loaded..."<<endl;
  int *index = new int[Total_Trace];
  int *inst_num_all = new int[Total_Trace];
  int *fetched_inst_num = new int[Total_Trace];
  int *fetched = new int[Total_Trace];
  int *ROB_flag = new int[Total_Trace];
  int *int_fetch_latency = new int[Total_Trace];
  int *int_finish_latency = new int[Total_Trace];
  bool *eof = new bool[Total_Trace];
#ifdef PREFETCH
  char Trace_Buffer[Total_Trace][20000];
  char AuxTrace_Buffer[Total_Trace][20000];
  char **Trace_Buffer = new char *[Total_Trace];
  for
  char **AuxTrace_Buffer = new char *[Total_Trace];
#endif

#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++) {
    curTick[i] = 0;
    nextFetchTick[i] = 0;
    lastFetchTick[i] = 0;
    inst_num_all[i] = 0;
    fetched_inst_num[i] = 0;
    fetched[i] = 0;
    eof[i] = false;
    int offset = i * Batch_size;
    std::string line, line1;
    int number_of_lines = 0;
#ifdef PREFETCH
    trace[i].rdbuf()->pubsetbuf(Trace_Buffer[i], 20000);
    aux_trace[i].rdbuf()->pubsetbuf(AuxTrace_Buffer[i], 20000);
#endif
    trace[i].open(argv[1]);
    aux_trace[i].open(argv[2]);
    while (std::getline(trace[i], line) && std::getline(aux_trace[i], line1) &&
           (number_of_lines < offset))
      ++number_of_lines;
    // number_of_lines = 0;
    // while (std::getline(aux_trace[i], line) && (number_of_lines < offset))
    //++number_of_lines;
    // cout <<endl<< "I:" << i << "\tSkipped: " << number_of_lines << endl;
    if (i == 0) {
      trace[0].seekg(0, trace[0].beg);
      aux_trace[0].seekg(0, aux_trace[0].beg);
    }
  }

  int i, count = 0, stop_flag = 0;
  struct ROB *rob = new ROB[Total_Trace];
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  Tick Case4 = 0;
  Tick Case5 = 0;
  double measured_time = 0.0;
  struct timeval start, end, total_start, total_end, end_first, start_first;
  struct timeval loop1_start, loop2_start, loop3_start, loop4_start,
      loop5_start, loop1_end, loop2_end, loop3_end, loop4_end, loop5_end;
  double loop1_time = 0, loop2_time = 0, loop3_time = 0, loop4_time = 0,
         loop5_time = 0;
  gettimeofday(&total_start, NULL);
  omp_set_num_threads(atoi(argv[7]));
#ifdef DEBUG
  cout << "Simulation starting....." << endl;
#endif
  Inst **newInst = new Inst *[Total_Trace];
  while (stop_flag != 1) {
    int *inference_count = new int[nGPU];
    at::Tensor output[nGPU];
    gettimeofday(&start, NULL);
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++) {
      if (!eof[i] || !rob[i].is_empty()) {
        // Retire instructions.
        if (ROB_flag[i]) {
          int retired = rob[i].retire_until(curTick[i]);
#ifdef DEBUG
          cout << "Count: " << count << ", Retired: " << retired
               << ", Retired until: " << inst_num_all[i] << endl;
#endif
          inst_num_all[i] += retired;
          fetched[i] = 0;
#ifdef DEBUG
          cout << "************"
               << ", Curtick: " << curTick[i] << ", Retired: " << retired
               << ", Is_full: " << rob->is_full() << "************"
               << "\n";
#endif
        }
        //if (fetched[i] < FETCH_BANDWIDTH && !rob[i].is_full() && !eof[i])
        if (!rob[i].is_full() && !eof[i]) {
          ROB_flag[i] = false;
          if (inst_num_all[i] > (Batch_size - 1)) {
            eof[i] = true;
            rob[i].head = (rob[i].tail);
#ifdef DEBUG
            cout << "Trace: " << i
                 << " ,end of batch size inst_num_all. Is empty: "
                 << rob[i].is_empty() << endl;
#endif
          }
          newInst[i] = rob[i].add();
          if (!newInst[i]->read_sim_data(trace[i], aux_trace[i])) {
            eof[i] = true;
#ifdef DEBUG
            cout << "Trace: " << i << ", Instr: " << inst_num_all[i]
                 << ", eof true from read_train_data" << endl;
#endif
            rob[i].tail = rob[i].dec(rob[i].tail);
          }

          fetched[i]++;
          fetched_inst_num[i]++;
#pragma omp atomic
          count += 1;
          newInst[i]->inTick = curTick[i];
          if (curTick[i] != lastFetchTick[i]) {
            rob[i].update_fetch_cycle(curTick[i] - lastFetchTick[i],
                                      curTick[i]);
          }
          // cout<<input<<endl;
          int GPU_ID = (i) % nGPU;
          int offset = i / nGPU;
          float *inputPtr = input[GPU_ID].data_ptr<float>();
          inputPtr = inputPtr + ML_SIZE * offset;
          rob[i].make_input_data(inputPtr, curTick[i]);
#ifdef NGPU_DEBUG
#pragma omp critical
          {
            cout << "Trace: " << i << " GPU_ID: " << GPU_ID
                 << " offset: " << offset << " inputPtr: " << inputPtr << endl;
          }
#endif
          if ((fetched[i] == FETCH_BANDWIDTH)) {
            ROB_flag[i] = true;
#ifdef DEBUG
            cout << "Rob flagged" << endl;
#endif
          }
        } else {
          ROB_flag[i] = true;
#ifdef DEBUG
          cout << "Else condition" << endl;
#endif
        }
#ifdef DEBUG
        cout << "Count:" << count << endl;
#endif
      }
    }
    gettimeofday(&end, NULL);
    loop1_time +=
        end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    gettimeofday(&start, NULL);

    // Parallel inference
    /***********************************************************************/
    // cout<<input[0]<<endl;
#pragma omp parallel for
    for (i = 0; i < nGPU; i++) {
      std::vector<torch::jit::IValue> inputs;
      string dev = "cuda:";
      string id = to_string(i);
      dev = dev + id;
      const std::string device_string = dev;
      inputs.push_back(input[i].to(device_string));
#ifdef NGPU_DEBUG
#pragma omp critical
      {
        cout << " GPU_ID: " << i << " Input dim: " << input[i].sizes()
             << " JIT shape: " << inputs.size() << endl;
      }
#endif
      output[i] = lat_module[i].forward(inputs).toTensor();
      output[i] = output[i].to(at::kCPU);
#ifdef CLASSIFY
      cla_output[i] = cla_module[i].forward(inputs).toTensor();
      cla_output[i] = cla_output[i].to(at::kCPU);
#endif
    }
    gettimeofday(&end, NULL);

    // Aggregate results
    gettimeofday(&start, NULL);
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++) {
      if (!eof[i]) {
        int GPU_ID = (i) % nGPU;
        int offset = i / nGPU;
// cout<<"Trace: "<<i<<" "<<GPU_ID<<" "<<offset<<endl;
#ifdef CLASSIFY
        int f_class, c_class;
        float *cla_ptr = cla_output[GPU_ID].data_ptr<float>();
        for (int i = 0; i < 2; i++) {
          float max = cla_ptr[20 * offset + 10 * i];
          int idx = 0;
          for (int j = 1; j < 10; j++) {
            if (max < cla_ptr[20 * offset + 10 * i + j]) {
              max = cla_ptr[20 * offset + 10 * i + j];
              idx = j;
            }
          }
          if (i == 0)
            f_class = idx;
          else
            c_class = idx;
        }
#endif
        float fetch_lat, finish_lat;
        float *output_arr = output[GPU_ID].data_ptr<float>();
        fetch_lat = output_arr[2 * offset + 0] * factor[1] + mean[1];
        finish_lat = output_arr[2 * offset + 1] * factor[3] + mean[3];
#ifdef DEBUG_L
#pragma omp critical
        {
          cout << "Trace: " << i << ", GPU_ID: " << GPU_ID << ", offset"
               << offset << ", fetch: " << fetch_lat
               << ", finish: " << finish_lat << endl;
        }
#endif
        int int_fetch_lat = round(fetch_lat);
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
#endif
#ifdef ROB_ANALYSIS
        int_finish_lat = newInst[i]->trueCompleteTick;
        int_fetch_lat = newInst[i]->trueFetchTick;
#endif

#ifdef DEBUG
#endif
        newInst[i]->train_data[0] = (-int_fetch_lat - mean[0]) / factor[0];
        newInst[i]->train_data[1] = (-int_fetch_lat - mean[1]) / factor[1];
        newInst[i]->train_data[2] =
            (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
        if (newInst[i]->train_data[2] >= 9 / factor[2])
          newInst[i]->train_data[2] = 9 / factor[2];
        newInst[i]->train_data[3] = (int_finish_lat - mean[3]) / factor[3];
        newInst[i]->tickNum = int_finish_lat;
        newInst[i]->completeTick = curTick[i] + int_finish_lat + int_fetch_lat;
        lastFetchTick[i] = curTick[i];
#ifdef DEBUG_L
        // cout<<"newInst update"<<endl;
        cout << newInst[i]->train_data[0] << " " << newInst[i]->train_data[1]
             << " " << newInst[i]->train_data[2] << " "
             << newInst[i]->train_data[3] << " " << newInst[i]->tickNum << " "
             << newInst[i]->completeTick << " " << nextFetchTick[i] << endl;
#endif
        int_fetch_latency[i] = int_fetch_lat;
        int_finish_latency[i] = int_finish_lat;

        if (int_fetch_lat) {
          nextFetchTick[i] = curTick[i] + int_fetch_lat;
          ROB_flag[i] = true;
#ifdef DEBUG
          cout << "continue" << endl;
#endif
        }
        // cout<<"Trace: "<<i<<" update completed"<<endl;
      }
    }
#ifdef DEBUG_L
    if (fetched[0])
      cout << curTick[0] << " f " << fetched[0] << "\n";
#endif
    gettimeofday(&end, NULL);
    loop3_time +=
        end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Advance clock.
    //cout<<"Result updated"<<endl;
    /**********************************************************************/
    gettimeofday(&start,NULL);
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++) {
      if (!rob[i].is_empty()) {
        if (ROB_flag[i]) {
          if ((rob[i].is_full() || rob[i].saturated) && int_fetch_latency[i]) {
            // Fast forward curTick to the next cycle when it is able to fetch
            // and retire instructions.
            curTick[i] = max(rob[i].getHead()->completeTick, nextFetchTick[i]);
            if (rob[i].is_full())
              Case1++;
            else
              Case2++;
          } else if (int_fetch_latency[i]) {
            // Fast forward curTick to fetch instructions.
            curTick[i] = nextFetchTick[i];
            Case0++;
          } else if (rob[i].is_full() || rob[i].saturated) {
            // Fast forward curTick to retire instructions.
            curTick[i] = rob[i].getHead()->completeTick;
            if (rob[i].is_full())
              Case3++;
            else
              Case4++;
          } else {
            curTick[i]++;
            Case5++;
          }
          int_fetch_latency[i] = 0;
        }
      }
    }
#ifdef DEBUG_L
    cout << "curTick: " << curTick[0] << endl;
#endif
    gettimeofday(&end, NULL);
    loop4_time +=
        end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    // stop_flag -= 1;
    stop_flag = true;

    gettimeofday(&start, NULL);
    for (int i = 0; i < Total_Trace; i++) {
      // cout<< "eof[ " << i << "]= " << eof[i]<<endl;
      if (!eof[i] || !rob[i].is_empty()) {
        stop_flag = false;
        break;
      }
    }
    gettimeofday(&end, NULL);
    loop5_time += end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    #ifdef DEBUG
    for(int i=0 ; i< Total_Trace;i++)
    {
      cout<<"Trace:"<<i<<", Inst: "<< inst_num_all[i]<<", curTick: " << curTick[i] << ", Rob status: "<< rob[i].is_empty()<< endl;
    }
    #endif
    // cout<<"Stop flag: "<<stop_flag<<endl;
    // #ifdef DEBUG
    // cout<<"Stop flag:"<<stop_flag<<endl<<endl;
    // #endif
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  int inst_num = 0;
  Tick curTick_final = 0;
  //cout << "Time: " << total_time << "\n";
  for (i = 0; i < Total_Trace; i++) {
    // cout<<"Inst: "<<inst_num_all[i] <<". Tick: "<<curTick[i]<<endl;
    inst_num += inst_num_all[i];
    curTick_final += curTick[i];
    trace[i].close();
    aux_trace[i].close();
  }

  cout << inst_num << " instructions finish by " << (curTick_final - 1) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << endl;
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3
       << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[8] << "\n";
#else
  cout << "Lat Model: " << argv[3] << "\n";
#endif
  cout << "Threads: " << atoi(argv[6]) << " ,Batch: " << Total_Trace
       << " ,GPUs: " << nGPU << endl;
  // cout<<","<<atoi(argv[6])<<","<< Total_Trace <<","<< nGPU << ","<<
  // curTick_final << endl;
  return 0;
}
