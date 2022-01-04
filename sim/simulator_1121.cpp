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
//#define CLASSIFY
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

#define FETCH_LAT 0
#define COMPLETE_LAT 1
#define STORE_LAT 2
//#define IN_START 11
#define IN_START 3
#define INSQ_BIT (IN_START+1)
#define SC_BIT (IN_START+9)
#define ILINEC_BIT (IN_START+15)
#define DADDRC_BIT (IN_START+17)
#define DLINEC_BIT (IN_START+18)

#define TD_SIZE (IN_START+33)
#define CONTEXTSIZE (ROBSIZE + SQSIZE)
#define TICK_STEP 500.0
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define CLASS_NUM 10

typedef long unsigned Tick;
typedef long unsigned Addr;

float default_val[ML_SIZE];

Addr getLine(Addr in) { return in & ~0x3f; }

struct Inst {
  int targets[IN_START];
  float train_data[TD_SIZE];
  Tick inTick;
  Tick completeTick;
  Tick storeTick;
  Addr pc;
  int isAddr;
  Addr addr;
  Addr addrEnd;
  bool inSQ() {
    return (bool)train_data[INSQ_BIT];
  }
  bool isStore() {
    return (bool)train_data[INSQ_BIT] || (bool)train_data[SC_BIT];
  }
  void init(Inst &copy) {
    std::copy(copy.targets, copy.targets + IN_START, targets);
    std::copy(copy.train_data, copy.train_data + TD_SIZE, train_data);
    inTick = copy.inTick;
    completeTick = copy.completeTick;
    storeTick = copy.storeTick;
    pc = copy.pc;
    isAddr = copy.isAddr;
    addr = copy.addr;
    addrEnd = copy.addrEnd;
  }
  // Read simulation data.
  bool read_sim_data(ifstream &trace) {
    for (int i = 0; i < IN_START; i++) {
      trace >> targets[i];
      if (trace.eof()) {
        assert(i == 0);
        return false;
      }
      train_data[i] = 0.0;
    }
    assert(targets[COMPLETE_LAT] >= MIN_COMP_LAT || targets[COMPLETE_LAT] == 0);
#if IN_START != 3
    assert(targets[STORE_LAT] == 0 || targets[STORE_LAT]>= MIN_ST_LAT);
#endif
    for (int i = IN_START; i < TD_SIZE; i++) {
      trace >> train_data[i];
      //cout << train_data[i] << " ";
    }
    //cout << "\n";
    trace >> pc;
    pc = getLine(pc);
    trace >> isAddr >> addr >> addrEnd;
    assert(!trace.eof());
    return true;
  }
};

struct Queue {
  Inst *insts;
  int size;
  int head = 0;
  int tail = 0;
  Queue(int init_size) {
    size = init_size;
    insts = new Inst[size + 1];
  }
  int inc(int input) {
    if (input == size)
      return 0;
    else
      return input + 1;
  }
  int dec(int input) {
    if (input == 0)
      return size;
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
  Inst *getTail() {
    return &insts[dec(tail)];
  }
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  int retire_until(Tick tick, Queue *sq = nullptr) {
    int retired = 0;
    if (sq) {
      while (!is_empty() && insts[head].completeTick <= tick &&
             retired < RETIRE_BANDWIDTH) {
        if (insts[head].inSQ()) {
          if (sq->is_full())
            break;
          Inst *newInst = sq->add();
          newInst->init(insts[head]);
#if IN_START == 3
          newInst->storeTick += tick;
#endif
        }
        retire();
        retired++;
      }
    } else {
      while (!is_empty() && insts[head].storeTick <= tick) {
        retire();
        retired++;
      }
    }
    return retired;
  }
  Inst &tail_inst() { return insts[dec(tail)]; }

  int make_input_data(float *context, Inst &new_inst, bool is_rob, Tick tick) {
    Addr pc = new_inst.pc;
    int isAddr = new_inst.isAddr;
    Addr addr = new_inst.addr;
    Addr addrEnd = new_inst.addrEnd;
    int i;
    int num;
    if (is_rob) {
      assert(!is_empty());
      assert(&new_inst == &insts[dec(tail)]);
      std::copy(new_inst.train_data, new_inst.train_data + TD_SIZE, context);
      i = dec(dec(tail));
      num = 1;
    } else {
      i = dec(tail);
      num = 0;
    }
    for (; i != dec(head); i = dec(i)) {
      // Update context instruction bits.
      insts[i].train_data[FETCH_LAT] += tick;
      assert(insts[i].train_data[FETCH_LAT] >= 0.0);
      insts[i].train_data[ILINEC_BIT] = insts[i].pc == pc ? 1.0 : 0.0;
      insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr && addr <= insts[i].addrEnd) ? 1.0 : 0.0;
      insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].isAddr && getLine(addr) == getLine(insts[i].addr)) ? 1.0 : 0.0;
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      num++;
    }
    return num;
  }
};

int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc < 5) {
    cerr << "Usage: ./simulator_1121 <trace> <lat module> <class module> <# inst>" << endl;
#else
  if (argc < 4) {
    cerr << "Usage: ./simulator_1121 <trace> <module> <# inst>" << endl;
#endif
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  torch::jit::script::Module lat_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    lat_module = torch::jit::load(argv[2]);
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
  int arg_idx = 3;
#ifdef CLASSIFY
  torch::jit::script::Module cla_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    cla_module = torch::jit::load(argv[arg_idx++]);
    cla_module.eval();
#ifdef GPU
    cla_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    cerr << e.msg() << endl;
    return 0;
  }
#endif
  unsigned long long total_num = atol(argv[arg_idx++]);
  bool  is_scale_pred = false;
  double scale_factor;
  if (arg_idx < argc) {
    is_scale_pred = true;
    scale_factor = atof(argv[arg_idx++]);
    cout << "Scale prediction with " << scale_factor << "\n";
  }
#ifdef DUMP_IPC
  string ipc_trace_name = argv[1];
#ifdef RUN_TRUTH
  ipc_trace_name += "_true";
#else
  string model_name = argv[2];
  model_name.replace(0, 7, "_");
  model_name.replace(model_name.end()-3, model_name.end(), "");
  ipc_trace_name += model_name;
#endif
  time_t rawtime = time(0);
  struct tm* timeinfo = localtime(&rawtime);
  char buffer[80];
  strftime(buffer,sizeof(buffer),"%m%d%y",timeinfo);
  string time_str(buffer);
  ipc_trace_name += "_" + time_str + ".ipc";
  ofstream ipc_trace(ipc_trace_name);
  if (!ipc_trace.is_open()) {
    cerr << "Cannot open ipc trace file.\n";
    return 0;
  }
  cout << "Write IPC trace to " << ipc_trace_name << "\n";
#endif

  for (int i = 0; i < TD_SIZE; i++) {
    default_val[i] = 0;
  }
  for (int i = TD_SIZE; i < ML_SIZE; i++)
    default_val[i] = default_val[i % TD_SIZE];
  at::Tensor input = torch::ones({1, ML_SIZE});
  float *inputPtr = input.data_ptr<float>();
  std::vector<torch::jit::IValue> inputs;
  at::Tensor output;
#ifdef CLASSIFY
  at::Tensor cla_output;
#endif

  unsigned long long inst_num = 0;
  unsigned long long fetched_inst_num = 0;
  double measured_time = 0.0;
  Tick curTick = 0;
  Tick lastFetchTick = 0;
  bool eof = false;
  struct Queue *rob = new Queue(ROBSIZE);
  struct Queue *sq = new Queue(SQSIZE);
  Tick nextFetchTick = 0;
  long totalFetchDiff = 0;
  long totalAbsFetchDiff = 0;
  long totalCompleteDiff = 0;
  long totalAbsCompleteDiff = 0;
  long totalStoreDiff = 0;
  long totalAbsStoreDiff = 0;
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  Tick Case4 = 0;
  Tick Case5 = 0;
#ifdef DUMP_IPC
  int interval_fetch_lat = 0;
#endif

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  while(true) {
    // Retire instructions.
    int sq_retired = sq->retire_until(curTick);
    // Instruction retired from ROB need to enter SQ sometimes.
    int rob_retired = rob->retire_until(curTick, sq);
    inst_num += rob_retired;
#ifdef DEBUG
    if (sq_retired || rob_retired)
      cout << curTick << " r " << rob_retired << " " << sq_retired << "\n";
#endif
    // Fetch instructions.
    int fetched = 0;
    int int_fetch_lat;
    //while (fetched < FETCH_BANDWIDTH && !rob->is_full() && !eof) ??
    while (curTick >= nextFetchTick && !rob->is_full() && !eof) {
      Inst *newInst = rob->add();
      if (!newInst->read_sim_data(trace)) {
        eof = true;
        rob->tail = rob->dec(rob->tail);
        break;
      }
      fetched++;
      fetched_inst_num++;
      if (total_num && fetched_inst_num == total_num)
        eof = true;
      newInst->inTick = curTick;
#ifdef RUN_TRUTH
      int_fetch_lat = newInst->targets[FETCH_LAT];
      int int_complete_lat = newInst->targets[COMPLETE_LAT];
      int int_store_lat = newInst->targets[STORE_LAT];
#else
      // Predict fetch, completion, and store latency.
      gettimeofday(&start, NULL);
      int rob_num = rob->make_input_data(inputPtr, *newInst, true, curTick - lastFetchTick);
      int sq_num = sq->make_input_data(inputPtr + rob_num * TD_SIZE, *newInst, false, curTick - lastFetchTick);
      int num = rob_num + sq_num;
      if (num < CONTEXTSIZE)
        std::copy(default_val, default_val + (CONTEXTSIZE - num) * TD_SIZE, inputPtr + num * TD_SIZE);
#ifdef DUMP_ML_INPUT
      for (int i = 0; i < IN_START; i++) {
        cout << newInst->targets[i] << " ";
      }
      for (int i = IN_START; i < ML_SIZE; i++) {
        if (i % TD_SIZE == 0 && inputPtr[i + IN_START] == 0)
          break;
        int inttmp = round(inputPtr[i]);
        if (abs(inputPtr[i] - inttmp) > 0.001)
          cout << inputPtr[i] << " ";
        else
          cout << inttmp << " ";
      }
      cout << endl;
      //cout << input << "\n";
#endif
      inputs.clear();
#ifdef GPU
      inputs.push_back(input.cuda());
#else
      inputs.push_back(input);
#endif
      gettimeofday(&end, NULL);
      output = lat_module.forward(inputs).toTensor();
#ifdef CLASSIFY
      cla_output = cla_module.forward(inputs).toTensor();
#endif
      measured_time += (end.tv_sec - start.tv_sec) * 1000000.0 + end.tv_usec - start.tv_usec;
#if defined(CLASSIFY) || defined(COMBINED)
      int classes[3];
      for (int i = 0; i < 3; i++) {
#if defined(CLASSIFY)
        float max = cla_output[0][CLASS_NUM*i].item<float>();
#else
        float max = output[0][CLASS_NUM*i+IN_START].item<float>();
#endif
        int idx = 0;
        for (int j = 1; j < CLASS_NUM; j++) {
#if defined(CLASSIFY)
          if (max < cla_output[0][CLASS_NUM*i+j].item<float>()) {
            max = cla_output[0][CLASS_NUM*i+j].item<float>();
            idx = j;
          }
#else
          if (max < output[0][CLASS_NUM*i+IN_START+j].item<float>()) {
            max = output[0][CLASS_NUM*i+IN_START+j].item<float>();
            idx = j;
          }
#endif
        }
        classes[i] = idx;
      }
#endif
      float fetch_lat = output[0][FETCH_LAT].item<float>();
      float complete_lat = output[0][COMPLETE_LAT].item<float>();
      float store_lat = output[0][STORE_LAT].item<float>();
      int_fetch_lat = round(fetch_lat);
      int int_complete_lat = round(complete_lat);
      int int_store_lat = round(store_lat);
#if defined(CLASSIFY) || defined(COMBINED)
      if (classes[0] <= 8)
        int_fetch_lat = classes[0];
      if (classes[1] <= 8)
        int_complete_lat = classes[1] + MIN_COMP_LAT;
      if (classes[2] == 0)
        int_store_lat = 0;
      else if (classes[2] <= 8)
#if IN_START == 3
        int_store_lat = classes[2];
#else
        int_store_lat = classes[2] + MIN_ST_LAT - 1;
#endif
#endif
      int all_lats[IN_START];
      for (int i = 3; i < IN_START; i++) {
#ifdef DUMP_ML_INPUT
        all_lats[i] = newInst->targets[i];
#else
        float lat = output[0][i].item<float>();
        all_lats[i] = round(lat);
#endif
      }
      if (is_scale_pred) {
        int_fetch_lat += round(
            ((int)newInst->targets[FETCH_LAT] - int_fetch_lat) * scale_factor);
        int_complete_lat +=
            round(((int)newInst->targets[COMPLETE_LAT] - int_complete_lat) *
                  scale_factor);
        int_store_lat += round(
            ((int)newInst->targets[STORE_LAT] - int_store_lat) * scale_factor);
        for (int i = 3; i < IN_START; i++)
          all_lats[i] += round(
              ((int)newInst->targets[i] - all_lats[i]) * scale_factor);
      }
      // Calibrate latency.
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;
      if (int_complete_lat < MIN_COMP_LAT)
        int_complete_lat = MIN_COMP_LAT;
#if IN_START == 3
      if (!newInst->inSQ()) {
        assert(newInst->targets[STORE_LAT] == 0);
        int_store_lat = 0;
      } else if (int_store_lat < 0)
        int_store_lat = 0;
#else
      if (!newInst->isStore()) {
        assert(newInst->targets[STORE_LAT] == 0);
        int_store_lat = 0;
      } else if (int_store_lat < MIN_ST_LAT)
        int_store_lat = MIN_ST_LAT;
#endif
      for (int i = 3; i < IN_START; i++) {
        if (all_lats[i] < 0)
          all_lats[i] = 0;
      }
      totalFetchDiff += (int)newInst->targets[FETCH_LAT] - int_fetch_lat;
      totalAbsFetchDiff +=
          abs((int)newInst->targets[FETCH_LAT] - int_fetch_lat);
      totalCompleteDiff +=
          (int)newInst->targets[COMPLETE_LAT] - int_complete_lat;
      totalAbsCompleteDiff +=
          abs((int)newInst->targets[COMPLETE_LAT] - int_complete_lat);
      totalStoreDiff += (int)newInst->targets[STORE_LAT] - int_store_lat;
      totalAbsStoreDiff +=
          abs((int)newInst->targets[STORE_LAT] - int_store_lat);

#ifdef DUMP_ML_INPUT
      int_fetch_lat = newInst->targets[FETCH_LAT];
      int_complete_lat = newInst->targets[COMPLETE_LAT];
      int_store_lat = newInst->targets[STORE_LAT];
#endif
      newInst->train_data[FETCH_LAT] = -int_fetch_lat;
      newInst->train_data[COMPLETE_LAT] = int_complete_lat;
      newInst->train_data[STORE_LAT] = int_store_lat;
      for (int i = 3; i < IN_START; i++)
        newInst->train_data[i] = all_lats[i];
#endif
      newInst->completeTick = curTick + int_fetch_lat + int_complete_lat + 1;
#if IN_START == 3
      newInst->storeTick = int_store_lat;
#else
      newInst->storeTick = curTick + int_fetch_lat + int_store_lat;
#endif
      lastFetchTick = curTick;
#ifdef DUMP_IPC
      interval_fetch_lat += int_fetch_lat;
      if (fetched_inst_num % DUMP_IPC_INTERVAL == 0) {
        ipc_trace << interval_fetch_lat << "\n";
        interval_fetch_lat = 0;
      }
#endif
      if (int_fetch_lat) {
        nextFetchTick = curTick + int_fetch_lat;
        break;
      }
    }
#ifdef DEBUG
    if (fetched)
      cout << curTick << " f " << fetched << "\n";
#endif
    if (int_fetch_lat) {
      Tick nextCommitTick = max(rob->getHead()->completeTick, curTick + 1);
      curTick = min(nextCommitTick, nextFetchTick);
      if (curTick == nextFetchTick)
        Case0++;
      else
        Case1++;
    } else if (curTick < nextFetchTick) {
      Tick nextCommitTick = max(rob->getHead()->completeTick, curTick + 1);
      curTick = min(nextCommitTick, nextFetchTick);
      if (curTick == nextFetchTick)
        Case3++;
      else
        Case4++;
    } else if (rob->is_full()) {
      curTick = max(rob->getHead()->completeTick, curTick + 1);
      Case2++;
    } else {
      assert(eof);
      if (rob->is_empty()) {
        if (!sq->is_empty())
          curTick = sq->getTail()->storeTick;
        break;
      }
      curTick = max(rob->getHead()->completeTick, curTick + 1);
      Case5++;
    }
    int_fetch_lat = 0;
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec +
                      (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace.close();
#ifdef DUMP_IPC
  ipc_trace.close();
#endif
  time_t now = time(0);
  cout << "Finish at " << ctime(&now);
  cout << inst_num << " instructions finish by " << curTick << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Fetch Diff: " << totalFetchDiff << " ("
       << (double)totalFetchDiff / inst_num
       << " per inst), Absolute Diff: " << totalAbsFetchDiff << " ("
       << (double)totalAbsFetchDiff / inst_num << " per inst)\n";
  cout << "Complete Diff: " << totalCompleteDiff << " ("
       << (double)totalCompleteDiff / inst_num
       << " per inst, Absolute Diff: " << totalAbsCompleteDiff << " ("
       << (double)totalAbsCompleteDiff / inst_num << " per inst)\n";
  cout << "Store Diff: " << totalStoreDiff << " ("
       << (double)totalStoreDiff / inst_num
       << " per inst, Absolute Diff: " << totalAbsStoreDiff << " ("
       << (double)totalAbsStoreDiff / inst_num << " per inst)\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3
       << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#if defined(RUN_TRUTH)
  cout << "Truth" << "\n";
#elif defined(CLASSIFY)
  cout << "Model: " << argv[2] << " " << argv[3] << "\n";
#elif defined(COMBINED)
  cout << "Combined Model: " << argv[2] << "\n";
#else
  cout << "Latency Model: " << argv[2] << "\n";
#endif
  return 0;
}
