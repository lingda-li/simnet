#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>

#include <torch/script.h> // One-stop header.

using namespace std;

//#define COMBINED
//#define CLASSIFY
//#define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU

#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TD_SIZE 50
#define ROBSIZE 94
#define SQSIZE 17
#define CONTEXTSIZE (ROBSIZE + SQSIZE)
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 8
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6
#define MIN_ST_LAT 10
#define CLASS_NUM 10

#define ILINEC_BIT 33
#define IPAGEC_BIT 37
#define DADDRC_BIT 41
#define DLINEC_BIT 42
#define DPAGEC_BIT 46

typedef long unsigned Tick;
typedef long unsigned Addr;

float default_val[ML_SIZE];

Addr getLine(Addr in) { return in & ~0x3f; }

struct Inst {
  float train_data[TD_SIZE];
  Tick inTick;
  Tick completeTick;
  Tick storeTick;
  Tick trueFetchTick;
  Tick trueCompleteTick;
  Tick trueStoreTick;
  Addr pc;
  int isAddr;
  Addr addr;
  Addr addrEnd;
  Addr iwalkAddr[3];
  Addr dwalkAddr[3];
  int inSQ() { return train_data[4]; }
  void init(Inst &copy) {
    std::copy(copy.train_data, copy.train_data + TD_SIZE, train_data);
    inTick = copy.inTick;
    completeTick = copy.completeTick;
    storeTick = copy.storeTick;
    pc = copy.pc;
    isAddr = copy.isAddr;
    addr = copy.addr;
    addrEnd = copy.addrEnd;
    std::copy(copy.iwalkAddr, copy.iwalkAddr+ 3, iwalkAddr);
    std::copy(copy.dwalkAddr, copy.dwalkAddr+ 3, dwalkAddr);
  }
  // Read simulation data.
  bool read_sim_data(ifstream &trace, ifstream &aux_trace) {
    trace >> trueFetchTick >> trueCompleteTick >> trueStoreTick;
    aux_trace >> pc;
    pc = getLine(pc);
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    assert(trueCompleteTick >= MIN_COMP_LAT);
    assert(trueStoreTick == 0 || trueStoreTick >= MIN_ST_LAT);
    for (int i = 3; i < TD_SIZE; i++) {
      trace >> train_data[i];
    }
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = 0.0;
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
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
      iwalkAddr[i] = new_inst.iwalkAddr[i];
      dwalkAddr[i] = new_inst.dwalkAddr[i];
    }
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
      insts[i].train_data[ILINEC_BIT] = insts[i].pc == pc ? 1.0 : 0.0;
      int conflict = 0;
      for (int j = 0; j < 3; j++) {
        if (insts[i].iwalkAddr[j] != 0 && insts[i].iwalkAddr[j] == iwalkAddr[j])
          conflict++;
      }
      insts[i].train_data[IPAGEC_BIT] = (float)conflict;
      insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr && addr <= insts[i].addrEnd) ? 1.0 : 0.0;
      insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].isAddr && getLine(addr) == getLine(insts[i].addr)) ? 1.0 : 0.0;
      conflict = 0;
      if (isAddr && insts[i].isAddr)
        for (int j = 0; j < 3; j++) {
          if (insts[i].dwalkAddr[j] != 0 && insts[i].dwalkAddr[j] == dwalkAddr[j])
            conflict++;
        }
      insts[i].train_data[DPAGEC_BIT] = (float)conflict;
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      num++;
    }
    return num;
  }
  void update_fetch_cycle(Tick tick, bool is_rob) {
    int i;
    if (is_rob) {
      assert(!is_empty());
      i = dec(dec(tail));
    } else {
      i = dec(tail);
    }
    for (; i != dec(head); i = dec(i)) {
      insts[i].train_data[0] += tick;
      assert(insts[i].train_data[0] >= 0.0);
    }
  }
};

int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc != 6) {
    cerr << "Usage: ./simulator_qq <trace> <aux trace> <lat module> <class module> <# inst>" << endl;
#else
  if (argc != 5) {
    cerr << "Usage: ./simulator_qq <trace> <aux trace> <module> <# inst>" << endl;
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
  torch::jit::script::Module lat_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    lat_module = torch::jit::load(argv[3]);
    lat_module.eval();
#ifdef GPU
    lat_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return 0;
  }
  int arg_idx = 4;
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
    return 0;
  }
#endif
  unsigned long long total_num = atol(argv[arg_idx++]);

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

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  while(!eof || !rob->is_empty()) {
    // Retire instructions.
    int sq_retired = sq->retire_until(curTick);
    // Instruction retired from ROB need to enter SQ sometimes.
    int rob_retired = rob->retire_until(curTick, sq);
    inst_num += rob_retired;
    //if (inst_num >= 10)
    //  break;
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
      if (!newInst->read_sim_data(trace, aux_trace)) {
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
      int_fetch_lat = newInst->trueFetchTick;
      int int_complete_lat = newInst->trueCompleteTick;
      int int_store_lat = newInst->trueStoreTick;
#else
      // Predict fetch, completion, and store latency.
      gettimeofday(&start, NULL);
      if (curTick != lastFetchTick) {
        rob->update_fetch_cycle(curTick - lastFetchTick, true);
        sq->update_fetch_cycle(curTick - lastFetchTick, false);
      }
      int rob_num = rob->make_input_data(inputPtr, *newInst, true, curTick);
      int sq_num = sq->make_input_data(inputPtr + rob_num * TD_SIZE, *newInst, false, curTick);
      int num = rob_num + sq_num;
      if (num < CONTEXTSIZE)
        std::copy(default_val, default_val + (CONTEXTSIZE - num) * TD_SIZE, inputPtr + num * TD_SIZE);
#ifdef DUMP_ML_INPUT
      cout << newInst->trueFetchTick << " " << newInst->trueCompleteTick << " " << newInst->trueStoreTick << " ";
      for (int i = 3; i < ML_SIZE; i++) {
        if (i % TD_SIZE == 0 && inputPtr[i + 3] == 0)
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
        float max = output[0][CLASS_NUM*i+3].item<float>();
#endif
        int idx = 0;
        for (int j = 1; j < CLASS_NUM; j++) {
#if defined(CLASSIFY)
          if (max < cla_output[0][CLASS_NUM*i+j].item<float>()) {
            max = cla_output[0][CLASS_NUM*i+j].item<float>();
            idx = j;
          }
#else
          if (max < output[0][CLASS_NUM*i+3+j].item<float>()) {
            max = output[0][CLASS_NUM*i+3+j].item<float>();
            idx = j;
          }
#endif
        }
        classes[i] = idx;
      }
#endif
      float fetch_lat = output[0][0].item<float>();
      float complete_lat = output[0][1].item<float>();
      float store_lat = output[0][2].item<float>();
      int_fetch_lat = round(fetch_lat);
      int int_complete_lat = round(complete_lat);
      int int_store_lat = round(store_lat);
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;
      if (int_complete_lat < MIN_COMP_LAT)
        int_complete_lat = MIN_COMP_LAT;
      if (int_store_lat < MIN_ST_LAT)
        int_store_lat = MIN_ST_LAT;
#if defined(CLASSIFY) || defined(COMBINED)
      if (classes[0] <= 8)
        int_fetch_lat = classes[0];
      if (classes[1] <= 8)
        int_complete_lat = classes[1] + MIN_COMP_LAT;
      if (classes[2] == 0)
        int_store_lat = 0;
      else if (classes[2] <= 8 )
        int_store_lat = classes[2] + MIN_ST_LAT - 1;
#endif
      if (!newInst->inSQ())
        int_store_lat = 0;
      //std::cout << curTick << ": ";
      //std::cout << " " << f_class << " " << fetch_lat << " " << int_fetch_lat << " " << newInst->trueFetchTick << " :";
      //std::cout << " " << c_class << " " << finish_lat << " " << int_finish_lat << " " << newInst->trueCompleteTick << '\n';
      totalFetchDiff += (long)newInst->trueFetchTick - int_fetch_lat;
      totalAbsFetchDiff += abs((long)newInst->trueFetchTick - int_fetch_lat);
      totalCompleteDiff+= (long)newInst->trueCompleteTick - int_complete_lat;
      totalAbsCompleteDiff += abs((long)newInst->trueCompleteTick - int_complete_lat);
      totalStoreDiff += (long)newInst->trueStoreTick - int_store_lat;
      totalAbsStoreDiff += abs((long)newInst->trueStoreTick - int_store_lat);
#ifdef DUMP_ML_INPUT
      int_fetch_lat = newInst->trueFetchTick;
      int_complete_lat = newInst->trueCompleteTick;
      int_store_lat = newInst->trueStoreTick;
#endif
      newInst->train_data[0] = -int_fetch_lat;
      newInst->train_data[1] = int_complete_lat;
      newInst->train_data[2] = int_store_lat;
#endif
      newInst->completeTick = curTick + int_fetch_lat + int_complete_lat + 1;
      newInst->storeTick = curTick + int_fetch_lat + int_store_lat;
      lastFetchTick = curTick;
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
      Case3++;
      if (curTick == nextFetchTick)
        Case3++;
      else
        Case4++;
    } else if (rob->is_full()) {
      curTick = max(rob->getHead()->completeTick, curTick + 1);
      Case2++;
    } else {
      assert(eof);
      curTick = max(rob->getHead()->completeTick, curTick + 1);
      Case5++;
    }
    int_fetch_lat = 0;
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace.close();
  aux_trace.close();
  time_t now = time(0);
  cout << "Finish at " << ctime(&now);
#ifdef RUN_TRUTH
  cout << "Truth" << "\n";
#endif
  cout << inst_num << " instructions finish by " << (curTick - 1) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Fetch Diff: " << totalFetchDiff << " (" << (double)totalFetchDiff / inst_num << " per inst), Absolute Diff: " << totalAbsFetchDiff << " (" << (double)totalAbsFetchDiff / inst_num << " per inst)\n";
  cout << "Complete Diff: " << totalCompleteDiff << " (" << (double)totalCompleteDiff / inst_num << " per inst, Absolute Diff: " << totalAbsCompleteDiff << " (" << (double)totalAbsCompleteDiff / inst_num << " per inst)\n";
  cout << "Store Diff: " << totalStoreDiff << " (" << (double)totalStoreDiff / inst_num << " per inst, Absolute Diff: " << totalAbsStoreDiff << " (" << (double)totalAbsStoreDiff / inst_num << " per inst)\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << argv[1] << " " << argv[2] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  cout << "Model: " << argv[3] << "\n";
#endif
  return 0;
}
