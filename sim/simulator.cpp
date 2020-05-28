#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>

#include <torch/script.h> // One-stop header.

using namespace std;

//#define CLASSIFY
//#define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU

#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define ROBSIZE 94
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define TD_SIZE 39
#define ML_SIZE (TD_SIZE * ROBSIZE)
#define MIN_COMP_LAT 6

#define ILINEC_BIT 8
#define IPAGEC_BIT 13
#define DADDRC_BIT 17
#define DLINEC_BIT 18
#define DPAGEC_BIT 22

typedef long unsigned Tick;
typedef long unsigned Addr;

Tick Num = 0;

float factor[TD_SIZE] = {
    4.312227940421713,
    23.77553044744847,
    4.555194111127984,
    35.2434314352038,
    15.265764787317297,
    0.29948663397402053,
    0.03179227967763891,
    1.0,
    0.3734605619940018,
    20.161017665013475,
    0.01824348838297176,
    0.0182174151577969,
    0.018094302417033036,
    0.0034398387559557687,
    1.0,
    1.0,
    0.38473787518201347,
    0.04555131694662134,
    0.09744266378500231,
    0.12392501424650113,
    0.12391492952272068,
    0.1238875588320556,
    0.06099769717651739,
    1.0,
    1.0,
    24.855520819765943,
    26.09130907621545,
    24.95569646402784,
    9.621855610687513,
    5.900770232974813,
    4.979413184960278,
    4.993246078979618,
    0.5821067191490825,
    14.657204204150103,
    22.100692970367547,
    22.241366163707752,
    20.590947782553783,
    2.58817823205487,
    0.13659586939272889
};

float mean[TD_SIZE];

float default_val[TD_SIZE];

struct Inst {
  int op;
  int isMicroOp;
  int isCondCtrl;
  int isUncondCtrl;
  int isMemBar;
  int srcNum;
  int destNum;
  int srcClass[MAXSRCREGNUM];
  int srcIndex[MAXSRCREGNUM];
  int destClass[MAXDSTREGNUM];
  int destIndex[MAXDSTREGNUM];
  Tick inTick;
  Tick completeTick;
  Tick tickNum;
  float train_data[TD_SIZE];
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
  // Read one instruction.
  bool read(ifstream &trace) {
    //Num++;
    //if (Num > 1000000)
    //  return false;
    Tick tmp;
    trace >> op >> tmp >> tmp >> tmp;
    if (trace.eof())
      return false;
    trace >> srcNum;
    for (int i = 0; i < srcNum; i++)
      trace >> srcClass[i] >> srcIndex[i];
    trace >> destNum;
    for (int i = 0; i < destNum; i++)
      trace >> destClass[i] >> destIndex[i];
    assert(!trace.eof());
    return true;
  }
  bool read_train_data(ifstream &trace, ifstream &aux_trace) {
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++)
      trace >> train_data[i];
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
  void dump(Tick tick) {
    cout << op << " " << tick - inTick << " " << tickNum << " ";
    cout << srcNum << " ";
    for (int i = 0; i < srcNum; i++)
      cout << srcClass[i] << " " << srcIndex[i] << " ";
    cout << destNum << " ";
    for (int i = 0; i < destNum; i++)
      cout << destClass[i] << " " << destIndex[i] << " ";
  }
};

struct ROB {
  Inst insts[ROBSIZE + 1];
  int head = 0;
  int tail = 0;
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
    while (!is_empty() && insts[head].completeTick <= tick &&
           retired < RETIRE_BANDWIDTH) {
      retire();
      retired++;
    }
    return retired;
  }
  void dump(Tick tick) {
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      insts[i].dump(tick);
    }
    cout << "\n";
  }
  void make_train_data(float *context) {
    int num = 0;
    Addr pc = insts[dec(tail)].pc;
    int isAddr = insts[dec(tail)].isAddr;
    Addr addr = insts[dec(tail)].addr;
    Addr addrEnd = insts[dec(tail)].addrEnd;
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
      iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
      dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
    }
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      if (i != dec(tail)) {
        // Update context instruction bits.
        insts[i].train_data[ILINEC_BIT] = insts[i].pc == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
        int conflict = 0;
        for (int j = 0; j < 3; j++) {
          if (insts[i].iwalkAddr[j] != 0 && insts[i].iwalkAddr[j] == iwalkAddr[j])
            conflict++;
        }
        insts[i].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
        insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr && addr <= insts[i].addrEnd) ? 1.0 / factor[DADDRC_BIT] : 0.0;
        insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].isAddr && (addr & ~0x3f) == (insts[i].addr & ~0x3f)) ? 1.0 / factor[DLINEC_BIT] : 0.0;
        conflict = 0;
        if (isAddr && insts[i].isAddr)
          for (int j = 0; j < 3; j++) {
            if (insts[i].dwalkAddr[j] != 0 && insts[i].dwalkAddr[j] == dwalkAddr[j])
              conflict++;
          }
        insts[i].train_data[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
      }
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      num++;
    }
    for (int i = num; i < ROBSIZE; i++) {
      std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
    }
  }
  void update_fetch_cycle(Tick tick) {
    for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
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

int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc < 5) {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <class module> <variances (optional)>" << endl;
#else
  if (argc < 4) {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <variances (optional)> " << endl;
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
#ifdef GPU
    lat_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return 0;
  }
#ifdef CLASSIFY
  torch::jit::script::Module cla_module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    cla_module = torch::jit::load(argv[4]);
#ifdef GPU
    cla_module.to(torch::kCUDA);
#endif
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return 0;
  }
#endif

  float *varPtr = NULL;
#ifdef CLASSIFY
  if (argc > 5)
    varPtr = read_numbers(argv[5], TD_SIZE);
#else
  if (argc > 4)
    varPtr = read_numbers(argv[4], TD_SIZE);
#endif
  if (varPtr)
    cout << "Use input factors.\n";

  for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    if (varPtr)
      factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i];
    cout << default_val[i] << " ";
  }
  cout << "\n";
  at::Tensor input = torch::ones({1, ML_SIZE});
  float *inputPtr = input.data_ptr<float>();

  unsigned long long inst_num = 0;
  double measured_time = 0.0;
  Tick curTick = 0;
  Tick lastFetchTick = 0;
  bool eof = false;
  struct ROB *rob = new ROB;
  Tick nextFetchTick = 0;
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  while(!eof || !rob->is_empty()) {
    // Retire instructions.
    int retired = rob->retire_until(curTick);
    inst_num += retired;
    //if (inst_num >= 10)
    //  break;
    //if (inst_num)
    //  cout << ".";
#ifdef DEBUG
    if (retired)
      cout << curTick << " " << retired << "\n";
#endif
    // Fetch instructions.
    int fetched = 0;
    int int_fetch_lat;
    while (fetched < FETCH_BANDWIDTH && !rob->is_full() && !eof) {
      Inst *newInst = rob->add();
      if (!newInst->read_train_data(trace, aux_trace)) {
        eof = true;
        rob->tail = rob->dec(rob->tail);
        break;
      }
      fetched++;
      newInst->inTick = curTick;
#ifdef RUN_TRUTH
      int int_finish_lat = newInst->trueCompleteTick;
      int_fetch_lat = newInst->trueFetchTick;
#else
      // Predict fetch and completion time.
      gettimeofday(&start, NULL);
      std::vector<torch::jit::IValue> inputs;
      //input.data_ptr<c10::ScalarType::Float>();
      if (curTick != lastFetchTick) {
        rob->update_fetch_cycle(curTick - lastFetchTick);
      }
      rob->make_train_data(inputPtr);
#ifdef DUMP_ML_INPUT
      cout << input << "\n";
#endif
#ifdef GPU
      inputs.push_back(input.cuda());
#else
      inputs.push_back(input);
#endif
      gettimeofday(&end, NULL);
      at::Tensor output = lat_module.forward(inputs).toTensor();
#ifdef CLASSIFY
      at::Tensor cla_output = cla_module.forward(inputs).toTensor();
#endif
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
      float fetch_lat = output[0][0].item<float>() * factor[1] + mean[1];
      float finish_lat = output[0][1].item<float>() * factor[3] + mean[3];
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
#endif
      newInst->tickNum = int_finish_lat;
      newInst->completeTick = curTick + int_finish_lat + int_fetch_lat;
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
    if (rob->is_full() && int_fetch_lat) {
      // Fast forward curTick to the next cycle when it is able to fetch and retire instructions.
      curTick = max(rob->getHead()->completeTick, nextFetchTick);
      Case0++;
    } else if (rob->is_full()) {
      // Fast forward curTick to retire instructions.
      curTick = rob->getHead()->completeTick;
      Case1++;
    } else if (int_fetch_lat) {
      // Fast forward curTick to fetch instructions.
      curTick = nextFetchTick;
      Case2++;
    } else {
      curTick++;
      Case3++;
    }
    int_fetch_lat = 0;
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace.close();
  aux_trace.close();
  cout << inst_num << " instructions finish by " << (curTick - 1) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  cout << "Lat Model: " << argv[3] << "\n";
#endif
#ifdef RUN_TRUTH
  cout << "Truth" << "\n";
#endif
  return 0;
}
