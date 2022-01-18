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
  bool inSQ() { return (bool)train_data[INSQ_BIT]; }
  bool isStore() { return (bool)train_data[INSQ_BIT] || (bool)train_data[ATOMIC_BIT] || (bool)train_data[SC_BIT]; }
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
    assert(trueCompleteTick >= MIN_COMP_LAT || trueCompleteTick == 0);
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
    assert(!is_empty());
    return &insts[head];
  }
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  void retire_one(Queue *sq = nullptr) {
    if (sq) {
      if (insts[head].isStore()) {
        if (sq->is_full())
          sq->retire_one();
        Inst *newInst = sq->add();
        newInst->init(insts[head]);
      }
      retire();
    } else {
      retire();
    }
  }
  int retire_until(Tick tick, Queue *sq = nullptr) {
    int retired = 0;
    if (sq) {
      while (!is_empty() && insts[head].completeTick <= tick) {
        if (insts[head].isStore()) {
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

class SimModule {
public:
  ifstream trace;
  ifstream aux_trace;
  ofstream ipc_trace;
  bool  is_scale_pred = false;
  double scale_factor;

  unsigned long long total_num;
  unsigned long long fetched_inst_num = 0;
  Tick curTick = 0;
  Tick totalFetchTick = 0;
  Tick lastFetchTick = 0;
  bool eof = false;
  struct Queue *rob;
  struct Queue *sq;
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

  Inst *newInst;
  int int_fetch_lat;
  int int_complete_lat;
  int int_store_lat;

  bool init(const char *trace_name, const char *aux_trace_name, const char *model_cstr_name, unsigned long long in_total_num, double scale_fac = 0.0);
  void preprocess(float *inputPtr);
  void postprocess(float *output);
  void finish(const char *trace_name, const char *aux_trace_name);
};

bool SimModule::init(const char *trace_name, const char *aux_trace_name, const char *model_cstr_name, unsigned long long in_total_num, double scale_fac) {
  trace.open(trace_name);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return false;
  }
  aux_trace.open(aux_trace_name);
  if (!aux_trace.is_open()) {
    cerr << "Cannot open auxiliary trace file.\n";
    return false;
  }
  if (scale_fac > 0.0) {
    is_scale_pred = true;
    scale_factor = scale_fac;
    cout << "Scale prediction with " << scale_factor << "\n";
  }
#ifdef DUMP_IPC
  string ipc_trace_name = trace_name;
#ifdef RUN_TRUTH
  ipc_trace_name += "_true";
#else
  string model_name = model_cstr_name;
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
  ipc_trace.open(ipc_trace_name);
  if (!ipc_trace.is_open()) {
    cerr << "Cannot open ipc trace file.\n";
    return false;
  }
  cout << "Write IPC trace to " << ipc_trace_name << "\n";
#endif

  total_num = in_total_num;
  rob = new Queue(ROBSIZE);
  sq = new Queue(SQSIZE);
  return true;
}

void SimModule::preprocess(float *inputPtr) {
  // Retire instructions.
  int sq_retired = sq->retire_until(curTick);
  // Instruction retired from ROB need to enter SQ sometimes.
  int rob_retired = rob->retire_until(curTick, sq);
#ifdef DEBUG
  if (sq_retired || rob_retired)
    cout << curTick << " r " << rob_retired << " " << sq_retired << "\n";
#endif
  // Fetch instructions.
  int_fetch_lat = 0;
  if (rob->is_full()) {
    rob->retire_one(sq);
    Case1++;
  }
  newInst = rob->add();
  if (!newInst->read_sim_data(trace, aux_trace)) {
    eof = true;
    rob->tail = rob->dec(rob->tail);
    return;
  }
  newInst->inTick = curTick;
  // Construct input.
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
}

void SimModule::postprocess(float *output) {
#ifdef RUN_TRUTH
  int_fetch_lat = newInst->trueFetchTick;
  int int_complete_lat = newInst->trueCompleteTick;
  int int_store_lat = newInst->trueStoreTick;
#else
  float fetch_lat = output[0];
  float complete_lat = output[1];
  float store_lat = output[2];
  int_fetch_lat = round(fetch_lat);
  int_complete_lat = round(complete_lat);
  int_store_lat = round(store_lat);
#if defined(COMBINED)
  int classes[3];
  for (int i = 0; i < 3; i++) {
    float max = output[CLASS_NUM*i+3];
    int idx = 0;
    for (int j = 1; j < CLASS_NUM; j++) {
      if (max < output[CLASS_NUM*i+3+j]) {
        max = output[CLASS_NUM*i+3+j];
        idx = j;
      }
    }
    classes[i] = idx;
  }
  if (classes[0] <= 8)
    int_fetch_lat = classes[0];
  if (classes[1] <= 8)
    int_complete_lat = classes[1] + MIN_COMP_LAT;
  if (classes[2] == 0)
    int_store_lat = 0;
  else if (classes[2] <= 8 )
    int_store_lat = classes[2] + MIN_ST_LAT - 1;
#endif
#ifdef DUMP_ML_INPUT
  int_fetch_lat = newInst->trueFetchTick;
  int_complete_lat = newInst->trueCompleteTick;
  int_store_lat = newInst->trueStoreTick;
#endif
  if (is_scale_pred) {
    int_fetch_lat += round(((int)newInst->trueFetchTick - int_fetch_lat) * scale_factor);
    int_complete_lat += round(((int)newInst->trueCompleteTick - int_complete_lat) * scale_factor);
    int_store_lat += round(((int)newInst->trueStoreTick - int_store_lat) * scale_factor);
  }
  // Calibrate latency.
  if (int_fetch_lat < 0)
    int_fetch_lat = 0;
  if (int_complete_lat < MIN_COMP_LAT)
    int_complete_lat = MIN_COMP_LAT;
  if (!newInst->isStore()) {
    assert(newInst->trueStoreTick == 0);
    int_store_lat = 0;
  } else if (int_store_lat < MIN_ST_LAT)
    int_store_lat = MIN_ST_LAT;
  totalFetchDiff += (int)newInst->trueFetchTick - int_fetch_lat;
  totalAbsFetchDiff += abs((int)newInst->trueFetchTick - int_fetch_lat);
  totalCompleteDiff+= (int)newInst->trueCompleteTick - int_complete_lat;
  totalAbsCompleteDiff += abs((int)newInst->trueCompleteTick - int_complete_lat);
  totalStoreDiff += (int)newInst->trueStoreTick - int_store_lat;
  totalAbsStoreDiff += abs((int)newInst->trueStoreTick - int_store_lat);
  newInst->train_data[0] = -int_fetch_lat;
  newInst->train_data[1] = int_complete_lat;
  newInst->train_data[2] = int_store_lat;
#endif
  newInst->completeTick = curTick + int_fetch_lat + int_complete_lat + 1;
  newInst->storeTick = curTick + int_fetch_lat + int_store_lat;
  lastFetchTick = curTick;
  fetched_inst_num++;
  if (total_num && fetched_inst_num == total_num)
    eof = true;
#ifdef DUMP_IPC
  interval_fetch_lat += int_fetch_lat;
  if (fetched_inst_num % DUMP_IPC_INTERVAL == 0) {
    ipc_trace << interval_fetch_lat << "\n";
    interval_fetch_lat = 0;
  }
#endif
#ifdef DEBUG
  cout << curTick << " f\n";
#endif
  totalFetchTick += int_fetch_lat;
  if (eof) {
    curTick = rob->tail_inst().completeTick;
    if (!sq->is_empty())
      curTick = max(curTick, sq->tail_inst().storeTick);
    Case2++;
  } else {
    curTick += int_fetch_lat;
    Case0++;
  }
}

void SimModule::finish(const char *trace_name, const char *aux_trace_name) {
  trace.close();
  aux_trace.close();
#ifdef DUMP_IPC
  ipc_trace.close();
#endif
  unsigned long long inst_num = fetched_inst_num;
  cout << inst_num << " instructions finish by " << curTick << "\n";
  cout << "Fetch finish by " << totalFetchTick << " (err: " << -(double)totalFetchDiff / (totalFetchTick + totalFetchDiff)<< ") (true: " << totalFetchTick + totalFetchDiff << ")\n";
  cout << "Fetch Diff: " << totalFetchDiff << " (" << (double)totalFetchDiff / inst_num << " per inst), Absolute Diff: " << totalAbsFetchDiff << " (" << (double)totalAbsFetchDiff / inst_num << " per inst)\n";
  cout << "Complete Diff: " << totalCompleteDiff << " (" << (double)totalCompleteDiff / inst_num << " per inst, Absolute Diff: " << totalAbsCompleteDiff << " (" << (double)totalAbsCompleteDiff / inst_num << " per inst)\n";
  cout << "Store Diff: " << totalStoreDiff << " (" << (double)totalStoreDiff / inst_num << " per inst, Absolute Diff: " << totalAbsStoreDiff << " (" << (double)totalAbsStoreDiff / inst_num << " per inst)\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << trace_name << " " << aux_trace_name << "\n";
}
