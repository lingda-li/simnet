#include <iostream>

using namespace std;

#define MAXREGNUM 8
#define ROBSIZE 100
#define CONTEXT_SIZE 96
#define TICK_STEP 500

typedef long unsigned Tick;
typedef long unsigned Addr;

struct Inst {
  int op;
  int isMicroOp;
  int isCondCtrl;
  int isUncondCtrl;
  int isSquashAfter;
  int isSerializeAfter;
  int isSerializeBefore;
  int isMisPredict;
  int isMemBar;
  int isQuiesce;
  int isNonSpeculative;
  int pcOffset;
  int srcNum;
  int destNum;
  int srcClass[MAXREGNUM];
  int srcIndex[MAXREGNUM];
  int destClass[MAXREGNUM];
  int destIndex[MAXREGNUM];
  int isAddr;
  Addr pc;
  Addr addr;
  unsigned int size;
  int depth;
  int fetchDepth;
  Addr addrEnd;
  int iwalkDepth[3];
  Addr iwalkAddr[3];
  int dwalkDepth[3];
  Addr dwalkAddr[3];
  Tick inTick;
  Tick outTick;
  Tick tickNum;
  // Read one instruction.
  bool read(ifstream &trace) {
    trace >> dec >> inTick >> tickNum >> outTick;
    if (trace.eof())
      return false;
    assert(inTick % TICK_STEP == tickNum % TICK_STEP == outTick % TICK_STEP ==
           0);
    inTick /= TICK_STEP;
    tickNum /= TICK_STEP;
    outTick /= TICK_STEP;
    // Read instruction type and etc.
    trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isSquashAfter >>
        isSerializeAfter >> isSerializeBefore >> isMisPredict;
    trace >> isMemBar >> isQuiesce >> isNonSpeculative >> pcOffset;
    combineOp();
    // Read source and destination registers.
    trace >> srcNum;
    for (int i = 0; i < srcNum; i++)
      trace >> srcClass[i] >> srcIndex[i];
    trace >> destNum;
    for (int i = 0; i < destNum; i++)
      trace >> destClass[i] >> destIndex[i];
    assert(srcNum <= MAXREGNUM && destNum <= MAXREGNUM);
    // Read data memory access info.
    trace >> isAddr;
    trace >> hex >> addr;
    trace >> dec >> size >> depth;
    if (isAddr)
      addrEnd = addr + size - 1;
    else
      depth = -1;
    for (int i = 0; i < 3; i++)
      trace >> dwalkDepth[i];
    for (int i = 0; i < 3; i++) {
      trace >> hex >> dwalkAddr[i];
      assert(dwalkAddr[i] == 0 || dwalkDepth[i] != -1);
    }
    assert(
        (dwalkDepth[0] == -1 && dwalkDepth[1] == -1 && dwalkDepth[2] == -1) ||
        isAddr);
    // Read instruction memory access info.
    trace >> hex >> pc;
    //cerr << hex << pc << endl;
    pc = pc & ~0x3f;
    trace >> dec >> fetchDepth;
    for (int i = 0; i < 3; i++)
      trace >> iwalkDepth[i];
    for (int i = 0; i < 3; i++) {
      trace >> hex >> iwalkAddr[i];
      assert(iwalkAddr[i] == 0 || iwalkDepth[i] != -1);
    }
    assert(!trace.eof());
    return true;
  }
  void combineOp() {
    if (isMisPredict) {
      assert(isCondCtrl || isUncondCtrl || isSquashAfter);
    }
    if (isMemBar) {
      assert(!isUncondCtrl && !isCondCtrl && !isSquashAfter &&
             !isSerializeBefore && !isQuiesce && !isNonSpeculative &&
             (op == 1 || op == 39 || op == 40));
      if (isSerializeAfter) {
        assert(op == 1);
        op = -5;
      } else
        op = -op;
    } else if (isSerializeBefore) {
      assert(!isUncondCtrl && !isCondCtrl && !isSquashAfter &&
             !isSerializeAfter && !isQuiesce && !isNonSpeculative && op == 1);
      op = -2;
    } else if (isSerializeAfter) {
      assert(!isUncondCtrl && !isCondCtrl && !isQuiesce && op == 1);
      assert(isNonSpeculative);
      if (isSquashAfter)
        op = -3;
      else
        op = -4;
    } else if (isSquashAfter) {
      assert(op == 1 && !isUncondCtrl && !isCondCtrl && !isQuiesce &&
             !isNonSpeculative);
      op = -6;
    } else if (isCondCtrl || isUncondCtrl) {
      assert(!isCondCtrl || !isUncondCtrl);
      assert(op == 1 && !isQuiesce && !isNonSpeculative);
      if (isCondCtrl)
        op = -7;
      else
        op = -8;
    }
  }
  void dump(Tick tick, bool first, int is_addr, Addr begin, Addr end, Addr PC,
            Addr *iwa, Addr *dwa) {
    assert(iwa && dwa);
    if (first)
      cout << inTick - tick;
    else
      cout << tick - inTick;
    cout << " " << tickNum << " ";
    cout << op << " " << isMicroOp << " " << isMisPredict << " ";
    cout << srcNum << " ";
    for (int i = 0; i < srcNum; i++)
      cout << srcClass[i] << " " << srcIndex[i] << " ";
    cout << destNum << " ";
    for (int i = 0; i < destNum; i++)
      cout << destClass[i] << " " << destIndex[i] << " ";
    // Instruction cache depth.
    cout << fetchDepth << " ";
    // Instruction cache line conflict.
    if (!first && pc == PC)
      cout << "1 ";
    else
      cout << "0 ";
    // PC offset
    cout << pcOffset << " ";
    // Instruction walk depth.
    for (int i = 0; i < 3; i++)
      cout << iwalkDepth[i] << " ";
    // Instruction page conflict.
    int iconflict = 0;
    if (!first)
      for (int i = 0; i < 3; i++) {
        if (iwalkAddr[i] != 0 && iwalkAddr[i] == iwa[i])
          iconflict++;
      }
    cout << iconflict << " ";
    // Data cache depth.
    cout << depth << " ";
    // Data address conflict.
    if (!first && is_addr && isAddr && end >= addr && begin <= addrEnd)
      cout << "1 ";
    else
      cout << "0 ";
    // Data cache line conflict.
    if (!first && is_addr && isAddr && (begin & ~0x3f) == (addr & ~0x3f))
      cout << "1 ";
    else
      cout << "0 ";
    // Data walk depth.
    for (int i = 0; i < 3; i++)
      cout << dwalkDepth[i] << " ";
    // Data page conflict.
    int dconflict = 0;
    if (!first && is_addr && isAddr)
      for (int i = 0; i < 3; i++) {
        if (dwalkAddr[i] != 0 && dwalkAddr[i] == dwa[i])
          dconflict++;
      }
    cout << dconflict << " ";
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
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  void retire_until(Tick tick) {
    while (!is_empty() && insts[head].outTick <= tick)
      retire();
  }
  void dump(Tick tick) {
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      insts[i].dump(tick, i == dec(tail), insts[dec(tail)].isAddr,
                    insts[dec(tail)].addr, insts[dec(tail)].addrEnd,
                    insts[dec(tail)].pc, insts[dec(tail)].iwalkAddr,
                    insts[dec(tail)].dwalkAddr);
    }
    cout << "\n";
  }
};

struct Context {
  Inst insts[CONTEXT_SIZE + 1];
  int head = 0;
  int tail = 0;
  int inc(int input) {
    if (input == CONTEXT_SIZE)
      return 0;
    else
      return input + 1;
  }
  int dec(int input) {
    if (input == 0)
      return CONTEXT_SIZE;
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
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  void dump(Tick tick) {
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      insts[i].dump(tick, false, 0, 0, 0, 0, NULL, NULL);
    }
    cout << "\n";
  }
};
