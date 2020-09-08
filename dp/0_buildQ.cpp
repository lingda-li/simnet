#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

#include "inst.h"
#include "queue.h"

using namespace std;

#define TICK_STEP 500

Addr getLine(Addr in) { return in & ~0x3f; }

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: ./buildROB <trace> <SQ trace>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream sqtrace(argv[2]);
  if (!trace.is_open()) {
    cerr << "Cannot open SQ trace file.\n";
    return 0;
  }

  // Current context.
  struct QUEUE *q = new QUEUE;
  Tick curTick;
  bool firstInst = true;
  Tick num = 0;
  while (!trace.eof()) {
    Inst *newInst = q->add();
    if (!newInst->read(trace, sqtrace))
      break;
    if (firstInst) {
      firstInst = false;
      curTick = newInst->inTick;
    }
    q->retire_until(curTick);
    //newInst->dump(curTick);
    q->dump(curTick);
    curTick = newInst->inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish at " << curTick << ".\n";
  trace.close();
  return 0;
}

bool Inst::read(ifstream &ROBtrace, ifstream &SQtrace) {
  ROBtrace >> dec >> sqIdx >> inTick >> completeTick >> outTick;
  ifstream *trace = &ROBtrace;
  int sqIdx2;
  Tick inTick2;
  int completeTick2, outTick2;
  if (sqIdx != -1) {
    SQtrace >> dec >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
        storeTick;
    trace = &SQtrace;
  }
  if (ROBtrace.eof()) {
    int tmp;
    if (sqIdx == -1)
      SQtrace >> tmp;
    assert(SQtrace.eof());
    return false;
  }

  // Read instruction type and etc.
  *trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isDirectCtrl >>
      isSquashAfter >> isSerializeAfter >> isSerializeBefore;
  *trace >> isAtomic >> isStoreConditional >> isMemBar >> isQuiesce >>
      isNonSpeculative;
  //fprintf(stderr, "t %d %lu %lu %lu : %d %lu %lu %lu\n", sqIdx, inTick, completeTick, outTick, sqIdx2, inTick2, completeTick2, outTick2);
  assert(!inSQ() || (sqIdx2 == sqIdx && inTick2 == inTick &&
                     completeTick2 == completeTick && outTick2 == outTick));
  inTick /= TICK_STEP;
  completeTick /= TICK_STEP;
  outTick /= TICK_STEP;
  if (sqIdx != -1)
    storeTick /= TICK_STEP;
  else
    storeTick = 0;
  combineOp();

  // Read source and destination registers.
  *trace >> srcNum;
  for (int i = 0; i < srcNum; i++)
    *trace >> srcClass[i] >> srcIndex[i];
  *trace >> destNum;
  for (int i = 0; i < destNum; i++)
    *trace >> destClass[i] >> destIndex[i];
  assert(srcNum <= SRCREGNUM && destNum <= DSTREGNUM);

  // Read data memory access info.
  *trace >> isAddr;
  *trace >> hex >> addr;
  *trace >> dec >> size >> depth;
  if (isAddr)
    addrEnd = addr + size - 1;
  else {
    addrEnd = 0;
    depth = -1;
  }
  for (int i = 0; i < 3; i++)
    *trace >> dwalkDepth[i];
  assert((dwalkDepth[0] == -1 && dwalkDepth[1] == -1 && dwalkDepth[2] == -1) ||
         isAddr);
  for (int i = 0; i < 3; i++) {
    *trace >> hex >> dwalkAddr[i];
    assert(dwalkAddr[i] == 0 || dwalkDepth[i] != -1);
  }
  for (int i = 0; i < 3; i++)
    *trace >> dec >> dWritebacks[i];

  // Read instruction memory access info.
  *trace >> hex >> pc;
  // cerr << hex << pc << endl;
  //pc = pc & ~0x3f;
  *trace >> dec >> isMisPredict >> fetchDepth;
  for (int i = 0; i < 3; i++)
    *trace >> iwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    *trace >> hex >> iwalkAddr[i];
    assert(iwalkAddr[i] == 0 || iwalkDepth[i] != -1);
  }
  for (int i = 0; i < 2; i++)
    *trace >> dec >> iWritebacks[i];
  assert(!ROBtrace.eof() && !SQtrace.eof());
  return true;
}

void printOP(Inst *i) {
  fprintf(stderr, "OP: %d %d %d %d %d %d %d : %d %d %d %d %d %d\n", i->op,
          i->isUncondCtrl, i->isCondCtrl, i->isDirectCtrl, i->isSquashAfter,
          i->isSerializeBefore, i->isSerializeAfter, i->isAtomic,
          i->isStoreConditional, i->isQuiesce, i->isNonSpeculative,
          i->isMemBar, i->isMisPredict);
}

void Inst::combineOp() {
  assert((isMicroOp == 0 || isMicroOp == 1) &&
         (isCondCtrl == 0 || isCondCtrl == 1) &&
         (isUncondCtrl == 0 || isUncondCtrl == 1) &&
         (isDirectCtrl == 0 || isDirectCtrl == 1) &&
         (isSquashAfter == 0 || isSquashAfter == 1) &&
         (isSerializeAfter == 0 || isSerializeAfter == 1) &&
         (isSerializeBefore == 0 || isSerializeBefore == 1) &&
         (isAtomic == 0 || isAtomic == 1) &&
         (isStoreConditional == 0 || isStoreConditional == 1) &&
         (isMemBar == 0 || isMemBar == 1) &&
         (isQuiesce == 0 || isQuiesce == 1) &&
         (isNonSpeculative == 0 || isNonSpeculative == 1) &&
         (isMisPredict == 0 || isMisPredict == 1));
  if (isAtomic)
    printOP(this);
  if (isMemBar) {
    assert(!isUncondCtrl && !isCondCtrl && !isDirectCtrl && !isSquashAfter &&
           !isSerializeBefore && !isQuiesce && !isNonSpeculative &&
           (op == 1 || op == 47 || op == 48));
    if (op == 1) {
      assert(!isAtomic && !isStoreConditional);
      if (isSerializeAfter)
        op = -1;
      else
        op = -2;
    } else if (op == 47) {
      assert(!isAtomic && !isStoreConditional);
      op = -3;
    } else {
      assert(op == 48 && !isAtomic);
      if (!isStoreConditional)
        op = -4;
      else
        op = -5;
    }
  } else if (isStoreConditional) {
    assert(!isUncondCtrl && !isCondCtrl && !isDirectCtrl && !isSquashAfter &&
           !isSerializeAfter && !isSerializeBefore && !isAtomic && !isQuiesce &&
           !isNonSpeculative && op == 48);
    op = -6;
  } else if (isSerializeBefore) {
    assert(!isUncondCtrl && !isCondCtrl && !isDirectCtrl && !isSquashAfter &&
           !isSerializeAfter && !isAtomic && !isStoreConditional &&
           !isQuiesce && !isNonSpeculative && op == 1);
    op = -7;
  } else if (isSerializeAfter) {
    assert(!isUncondCtrl && !isCondCtrl && !isDirectCtrl && !isAtomic &&
           !isStoreConditional && !isQuiesce && op == 1);
    assert(isNonSpeculative);
    if (isSquashAfter)
      op = -8;
    else
      op = -9;
  } else if (isSquashAfter) {
    assert(op == 1 && !isUncondCtrl && !isCondCtrl && !isDirectCtrl &&
           !isAtomic && !isStoreConditional && !isQuiesce && !isNonSpeculative);
    op = -10;
  } else if (isCondCtrl || isUncondCtrl || isDirectCtrl) {
    assert(!isCondCtrl || !isUncondCtrl);
    assert(op == 1 && !isAtomic && !isStoreConditional && !isQuiesce &&
           !isNonSpeculative);
    if (isDirectCtrl) {
      if (isCondCtrl)
        op = -11;
      else
        op = -12;
    } else {
      if (isCondCtrl)
        op = -13;
      else
        op = -14;
    }
  } else
    assert(!isAtomic);
}

void Inst::dump(Tick tick, bool first, int is_addr, Addr begin, Addr end,
                Addr PC, Addr *iwa, Addr *dwa) {
  assert(first || (iwa && dwa));
  if (first)
    cout << inTick - tick;
  else
    cout << tick - inTick;
  cout << " " << completeTick << " " << storeTick << " ";
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
  if (!first && getLine(pc) == getLine(PC))
    cout << "1 ";
  else
    cout << "0 ";
  // PC offset
  cout << pc % 64 << " ";
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
  // Instruction cache writebacks.
  for (int i = 0; i < 2; i++)
    cout << iWritebacks[i] << " ";

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
  // Data cache writebacks.
  for (int i = 0; i < 3; i++)
    cout << dWritebacks[i] << " ";
}

void Inst::dumpSim() {
  cout << pc << " ";
  cout << isAddr << " ";
  cout << addr << " ";
  cout << addrEnd << " ";
  // Instruction walk addrs.
  for (int i = 0; i < 3; i++)
    cout << iwalkAddr[i] << " ";
  // Data walk addrs.
  for (int i = 0; i < 3; i++)
    cout << dwalkAddr[i] << " ";
  cout << "\n";
}
