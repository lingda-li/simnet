#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

#include "inst.h"
#include "queue.h"

using namespace std;

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

  // Current ROB context.
  struct ROB *rob = new ROB;
  Tick curTick;
  bool firstInst = true;
  Tick num = 0;
  while (!trace.eof()) {
    Inst *newInst = rob->add();
    if (!newInst->read(trace, sqtrace))
      break;
    if (firstInst) {
      firstInst = false;
      curTick = newInst->inTick;
    }
    rob->retire_until(curTick);
    //newInst->dump(curTick);
    rob->dump(curTick);
    curTick = newInst->inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish at " << curTick << ".\n";
  trace.close();
  return 0;
}

bool Inst::read(ifstream &trace, ifstream &SQtrace) {
  trace >> dec >> sqIdx >> inTick >> completeTick >> outTick;
  if (sqIdx != -1) {
    int sqIdx2;
    Tick inTick2, completeTick2, outTick2;
    SQtrace >> dec >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
        storeTick;
    assert(trace.eof() ||
           (sqIdx2 == sqIdx && inTick2 == inTick &&
            completeTick2 == completeTick && outTick2 == outTick));
  }
  if (trace.eof()) {
    assert(SQtrace.eof());
    return false;
  }

  inTick /= TICK_STEP;
  completeTick /= TICK_STEP;
  outTick /= TICK_STEP;
  if (sqIdx != -1)
    storeTick /= TICK_STEP;

  // Read instruction type and etc.
  trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isDirectCtrl >>
      isSquashAfter >> isSerializeAfter >> isSerializeBefore;
  trace >> isAtomic >> isStoreConditional >> isMemBar >> isQuiesce >>
      isNonSpeculative;
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
  else {
    addrEnd = 0;
    depth = -1;
  }
  for (int i = 0; i < 3; i++)
    trace >> dwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    trace >> hex >> dwalkAddr[i];
    assert(dwalkAddr[i] == 0 || dwalkDepth[i] != -1);
  }
  for (int i = 0; i < 2; i++)
    trace >> dWritebacks[i];
  assert((dwalkDepth[0] == -1 && dwalkDepth[1] == -1 && dwalkDepth[2] == -1) ||
         isAddr);

  // Read instruction memory access info.
  trace >> hex >> pc;
  // cerr << hex << pc << endl;
  pc = pc & ~0x3f;
  trace >> dec >> fetchDepth;
  for (int i = 0; i < 3; i++)
    trace >> iwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    trace >> hex >> iwalkAddr[i];
    assert(iwalkAddr[i] == 0 || iwalkDepth[i] != -1);
  }
  for (int i = 0; i < 2; i++)
    trace >> iWritebacks[i];
  assert(!trace.eof());
  return true;
}

void Inst::combineOp() {
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

void Inst::dump(Tick tick, bool first, int is_addr, Addr begin, Addr end,
                Addr PC, Addr *iwa, Addr *dwa) {
  assert(first || (iwa && dwa));
  if (first)
    cout << inTick - tick;
  else
    cout << tick - inTick;
  cout << " " << completeTick << " ";
  if (sqIdx != -1)
    cout << storeTick << " ";
  else
    cout << "0 ";
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
  for (int i = 0; i < 2; i++)
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
