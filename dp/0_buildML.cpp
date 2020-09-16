#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>

#include "inst.h"
#include "queue.h"

using namespace std;

#define TICK_STEP 500
#define MINCOMPLETELAT 6
#define MINSTORELAT 6

Addr getLine(Addr in) { return in & ~0x3f; }
int getReg(int C, int I) { return C * MAXREGIDX + I + 1; }

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: ./buildML <trace> <SQ trace>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream sqtrace(argv[2]);
  if (!sqtrace.is_open()) {
    cerr << "Cannot open SQ trace file.\n";
    return 0;
  }

  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "ML");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
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
    q->dump(curTick, output);
    curTick = newInst->inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish at " << curTick << ".\n";
  trace.close();
  sqtrace.close();
  output.close();
  return 0;
}

bool Inst::read(ifstream &ROBtrace, ifstream &SQtrace) {
  ROBtrace >> dec >> sqIdx >> inTick >> completeTick >> outTick;
  if (ROBtrace.eof()) {
    int tmp;
    SQtrace >> tmp;
    assert(SQtrace.eof());
    return false;
  }
  ifstream *trace = &ROBtrace;
  int sqIdx2;
  Tick inTick2;
  int completeTick2, outTick2;
  if (sqIdx != -1) {
    SQtrace >> dec >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
        storeTick;
    if (SQtrace.eof())
      return false;
    trace = &SQtrace;
  }

  // Read instruction type and etc.
  *trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isDirectCtrl >>
      isSquashAfter >> isSerializeAfter >> isSerializeBefore;
  *trace >> isAtomic >> isStoreConditional >> isMemBar >> isQuiesce >>
      isNonSpeculative;
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
         (isNonSpeculative == 0 || isNonSpeculative == 1));
  assert(!inSQ() || (sqIdx2 == sqIdx && inTick2 == inTick &&
                     completeTick2 == completeTick && outTick2 == outTick));
  inTick /= TICK_STEP;
  completeTick /= TICK_STEP;
  outTick /= TICK_STEP;
  assert(completeTick >= MINCOMPLETELAT);
  if (sqIdx != -1) {
    storeTick /= TICK_STEP;
    assert(storeTick >= MINSTORELAT);
  } else
    storeTick = 0;

  // Read source and destination registers.
  *trace >> srcNum;
  for (int i = 0; i < srcNum; i++) {
    *trace >> srcClass[i] >> srcIndex[i];
    assert(srcClass[i] <= MAXREGCLASS);
    assert(srcClass[i] == MAXREGCLASS || srcIndex[i] < MAXREGIDX);
  }
  *trace >> destNum;
  for (int i = 0; i < destNum; i++) {
    *trace >> destClass[i] >> destIndex[i];
    assert(destClass[i] <= MAXREGCLASS);
    assert(destClass[i] == MAXREGCLASS || destIndex[i] < MAXREGIDX);
  }
  assert(srcNum <= SRCREGNUM && destNum <= DSTREGNUM);

  // Read data memory access info.
  *trace >> isAddr;
  *trace >> hex >> addr;
  *trace >> dec >> size >> depth;
  if (isAddr)
    addrEnd = addr + size - 1;
  else {
    addrEnd = 0;
    //depth = -1;
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

void Inst::dump(Tick tick, bool first, int is_addr, Addr begin, Addr end,
                Addr PC, Addr *iwa, Addr *dwa, ostream &out) {
  assert(first || (iwa && dwa));
  Tick fetchLat;
  if (first)
    fetchLat = inTick - tick;
  else
    fetchLat = tick - inTick;
  int fetchC, completeC, storeC;
  if (fetchLat <= 8)
    fetchC = fetchLat;
  else
    fetchC = 9;
  if (completeTick <= MINCOMPLETELAT + 8)
    completeC = completeTick - MINCOMPLETELAT;
  else
    completeC = 9;
  if (storeTick == 0)
    storeC = 0;
  else if (storeTick <= MINSTORELAT + 7)
    storeC = storeTick - MINSTORELAT + 1;
  else
    storeC = 9;
  out << fetchC << " " << fetchLat << " ";
  out << completeC << " " << completeTick << " ";
  out << storeC << " " << storeTick << " ";

  out << op + 1 << " " << isMicroOp << " " << isMisPredict << " " << isCondCtrl
       << " " << isUncondCtrl << " " << isDirectCtrl << " " << isSquashAfter
       << " " << isSerializeAfter << " " << isSerializeBefore << " " << isAtomic
       << " " << isStoreConditional << " " << isMemBar << " " << isQuiesce
       << " " << isNonSpeculative << " ";
  for (int i = 0; i < srcNum; i++)
    out << getReg(srcClass[i], srcIndex[i]) << " ";
  for (int i = srcNum; i < SRCREGNUM; i++)
    out << "0 ";
  for (int i = 0; i < destNum; i++)
    out << getReg(destClass[i], destIndex[i]) << " ";
  for (int i = destNum; i < DSTREGNUM; i++)
    out << "0 ";

  // Instruction cache depth.
  out << fetchDepth << " ";
  // Instruction cache line conflict.
  if (!first && getLine(pc) == getLine(PC))
    out << "1 ";
  else
    out << "0 ";
  // PC offset
  out << pc % 64 << " ";
  // Instruction walk depth.
  for (int i = 0; i < 3; i++)
    out << iwalkDepth[i] + 1 << " ";
  // Instruction page conflict.
  int iconflict = 0;
  if (!first)
    for (int i = 0; i < 3; i++) {
      if (iwalkAddr[i] != 0 && iwalkAddr[i] == iwa[i])
        iconflict++;
    }
  out << iconflict << " ";
  // Instruction cache writebacks.
  for (int i = 0; i < 2; i++)
    out << iWritebacks[i] << " ";

  // Data cache depth.
  out << depth << " ";
  // Data address conflict.
  if (!first && is_addr && isAddr && end >= addr && begin <= addrEnd)
    out << "1 ";
  else
    out << "0 ";
  // Data cache line conflict.
  if (!first && is_addr && isAddr && getLine(begin) == getLine(addr))
    out << "1 ";
  else
    out << "0 ";
  // Data walk depth.
  for (int i = 0; i < 3; i++)
    out << dwalkDepth[i] + 1 << " ";
  // Data page conflict.
  int dconflict = 0;
  if (!first && is_addr && isAddr)
    for (int i = 0; i < 3; i++) {
      if (dwalkAddr[i] != 0 && dwalkAddr[i] == dwa[i])
        dconflict++;
    }
  out << dconflict << " ";
  // Data cache writebacks.
  for (int i = 0; i < 3; i++)
    out << dWritebacks[i] << " ";
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
