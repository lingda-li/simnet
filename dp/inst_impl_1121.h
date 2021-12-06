#ifndef __INST_IMPL_Q_H__
#define __INST_IMPL_Q_H__

#include <iostream>

using namespace std;

#define TICK_STEP 500
Tick minCompleteLat = 100;
Tick minStoreLat = 100;

Addr getLine(Addr in) { return in & ~0x3f; }
int getReg(int C, int I) { return C * MAXREGIDX + I + 1; }

bool Inst::read(ifstream &ROBtrace, ifstream &SQtrace) {
  ROBtrace >> sqIdx >> inTick >> completeTick >> outTick;
  ROBtrace >> decodeTick >> renameTick >> dispatchTick >> issueTick;
  if (ROBtrace.eof()) {
    int tmp;
    SQtrace >> tmp;
    assert(SQtrace.eof());
    return false;
  }
  ifstream *trace = &ROBtrace;
  int sqIdx2;
  Tick inTick2;
  int completeTick2, outTick2, decodeTick2, renameTick2, dispatchTick2,
      issueTick2;
  if (sqIdx != -1) {
    SQtrace >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
        decodeTick2 >> renameTick2 >> dispatchTick2 >> issueTick2 >>
        storeTick >> sqOutTick;
    if (SQtrace.eof())
      return false;
    assert(sqIdx2 == sqIdx && inTick2 == inTick && decodeTick2 == decodeTick &&
           renameTick2 == renameTick && dispatchTick2 == dispatchTick &&
           issueTick2 == issueTick);
    assert(sqOutTick >= storeTick);
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
         (isAtomic == 0) &&
         (isStoreConditional == 0 || isStoreConditional == 1) &&
         (isMemBar == 0 || isMemBar == 1) &&
         (isQuiesce == 0) &&
         (isNonSpeculative == 0 || isNonSpeculative == 1));
  assert(!inSQ() || (completeTick2 == completeTick && outTick2 == outTick &&
                     storeTick >= outTick));

  if (issueTick == (Tick)(-1)) {
    assert(dispatchTick == completeTick);
    issueTick = dispatchTick;
  }
  assert(outTick >= completeTick && completeTick >= issueTick &&
         issueTick >= dispatchTick && dispatchTick >= renameTick &&
         renameTick >= decodeTick);
  inTick /= TICK_STEP;
  completeTick /= TICK_STEP;
  outTick /= TICK_STEP;
  decodeTick /= TICK_STEP;
  renameTick /= TICK_STEP;
  dispatchTick /= TICK_STEP;
  issueTick /= TICK_STEP;
  if (completeTick < minCompleteLat)
    minCompleteLat = completeTick;
  if (sqIdx != -1) {
    storeTick /= TICK_STEP;
    sqOutTick /= TICK_STEP;
    if (storeTick < minStoreLat)
      minStoreLat = storeTick;
  } else {
    storeTick = 0;
    sqOutTick = 0;
  }

  // Read data memory access info.
  *trace >> isAddr;
  *trace >> addr;
  *trace >> size >> depth;
  if (depth == -1)
    depth = 0;
  if (isAddr)
    addrEnd = addr + size - 1;
  else {
    addrEnd = 0;
    //depth = -1;
  }
  for (int i = 0; i < 3; i++)
    *trace >> dwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    *trace >> dwalkAddr[i];
    assert(dwalkAddr[i] == 0 && dwalkDepth[i] == -1);
  }
  for (int i = 0; i < 3; i++)
    *trace >> dWritebacks[i];

  // Read instruction memory access info.
  *trace >> pc;
  *trace >> branching >> isMisPredict >> fetchDepth;
  if (fetchDepth == -1)
    fetchDepth = 0;
  for (int i = 0; i < 3; i++)
    *trace >> iwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    *trace >> iwalkAddr[i];
    assert(iwalkAddr[i] == 0 && iwalkDepth[i] == -1);
  }
  for (int i = 0; i < 2; i++)
    *trace >> iWritebacks[i];

  // Read source and destination registers.
  *trace >> srcNum >> destNum;
  assert(srcNum <= SRCREGNUM && destNum <= DSTREGNUM);
  for (int i = 0; i < srcNum; i++) {
    *trace >> srcClass[i] >> srcIndex[i];
    assert(srcClass[i] <= MAXREGCLASS);
    assert(srcClass[i] == MAXREGCLASS || srcIndex[i] < MAXREGIDX);
  }
  for (int i = 0; i < destNum; i++) {
    *trace >> destClass[i] >> destIndex[i];
    assert(destClass[i] <= MAXREGCLASS);
    assert(destClass[i] == MAXREGCLASS || destIndex[i] < MAXREGIDX);
  }
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
  Tick fetchLat;
  if (first)
    fetchLat = inTick - tick;
  else
    fetchLat = tick - inTick;
  out << fetchLat << " " << completeTick << " " << storeTick << " ";
  out << decodeTick << " " << renameTick - decodeTick << " "
      << dispatchTick - renameTick << " " << issueTick - dispatchTick << " "
      << completeTick - issueTick << " " << outTick - completeTick << " ";
  if (inSQ())
    out << storeTick - outTick << " " << sqOutTick - storeTick << " ";
  else
    out << "0 0 ";

  out << op + 1 << " " << inSQ() << " " << isMicroOp << " " << isCondCtrl << " "
      << isUncondCtrl << " " << isDirectCtrl << " " << isSquashAfter << " "
      << isSerializeAfter << " " << isSerializeBefore << " "
      << isStoreConditional << " " << isMemBar << " " << isNonSpeculative
      << " ";

  out << branching << " " << isMisPredict << " ";
  // Instruction cache depth.
  out << fetchDepth << " ";
  // Instruction cache line conflict.
  if (!first && getLine(pc) == getLine(PC))
    out << "1 ";
  else
    out << "0 ";
  //// Instruction cache writebacks.
  //for (int i = 0; i < 2; i++)
  //  out << iWritebacks[i] << " ";

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
  //// Data cache writebacks.
  //for (int i = 0; i < 3; i++)
  //  out << dWritebacks[i] << " ";

  // Registers.
  for (int i = 0; i < srcNum; i++)
    out << getReg(srcClass[i], srcIndex[i]) << " ";
  for (int i = srcNum; i < SRCREGNUM; i++)
    out << "0 ";
  for (int i = 0; i < destNum; i++)
    out << getReg(destClass[i], destIndex[i]) << " ";
  for (int i = destNum; i < DSTREGNUM; i++)
    out << "0 ";
}

void Inst::dumpSim(ostream &out) {
  out << pc << " ";
  out << isAddr << " ";
  out << addr << " ";
  out << addrEnd << " ";
}

#endif
