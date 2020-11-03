#ifndef __INST_H__
#define __INST_H__

#include <iostream>

using namespace std;

#define SRCREGNUM 8
#define DSTREGNUM 6
#define MAXREGCLASS 6
#define MAXREGIDX 50

typedef long unsigned Tick;
typedef long unsigned Addr;

struct Inst {
  // Operation.
  int op;
  int isMicroOp;
  int isCondCtrl;
  int isUncondCtrl;
  int isDirectCtrl;
  int isSquashAfter;
  int isSerializeAfter;
  int isSerializeBefore;
  int isAtomic;
  int isStoreConditional;
  int isMemBar;
  int isQuiesce;
  int isNonSpeculative;

  // Registers.
  int srcNum;
  int destNum;
  int srcClass[SRCREGNUM];
  int srcIndex[SRCREGNUM];
  int destClass[DSTREGNUM];
  int destIndex[DSTREGNUM];

  // Data access.
  int isAddr;
  Addr addr;
  Addr addrEnd;
  unsigned int size;
  int depth;
  int dwalkDepth[3];
  Addr dwalkAddr[3];
  int dWritebacks[3];
  int sqIdx;

  // Instruction access.
  Addr pc;
  int isMisPredict;
  int fetchDepth;
  int iwalkDepth[3];
  Addr iwalkAddr[3];
  int iWritebacks[2];

  // Timing.
  Tick inTick;
  Tick completeTick;
  Tick outTick;
  Tick storeTick;
  Tick sqOutTick = 0;

  // Read one instruction from SQ and ROB traces.
  bool read(ifstream &ROBtrace, ifstream &SQtrace);

  // Generate the final OP code.
  void combineOp();

  // Whether it is in SQ.
  bool inSQ() {
    if (sqIdx != -1 && !isStoreConditional && !isAtomic)
      return true;
    else
      return false;
  }

  // Get ticks.
  Tick robTick() { return inTick + outTick; }
  Tick sqTick() {
    if (sqOutTick == 0)
      return inTick + storeTick;
    else
      return inTick + sqOutTick;
  }

  // Dump instruction for ML input.
  void dump(Tick tick, bool first, int is_addr, Addr begin, Addr end, Addr PC,
            Addr *iwa, Addr *dwa, ostream &out = cout);

  // Dump instruction for simulator input.
  void dumpSim();
};

#endif
