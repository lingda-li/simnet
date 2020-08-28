#ifndef __INST_H__
#define __INST_H__

#include <iostream>

using namespace std;

#define MAXREGNUM 8
#define ROBSIZE 100
#define CONTEXT_SIZE 96
#define TICK_STEP 500

#define SRCREGNUM 8
#define DSTREGNUM 6

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
  int srcClass[MAXREGNUM];
  int srcIndex[MAXREGNUM];
  int destClass[MAXREGNUM];
  int destIndex[MAXREGNUM];

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

  // Read one instruction from SQ and ROB traces.
  bool read(ifstream &trace, ifstream &SQtrace);

  // Generate the final OP code.
  void combineOp();

  // Dump instruction for ML input.
  void dump(Tick tick, bool first, int is_addr, Addr begin, Addr end, Addr PC,
            Addr *iwa, Addr *dwa);

  // Dump instruction for simulator input.
  void dumpSim();
};

#endif
