#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "inst.h"
#include "inst_impl_1121.h"

using namespace std;

//#define AC_TRACE

int main(int argc, char *argv[]) {
#ifndef AC_TRACE
  if (argc != 4) {
    cerr << "Usage: ./buildLSTM <trace> <SQ trace> <start tick>"
         << endl;
#else
  if (argc != 3) {
    cerr << "Usage: ./buildLSTM <trace> <start tick>" << endl;
#endif
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
#ifndef AC_TRACE
  ifstream sqtrace(argv[2]);
  if (!sqtrace.is_open()) {
    cerr << "Cannot open SQ trace file.\n";
    return 0;
  }

  Tick curTick = atol(argv[3]);
#else
  Tick curTick = atol(argv[2]);
#endif
  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "seq");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  Tick num = 0;
  Inst newInst;
  // FIXME
  if (curTick != 0)
    curTick += TICK_STEP;
  bool firstInst = true;
  while (!trace.eof()) {
#ifndef AC_TRACE
    if (!newInst.read(trace, sqtrace))
#else
    if (!newInst.read(trace))
#endif
      break;
    if (firstInst) {
      firstInst = false;
      // First instruction fetch always misses.
      newInst.fetchDepth = 2;
    }
    newInst.dump(curTick, true, 0, 0, 0, 0, 0, 0, output);
    output << endl;
    curTick = newInst.inTick;
    num++;
    if (num % 1000000 == 0)
      cerr << ".";
  }

  cerr << "\nFinish with " << num << " instructions.\n";
  trace.close();
#ifndef AC_TRACE
  sqtrace.close();
#endif
  output.close();
  return 0;
}
