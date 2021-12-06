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
  if (argc != 6) {
    cerr << "Usage: ./build1121Sim <trace> <SQ trace> <start tick> <output name> <# insts>"
         << endl;
#else
  if (argc != 5) {
    cerr << "Usage: ./build1121Sim <trace> <start tick> <output name> <# insts>" << endl;
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
  string outputName = argv[4];
  unsigned long long total_num = atol(argv[5]);
#else
  Tick curTick = atol(argv[2]);
  string outputName = argv[3];
  unsigned long long total_num = atol(argv[4]);
#endif
  ofstream output(outputName + ".tr");
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  Tick num = 0;
  Inst newInst;
  if (curTick != 0)
    curTick += TICK_STEP;
  bool firstInst = true;
  while (!trace.eof() && num < total_num) {
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
    newInst.dumpSim(output);
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
