#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "inst.h"
#include "inst_impl_q.h"

using namespace std;

//#define AC_TRACE

int main(int argc, char *argv[]) {
#ifndef AC_TRACE
  if (argc != 5) {
    cerr << "Usage: ./buildQSim <trace> <SQ trace> <output name> <# insts>"
         << endl;
#else
  if (argc != 4) {
    cerr << "Usage: ./buildQSim <trace> <output name> <# insts>" << endl;
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

  string outputName = argv[3];
  unsigned long long total_num = atol(argv[4]);
#else
  string outputName = argv[2];
  unsigned long long total_num = atol(argv[3]);
#endif
  ofstream output(outputName + ".tr");
  ofstream aux_output(outputName + ".tra");
  if (!output.is_open() || !aux_output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  Tick num = 0;
  Inst newInst;
  Tick curTick;
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
      curTick = newInst.inTick;
    }
    newInst.dump(curTick, true, 0, 0, 0, 0, 0, 0, output);
    output << endl;
    newInst.dumpSim(aux_output);
    curTick = newInst.inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish with " << num << " instructions.\n";
  trace.close();
#ifndef AC_TRACE
  sqtrace.close();
#endif
  output.close();
  aux_output.close();
  return 0;
}
