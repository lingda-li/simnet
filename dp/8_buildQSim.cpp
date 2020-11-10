#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>

#include "inst.h"
#include "inst_impl_q.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 5) {
    cerr << "Usage: ./buildQSim <trace> <SQ trace> <output name> <# insts>" << endl;
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

  string outputName = argv[3];
  outputName.replace(outputName.end()-3, outputName.end(), "q");
  ofstream output(outputName + ".tr");
  ofstream aux_output(outputName + ".tra");
  if (!output.is_open() || !aux_output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  unsigned long long total_num = atol(argv[4]);

  Tick num = 0;
  Inst newInst;
  Tick curTick;
  bool firstInst = true;
  while (!trace.eof() && num < total_num) {
    if (!newInst.read(trace, sqtrace))
      break;
    if (firstInst) {
      firstInst = false;
      curTick = newInst.inTick;
    }
    newInst.dump(curTick, true, 0, 0, 0, 0, 0, 0, output);
    newInst.dumpSim(aux_output);
    curTick = newInst.inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish with " << num << " instructions.\n";
  trace.close();
  sqtrace.close();
  output.close();
  aux_output.close();
  return 0;
}
