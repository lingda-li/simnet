#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>

#include "inst.h"
#include "queue.h"
#include "inst_impl_1121.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "Usage: ./build1121 <trace> <SQ trace> <start tick>" << endl;
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
  outputName.replace(outputName.end()-3, outputName.end(), "21");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  // Current context.
  struct QUEUE *q = new QUEUE;
  Tick curTick = atol(argv[3]);
  if (curTick != 0)
    curTick += TICK_STEP;
  bool firstInst = true;
  Tick num = 0;
  while (!trace.eof()) {
    Inst *newInst = q->add();
    if (!newInst->read(trace, sqtrace))
      break;
    if (firstInst) {
      firstInst = false;
      // First instruction fetch always misses.
      newInst->fetchDepth = 2;
    }
    q->retire_until(curTick);
    q->dump(curTick, output);
    curTick = newInst->inTick;
    num++;
    if (num % 1000000 == 0)
      cerr << ".";
    if (num % 100000000 == 0)
      cerr << "\n";
  }

  cerr << "\nFinish at " << curTick << " with " << num << " instructions.\n";
  cerr << "Max rob and sq sizes are " << q->max_rob_size << " "
       << q->max_sq_size << "\n";
  cerr << "Min complete and store latency is " << minCompleteLat << " "
       << minStoreLat;
  trace.close();
  sqtrace.close();
  output.close();
  return 0;
}
