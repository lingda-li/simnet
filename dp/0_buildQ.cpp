#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>

#include "inst.h"
#include "queue.h"
#include "inst_impl_q.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: ./buildQ <trace> <SQ trace>" << endl;
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
  outputName.replace(outputName.end()-3, outputName.end(), "qq");
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

  cerr << "\nFinish at " << curTick << " with " << num << " instructions.\n";
  cerr << "Max rob and sq sizes are " << q->max_rob_size << " "
       << q->max_sq_size;
  trace.close();
  sqtrace.close();
  output.close();
  return 0;
}
