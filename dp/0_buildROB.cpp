#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

#include "data.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Usage: ./buildROB <trace>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }

  // Current ROB context.
  struct ROB *rob = new ROB;
  Tick curTick;
  bool firstInst = true;
  Tick num = 0;
  while (!trace.eof()) {
    Inst *newInst = rob->add();
    if (!newInst->read(trace))
      break;
    if (firstInst) {
      firstInst = false;
      curTick = newInst->inTick;
    }
    rob->retire_until(curTick);
    //newInst->dump(curTick);
    rob->dump(curTick);
    curTick = newInst->inTick;
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish at " << curTick << ".\n";
  trace.close();
  return 0;
}
