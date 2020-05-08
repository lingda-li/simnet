#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

#include "data.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Usage: ./buildCtx <trace>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }

  // Current ROB context.
  struct Context *cxt = new Context;
  while (true) {
    Inst *newInst = cxt->add();
    if (cxt->is_full())
      cxt->retire();
    if (!newInst->read(trace))
      break;
    Tick curTick = newInst->inTick;
    cxt->dump(curTick);
  }

  trace.close();
  return 0;
}
