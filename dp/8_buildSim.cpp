#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

#include "data.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: ./buildSim <trace> <# insts>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  unsigned long long total_num = atol(argv[2]);

  Tick num = 0;
  Inst newInst;
  while (!trace.eof() && num < total_num) {
    if (!newInst.read(trace))
      break;
    newInst.dumpSim();
    num++;
    if (num % 100000 == 0)
      cerr << ".";
  }

  cerr << "Finish with " << num << " instructions.\n";
  trace.close();
  return 0;
}
