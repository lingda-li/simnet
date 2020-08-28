#ifndef __QUEUE_H__
#define __QUEUE_H__

#include <iostream>
#include "inst.h"

using namespace std;

struct ROB {
  Inst insts[ROBSIZE + 1];
  int head = 0;
  int tail = 0;
  int inc(int input) {
    if (input == ROBSIZE)
      return 0;
    else
      return input + 1;
  }
  int dec(int input) {
    if (input == 0)
      return ROBSIZE;
    else
      return input - 1;
  }
  bool is_empty() { return head == tail; }
  bool is_full() { return head == inc(tail); }
  Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    return &insts[old_tail];
  }
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  void retire_until(Tick tick) {
    while (!is_empty() && insts[head].outTick <= tick)
      retire();
  }
  void dump(Tick tick) {
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      insts[i].dump(tick, i == dec(tail), insts[dec(tail)].isAddr,
                    insts[dec(tail)].addr, insts[dec(tail)].addrEnd,
                    insts[dec(tail)].pc, insts[dec(tail)].iwalkAddr,
                    insts[dec(tail)].dwalkAddr);
    }
    cout << "\n";
  }
};

#endif
