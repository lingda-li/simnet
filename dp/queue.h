#ifndef __QUEUE_H__
#define __QUEUE_H__

#include <iostream>
#include "inst.h"

using namespace std;

#define QSIZE 400

struct QUEUE {
  Inst insts[QSIZE];
  int head = 0;
  int rob_head = 0;
  int tail = 0;
  int inc(int input) {
    if (input == QSIZE - 1)
      return 0;
    else
      return input + 1;
  }
  int dec(int input) {
    if (input == 0)
      return QSIZE - 1;
    else
      return input - 1;
  }
  bool is_empty() { return head == tail; }
  bool is_rob_empty() { return rob_head == tail; }
  bool is_sq_empty() { return rob_head == head; }
  bool is_full() { return head == inc(tail); }
  Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    return &insts[old_tail];
  }
  void retire_sq_head() {
    assert(!is_sq_empty() && !is_empty());
    head = inc(head);
  }
  void retire_rob_head() {
    assert(!is_rob_empty() && !is_empty());
    rob_head = inc(rob_head);
  }
  void retire_until(Tick tick) {
    while (!is_rob_empty() && insts[rob_head].robTick() <= tick)
      retire_rob_head();
    while (!is_sq_empty() && insts[head].sqTick() <= tick)
      retire_sq_head();
  }
  void dump(Tick tick, ostream &out = cout) {
    for (int i = dec(tail); i != dec(rob_head); i = dec(i)) {
      insts[i].dump(tick, i == dec(tail), insts[dec(tail)].isAddr,
                    insts[dec(tail)].addr, insts[dec(tail)].addrEnd,
                    insts[dec(tail)].pc, insts[dec(tail)].iwalkAddr,
                    insts[dec(tail)].dwalkAddr, out);
    }
    for (int i = dec(rob_head); i != dec(head); i = dec(i)) {
      if (insts[i].inSQ())
        insts[i].dump(tick, i == dec(tail), insts[dec(tail)].isAddr,
                      insts[dec(tail)].addr, insts[dec(tail)].addrEnd,
                      insts[dec(tail)].pc, insts[dec(tail)].iwalkAddr,
                      insts[dec(tail)].dwalkAddr, out);
    }
    out << "\n";
  }
};

#endif
