#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>

typedef long unsigned Tick;
typedef long unsigned Addr;
#define TD_SIZE 51
#define ROBSIZE 400
#include "herror.h"
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TD_SIZE 51
#define CONTEXTSIZE 111
#define ROBSIZE 400
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6

#define ILINEC_BIT 33
#define IPAGEC_BIT 38
#define DADDRC_BIT 42
#define DLINEC_BIT 43
#define DPAGEC_BIT 47

//float factor[TD_SIZE];
//float mean[TD_SIZE];
//float default_val[TD_SIZE];


struct Inst {
  float train_data[TD_SIZE];
  Tick inTick;
  Tick completeTick;
  Tick tickNum;
  Tick trueFetchTick;
  Tick trueCompleteTick;
  int trueFetchClass;
  int trueCompleteClass;
  Addr pc;
  int isAddr;
  Addr addr;
  Addr addrEnd;
  Addr iwalkAddr[3];
  Addr dwalkAddr[3];
  // Read simulation data.
  bool read_sim_data(ifstream &trace, ifstream &aux_trace) {
    cout<<"read started...\n";
          trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++) {
      trace >> train_data[i];
      train_data[i] /= factor[i];
    }
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = train_data[3] = 0.0;
    aux_trace >> isAddr >> addr >> addrEnd;
    for (int i = 0; i < 3; i++)
      aux_trace >> iwalkAddr[i];
    for (int i = 0; i < 3; i++)
      aux_trace >> dwalkAddr[i];
    assert(!trace.eof() && !aux_trace.eof());
    //cout << "in: ";
    //for (int i = 0; i < TD_SIZE; i++)
    //  cout << train_data[i] << " ";
    cout << "Read complete\n";
    return true;
  }
};



class Inst_d{
public:
    float *train_data;
    Tick inTick;
    Tick completeTick;
    Tick tickNum;
    Tick trueFetchTick;
    Tick trueCompleteTick;
    int trueFetchClass;
    int trueCompleteClass;
    Addr pc;
    int isAddr=2;
    Addr addr;
    Addr addrEnd;
    Addr iwalkAddr[3];
    Addr dwalkAddr[3];
    ~Inst_d(){};
    Inst_d(){};	    
    void init(){
        H_ERR(cudaMalloc((void **)&train_data, sizeof(int)*TD_SIZE));
    }
};

class ROB{
public:
    Inst insts[ROBSIZE +1];
    int head= 0;
    int tail= 0;
    bool saturated= false;
    float factor[TD_SIZE];
    float mean[TD_SIZE];
    float default_val[TD_SIZE];
    void init(){
        //insts.init();
        //H_ERR(cudaMalloc((void **)&train_data, sizeof(int)*(ROBSIZE +1)));
    }
    ~ROB(){};
    ROB(){
    	H_ERR(cudaMalloc((void **)&insts, sizeof(Inst)*(ROBSIZE+1)));
    };
    __device__ int inc(int input) {
        if (input == ROBSIZE)
          return 0;
        else
          return input + 1;
    }

    __device__ int dec(int input) {
        if (input == 0)
          return ROBSIZE;
        else
          return input - 1;
    }
    __device__ bool is_empty() { return head == tail; }
    __device__ bool is_full() { return head == inc(tail); }
    
    
     __device__ Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    printf("index updated.\n");
    return &insts[old_tail];
  }
    
    __device__ 
    Inst_d *getHead() {
        return &insts[head];
      }
    __device__ 
      void retire() {
        assert(!is_empty());
        head = inc(head);
    }
    __device__ 
    int retire_until(Tick tick) {
        printf("Retire...\n");
	    int retired = 0;
        while (!is_empty() && insts[head].completeTick <= tick) {
          retire();
          retired++;
        }
        return retired;
    }

    __device__ 
    void make_input_data(float *context, Tick tick) {
        assert(!is_empty());
        saturated = false;
        Addr pc = insts[dec(tail)].pc;
        int isAddr = insts[dec(tail)].isAddr;
        Addr addr = insts[dec(tail)].addr;
        Addr addrEnd = insts[dec(tail)].addrEnd;
        Addr iwalkAddr[3], dwalkAddr[3];
        for (int i = 0; i < 3; i++) {
          iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
          dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
        }
        //std::copy(insts[dec(tail)].train_data, insts[dec(tail)].train_data + TD_SIZE, context);
        int num = 1;
        for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
          if (insts[i].completeTick <= tick)
            continue;
          if (num >= CONTEXTSIZE) {
            saturated = true;
            break;
          }
          // Update context instruction bits.
          insts[i].train_data[ILINEC_BIT] = insts[i].pc == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
          int conflict = 0;
          for (int j = 0; j < 3; j++) {
            if (insts[i].iwalkAddr[j] != 0 && insts[i].iwalkAddr[j] == iwalkAddr[j])
              conflict++;
          }
          insts[i].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
          insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr && addr <= insts[i].addrEnd) ? 1.0 / factor[DADDRC_BIT] : 0.0;
          insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].isAddr && (addr & ~0x3f) == (insts[i].addr & ~0x3f)) ? 1.0 / factor[DLINEC_BIT] : 0.0;
          conflict = 0;
          if (isAddr && insts[i].isAddr)
            for (int j = 0; j < 3; j++) {
              if (insts[i].dwalkAddr[j] != 0 && insts[i].dwalkAddr[j] == dwalkAddr[j])
                conflict++;
            }
          insts[i].train_data[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
          //std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
          num++;
        }
        for (int i = num; i < CONTEXTSIZE; i++) {
          //std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
        }
      }

    __device__ 
    void update_fetch_cycle(Tick tick, Tick curTick) {
        assert(!is_empty());
        for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
          if (insts[i].completeTick <= curTick)
            continue;
          insts[i].train_data[0] += tick / factor[0];
          if (insts[i].train_data[0] >= 9 / factor[0])
            insts[i].train_data[0] = 9 / factor[0];
          insts[i].train_data[1] += tick / factor[1];
          assert(insts[i].train_data[0] >= 0.0);
          assert(insts[i].train_data[1] >= 0.0);
        }
      }

};
