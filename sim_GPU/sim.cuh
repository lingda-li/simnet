#ifndef SIM_H
#define SIM_H

#define TD_SIZE 51
#define INST_SIZE 62
#define CONTEXTSIZE 111
#define ROBSIZE 400
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define ILINEC_BIT 33
#define IPAGEC_BIT 38
#define DADDRC_BIT 42
#define DLINEC_BIT 43
#define DPAGEC_BIT 47
#define PC 51
#define ISADDR 52
#define ADDR 53
#define ADDREND 54
#define IWALK0 55
#define IWALK1 56
#define IWALK2 57
#define DWALK0 58
#define DWALK1 59
#define DWALK2 60
#define COMPLETETICK 61
#define WARPSIZE 32
#define TRACE_DIM 51
#define AUX_TRACE_DIM 10
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6
typedef long unsigned Tick;
typedef long unsigned Addr;

float factor[TD_SIZE];
float mean[TD_SIZE];
float default_val[TD_SIZE];

class Inst
{
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
  int isAddr;
  Addr addr;
  Addr addrEnd;
  Addr iwalkAddr[3];
  Addr dwalkAddr[3];
  int offset;
  //H_ERR(cudaMalloc((void **)&train_data_d, sizeof(int)*TD_SIZE));
  // Read simulation data.

  __device__ void
  dis(float *data, int size, int rows)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < size; j++)
      {
        printf("%.1f  ", data[i * size + j]);
      }
      printf("\n");
    }
  }

  Inst() {}
  Inst(float *pointer)
  {
    train_data = pointer;
  }

  bool read_sim_mem(float *trace, uint64_t *aux_trace, float *train_d, int index)
  {
    train_data = train_d;
    trueFetchClass = trace[0];
    trueFetchTick = trace[1];
    trueCompleteClass = trace[2];
    trueCompleteTick = trace[3];
    pc = aux_trace[0];
    offset = INST_SIZE * index;
    //cout<< "Offset: "<< offset <<"   Memory: "<<train_data;
    //assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++)
    {
      train_data[i + offset] = trace[i] / factor[i];
      //cout<< trace[i]<<"\t" << train_data[i+offset]<<"\n";
    }
    train_data[0 + offset] = train_data[1 + offset] = 0.0;
    train_data[2 + offset] = train_data[3 + offset] = 0.0;
    //aux_trace >> isAddr >> addr >> addrEnd;
    isAddr = aux_trace[1];
    addr = aux_trace[2];
    addrEnd = aux_trace[3];
    for (int i = 0; i < 3; i++)
      iwalkAddr[i] = aux_trace[3 + i];
    for (int i = 0; i < 3; i++)
      dwalkAddr[i] = aux_trace[6 + i];
    train_data[PC] = (float)pc;
    train_data[ISADDR] = (float)isAddr;
    train_data[ADDR] = (float)addr;
    train_data[ADDREND] = (float)addrEnd;
    train_data[IWALK0] = (float)iwalkAddr[0];
    train_data[IWALK1] = (float)iwalkAddr[1];
    train_data[IWALK2] = (float)iwalkAddr[2];
    train_data[DWALK0] = (float)dwalkAddr[0];
    train_data[DWALK1] = (float)dwalkAddr[1];
    train_data[DWALK2] = (float)dwalkAddr[2];
    //cout << "in: ";
    for (int i = 0; i < TD_SIZE; i++)
    {
    } //cout << train_data[i] << " ";
      //cout << "\n";
    return true;
  }
};

class ROB
{
public:
  float *insts;
  int head = 0;
  int tail = 0;
  int len = 0;
  bool saturated = false;
  ~ROB(){};
  ROB()
  {
    H_ERR(cudaMalloc((void **)&insts, sizeof(float) * (ROBSIZE * INST_SIZE)));
  };
  __host__ __device__ int inc(int input)
  {
    if (input == ROBSIZE)
      return 0;
    else
      return input + 1;
  }

  __host__ __device__ int dec(int input)
  {
    if (input == 0)
      return ROBSIZE;
    else
      return input - 1;
  }
  __host__ __device__ bool is_empty() { return head == tail; }
  __host__ __device__ bool is_full() { return head == inc(tail); }

  __host__ __device__ int add()
  {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    len += 1;
    //printf("index updated.\n");
    return old_tail;
  }

  __device__ int getHead()
  {
    return head;
  }

  __device__ void
  retire()
  {
    assert(!is_empty());
    head = inc(head);
    len -= 1;
  }

  __device__ int retire_until(Tick tick, float *insts)
  {
    int retired = 0;
    while (!is_empty() && insts[COMPLETETICK] <= tick)
    {
      retire();
      retired++;
    }
    return retired;
  }

  __device__ void
  dis(float *data, int size, int rows)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < size; j++)
      {
        printf("%.1f  ", data[i * size + j]);
      }
      printf("\n");
    }
  }

  __device__ void update_fetch_cycle(Tick tick, Tick curTick, float *factor, float *insts)
  {
    int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
    //int warpID= TID / WARPSIZE;
    int warpTID = threadIdx.x % WARPSIZE;
    assert(!is_empty());
    int context;
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    int length = len - 1;
    int i = warpTID;
    //{printf("TID: %d, Index: %d,len: %d, Update: start: %d, end: %d\n",warpTID,i,len,start_context,end_context);}
    //for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
    if (warpTID == 0)
    {
      printf("ROB:, head: %d, tail: %d \n", head, tail);
      dis(insts, INST_SIZE, 4);
    }
    __syncwarp();
    while (i < length)
    {
      //printf("I: %d\n",i);
      context = start_context - i;
      context = (context >= 0) ? context : context + ROBSIZE;
      float *inst = insts + context * INST_SIZE;
      //printf("warpTID:%d, Context: %d, curTick: %ld, %.2f\n", warpTID, context, curTick, inst[COMPLETETICK]);
      if (inst[COMPLETETICK] <= (float)curTick)
      {
        //printf("COntext: %d, warpTID: %d, Curtick: %ld, Inst: %.2f,continue\n", context, warpTID, curTick, inst[COMPLETETICK]);
        i += WARPSIZE;
        continue;
      }
      //printf("Context: %d, Before, %.3f, %.3f, Next: %d\n", context, inst[0], inst[1], dec(i - 32));
      inst[0] += tick / factor[0];
      if (inst[0] >= 9 / factor[0])
        inst[0] = 9 / factor[0];
      inst[1] += tick / factor[1];
      //printf("Context: %d, After, %.3f, %.3f,Next: %d\n", context, inst[0], inst[1], dec(i - 32));
      assert(inst[0] >= 0.0);
      assert(inst[1] >= 0.0);
      i += WARPSIZE;
    }
    __syncwarp();
  }

  __device__ int make_input_data(float *context, float *insts, Tick tick, float *factor, float *default_val)
  {
    //if(){printf("Here. Head: %d, Tail: %d\n",head,tail);}

    int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
    int warpID = TID / WARPSIZE;
    int warpTID = TID % WARPSIZE;
    int curr = dec(tail);
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    assert(!is_empty());
    saturated = false;
    __shared__ int num[4];
    Addr pc = insts[curr * INST_SIZE + PC];
    int isAddr = insts[curr * INST_SIZE + ISADDR];
    Addr addr = insts[curr * INST_SIZE + ADDR];
    Addr addrEnd = insts[curr * INST_SIZE + ADDREND];
    Addr iwalkAddr[3], dwalkAddr[3];
    int i = warpTID;
    int length = len - 1;
    //if (warpTID==0){
    while (i < 3)
    {
      //for (int i = 0; i < 3; i++) {
      iwalkAddr[i] = insts[curr * INST_SIZE + IWALK0 + i];
      dwalkAddr[i] = insts[curr * INST_SIZE + DWALK0 + i];
      i++;
    }
    __syncwarp();
    i = warpTID;
    while (i > length)
    {
      int context_ = start_context - i;
      context_ = (context_ >= 0) ? context_ : context_ + ROBSIZE;
      float *inst = insts + context_ * INST_SIZE;
      printf("ThreadID: %d, inst id: %d\n", warpTID, i);
      if (inst[COMPLETETICK] <= tick)
        continue;
      if (num[warpID] >= CONTEXTSIZE)
      {
        saturated = true;
        return 0;
      }
      // Update context instruction bits.
      inst[ILINEC_BIT] = inst[PC] == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
      int conflict = 0;
      for (int j = 0; j < 3; j++)
      {
        if (inst[j] != 0 && inst[j] == iwalkAddr[j])
          conflict++;
      }
      inst[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
      inst[DADDRC_BIT] = (isAddr && insts[ISADDR] && addrEnd >= inst[ADDR] && addr <= inst[ADDREND]) ? 1.0 / factor[DADDRC_BIT] : 0.0;
      inst[DLINEC_BIT] = (isAddr && inst[ISADDR] && (addr) == (inst[ADDR])) ? 1.0 / factor[DLINEC_BIT] : 0.0;
      conflict = 0;
      if (isAddr && inst[ISADDR])
        for (int j = 0; j < 3; j++)
        {
          if (inst[j] != 0 && inst[j] == dwalkAddr[j])
            conflict++;
        }
      inst[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
      //std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      //num++;
      atomicAdd(&num[warpID], 1);
      i -= WARPSIZE;
    }
    __syncwarp();
    i = warpTID;
    while (i < TD_SIZE)
    {
      //printf("thread: %d, i: %d\n",warpTID,i);
      //if(warpTID==0){printf("");}
      int j = curr;
      while (j != end)
      {
        context[i + j * TD_SIZE] = insts[j * INST_SIZE + i];
        //printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", j,i,i+j*TD_SIZE,warpTID, default_val[i]);
        j = dec(j);
      }
      i += WARPSIZE;
    }
    __syncwarp();

    //printf("Adding default values.\n");
    i = warpTID;
    while (i < TD_SIZE)
    {
      //for (int i = num; i < CONTEXTSIZE; i++) { //printf("thread: %d, i: %d\n",warpTID,i);
      //if(warpTID==0){printf("");}
      int j = 1;
      while (j != end)
      {
        context[i + j * TD_SIZE] = default_val[i];
        //printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", j,i,i+j*TD_SIZE,warpTID, default_val[i]);
        j++;
      }
      i += WARPSIZE;
    }
    __syncwarp();
    return 0;
  }
};

class ROB_d
{
public:
  ROB *rob;
  ROB_d(int Total_Trace)
  {
    //ROB rob[Total_Trace];
    H_ERR(cudaMalloc((void **)&rob, sizeof(ROB) * (Total_Trace)));
  }
};


#endif