#ifndef SIM_H
#define SIM_H
#include <stdio.h>
#include <iostream>
#define TD_SIZE 50
#define ROBSIZE 94
#define SQSIZE 17
#define CONTEXTSIZE (ROBSIZE + SQSIZE)
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 8

#define MIN_ST_LAT 10
#define MIN_COMP_LAT 6
#define CLASS_NUM 10
#define LAT_NUM 3
#define INSQ_BIT 4
#define ATOMIC_BIT 13
#define SC_BIT 14

#define ILINEC_BIT 33
#define IPAGEC_BIT 37
#define DADDRC_BIT 41
#define DLINEC_BIT 42
#define DPAGEC_BIT 46

#define WARPSIZE 32
#define TRACE_DIM 51
#define AUX_TRACE_DIM 10
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define COMBINED

typedef long unsigned Tick;
typedef long unsigned Addr;
float default_val[ML_SIZE];

__device__ __host__ Addr getLine(Addr in) { return in & ~0x3f; }

struct Inst
{
  float train_data[TD_SIZE];
  Tick inTick;
  Tick completeTick;
  Tick storeTick;
  Tick trueFetchTick;
  Tick trueCompleteTick;
  Tick trueStoreTick;
  Addr pc;
  int isAddr;
  Addr addr;
  Addr addrEnd;
  Addr iwalkAddr[3];
  Addr dwalkAddr[3];
  int offset;
  // Read simulation data.
  __host__ __device__ bool inSQ() { return (bool)train_data[INSQ_BIT]; }
  __host__ __device__ bool isStore() { return (bool)train_data[INSQ_BIT] || (bool)train_data[ATOMIC_BIT] || (bool)train_data[SC_BIT]; }


  __device__ __host__ void init(Inst &copy) {
    memcpy(train_data, copy.train_data, sizeof(float)* TD_SIZE);
    inTick = copy.inTick;
    completeTick = copy.completeTick;
    storeTick = copy.storeTick;
    pc = copy.pc;
    isAddr = copy.isAddr;
    addr = copy.addr;
    addrEnd = copy.addrEnd;
    memcpy(iwalkAddr, copy.iwalkAddr, sizeof(Addr)*3);
    memcpy(dwalkAddr, copy.dwalkAddr, sizeof(Addr)*3); 
  }
  
  __host__ __device__ Addr getLine(Addr in) { return in & ~0x3f; }

  bool read_sim_mem(float *trace, uint64_t *aux_trace, int index)
  {
    //train_data = train_d;
    trueFetchTick= trace[0];
    trueCompleteTick= trace[1];
    trueStoreTick= trace[2]; 
    pc = aux_trace[0];
    //cout<< "Before: "<< pc << ", After: "<< getLine(pc) << endl;
    pc= getLine(pc);
    offset = TD_SIZE * index; 
    assert(trueCompleteTick >= MIN_COMP_LAT || trueCompleteTick == 0);
    assert(trueStoreTick == 0 || trueStoreTick >= MIN_ST_LAT);
    for (int i = 3; i < TD_SIZE; i++)
    {
      train_data[i] = trace[i];
      //cout<< trace[i]<<"\t" << train_data[i+offset]<<"\n";
    }
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = 0; 
    isAddr = aux_trace[1];
    addr = aux_trace[2];
    addrEnd = aux_trace[3];
    for (int i = 0; i < 3; i++)
      iwalkAddr[i] = aux_trace[4 + i];
    for (int i = 0; i < 3; i++)
      dwalkAddr[i] = aux_trace[7 + i]; 
    return true;
  }
};

struct ROB
{
  Inst insts[ROBSIZE];
  int head = 0;
  int tail = 0;
  int len = 0;
  bool saturated = false;
  Tick curTick=0;
  Tick lastFetchTick=0;
  __host__ __device__ int inc(int input)
  {
    if (input == (ROBSIZE-1)){
      return 0;}
    else{ 
      return input + 1;}
  }
 

  __host__ __device__ int dec(int input)
  {
    if (input == 0){
     	    return (ROBSIZE-1);}
    else{
            return input - 1;}
  }
  __host__ __device__ bool is_empty() { return head == tail; }
  __host__ __device__ bool is_full() { return head == inc(tail); }

  __host__ __device__ Inst *add()
  {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    len += 1;
    //printf("index updated.\n");
    return &insts[old_tail];
  }

  __device__ Inst *getHead()
  {
    return &insts[head];
  }

  __device__ Tick getHeadTick()
  {
	  return insts[head].completeTick;
  }

  __device__ void
  retire()
  {
    assert(!is_empty());
    head = inc(head);
    len -= 1;
#ifdef DEBUG
    printf("len decreased to retire: %d\n",len);
#endif
  }

  __device__ int retire_until(Tick tick)
  {
    int retired = 0;
#ifdef DEBUG
    printf("Head: %d, Head Tick: %lu, Tick: %lu\n",head,insts[head].completeTick,tick);
#endif
    while (!is_empty() && insts[head].storeTick <= tick)
    {
      //printf("Retire\n");
      retire();
      retired++;
    }
    return retired;
  }

  Inst &tail_inst() { return insts[dec(tail)]; }

    
  __device__ void update_fetch_cycle(Tick tick)
  {
        int warpTID = threadIdx.x % WARPSIZE;
    assert(!is_empty());
    int context;
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    int length = len - 1;
    int i = warpTID;
    while (i < length)
    {
     
      context = start_context - i;
      context = (context >= 0) ? context : context + ROBSIZE;
      //printf("warpTID:%d, Context: %d, curTick: %ld, %.2f\n", warpTID, context, curTick, inst[COMPLETETICK]);
      //printf("Context: %d, Before, %.3f, %.3f, Next: %d\n", context, inst[0], inst[1], dec(i - 32));
      insts[context].train_data[0] += tick;
      assert(insts[context].train_data[0] >= 0.0);
      i += WARPSIZE;
    }
    __syncwarp();
  }

  __device__ int make_input_data(float *inputs, Tick tick, Inst &new_inst)
  { 
    //int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
    //int warpID = TID / WARPSIZE;
    int warpTID = threadIdx.x % WARPSIZE;
    int curr = dec(tail);
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    int W= threadIdx.x/WARPSIZE;
#ifdef DEBUG
    if(warpTID==0){printf("Here. Head: %d, Tail: %d, dec(tail): %d, len: %d\n",head,tail,dec(tail),len);}
#endif
    __syncwarp();
    assert(!is_empty());
    saturated = false;
    __shared__ int num[4];
    int length= len - 1;   
    Addr pc = new_inst.pc;
    int isAddr = new_inst.isAddr;
    Addr addr = new_inst.addr;
    Addr addrEnd = new_inst.addrEnd;	
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
      num[i]=1;
      iwalkAddr[i] = new_inst.iwalkAddr[i];
      dwalkAddr[i] = new_inst.dwalkAddr[i];
    }
   if(warpTID==0){memcpy(inputs, insts[dec(tail)].train_data, sizeof(float)*TD_SIZE);}
    __syncwarp();
   int i = warpTID;
   while (i < length)
    {
      int context = start_context - i;
      context = (context >= 0) ? context : context + ROBSIZE;
      if (insts[context].completeTick <= tick)
      {
      	//{i+=WARPSIZE; printf("context: %d, %.2f, %ld\n",i,inst[COMPLETETICK],tick);continue;}
      	i+=WARPSIZE;
	continue;
      }
      if (num[W] >= CONTEXTSIZE)
      	{	
		//printf("context: %d, %.2f, %ld\n" ,i,inst[COMPLETETICK],tick);
        	saturated = true;
        	i+=WARPSIZE; break;
      	}
      // Update context instruction bits.
      insts[context].train_data[ILINEC_BIT] = insts[context].pc == pc ? 1.0 : 0.0;
       int conflict = 0;
       for (int j = 0; j < 3; j++){
         if (insts[context].iwalkAddr[j] != 0 && insts[context].iwalkAddr[j] == iwalkAddr[j])
         conflict++;
#ifdef DEBUG
	 printf("context: %d, j: %d, %lu, %lu, %lu,conflict: %d\n", context,j,insts[context].iwalkAddr[j],insts[context].iwalkAddr[j],iwalkAddr[j] ,conflict);
#endif
       }
      //printf("context: %d, conflict: %d\n", context, conflict);
      insts[context].train_data[IPAGEC_BIT] = (float)conflict;
      insts[context].train_data[DADDRC_BIT] = (isAddr && insts[context].isAddr && addrEnd >= insts[context].addr && addr <= insts[context].addrEnd) ? 1.0 : 0.0;
      insts[context].train_data[DLINEC_BIT] = (isAddr && insts[context].isAddr && (addr & ~0x3f) == (insts[context].addr & ~0x3f)) ? 1.0 : 0.0; 
      conflict = 0;
      if (isAddr && insts[context].isAddr)
        for (int j = 0; j < 3; j++)
        {
        if (insts[context].dwalkAddr[j] != 0 && insts[context].dwalkAddr[j] == dwalkAddr[j])  
            conflict++;
        }
      insts[context].train_data[DPAGEC_BIT] = (float)conflict;     
      #ifdef DEBUG
      printf("context: %d,ilinec: %.2f,ipagec: %.2f,daddr: %.2f,dlinec: %.2f,dpagec: %.2f\n",context,insts[context].train_data[ILINEC_BIT],insts[context].train_data[IPAGEC_BIT],insts[context].train_data[DADDRC_BIT],insts[context].train_data[DLINEC_BIT],insts[context].train_data[DPAGEC_BIT]);
	#endif
      int poss= atomicAdd(&num[W], 1);
      //printf("Context: %d, poss: %d\n",context,poss);
      memcpy(&inputs[poss*TD_SIZE],&insts[context].train_data,  sizeof(float)*TD_SIZE);
      //{printf("Poss: %d\n",poss);}
      i += WARPSIZE;
    }
    __syncwarp();
    //if(warpTID==0){printf("*********Copy context values.***********. Start: %d, end: %d\n",dec(tail), num[warpID]);}
    /*
    i = warpTID;
    //int cus_T= 0;
    while (i < TD_SIZE)
    { 
      int j = dec(tail);
      int k=0;     
      while (k <= num[W])
      {
        //inputs[i + k * TD_SIZE]= insts[k].train_data[i];
        //printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", k,i,i+j*TD_SIZE,warpTID, inputs[i+j* TD_SIZE]);
       inputs[i + k * TD_SIZE]= insts[j].train_data[i];
	      j= dec(j);
	k++;
      }
      i+= WARPSIZE;
    }
    __syncwarp();
    
   */
    /*
    //printf("Here. 2\n");
    //if(warpTID==0){printf("************Adding default values.*****************. Start: %d\n",num[W]);}
    i = warpTID;
    while (i < TD_SIZE)
    {
      int j = num[W] ;
      while (j < CONTEXTSIZE)
      {
        inputs[i + j * TD_SIZE] = default_val[i];
        //printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", j,i,i+j*TD_SIZE,warpTID, default_val[i]);
        j+=1;
      }
      i+= WARPSIZE;
    }

    */
    //printf("Here. 3\n");
    __syncwarp();
    return num[W];
  }
};


  __device__ void
  dis(float *data, int size, int rows)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < size; j++)
      {
        printf("%.3f  ", data[i * size + j]);
      }
      printf("\n");
    }
  }
  

__global__ void
result(ROB *rob_d, int Total_Trace, int instructions)
{
  Tick sum = 0;
  for (int i = 0; i < Total_Trace; i++)
  {
    //printf("I: %d\n",i);
    ROB *rob= &rob_d[i];	  
    sum += rob->curTick;
    //printf("T: %d, Tick: %lu\n", i, rob->curTick);
  }
  printf("~~~~~~~~~Instructions: %d, Batch: %d, Prediction: %lu ~~~~~~~~~\n", instructions,Total_Trace, sum);
}


__device__ void inst_copy(Inst *dest, Inst *source)
{
        //dest->trueFetchClass= source->trueFetchClass;
        dest->trueFetchTick= source->trueFetchTick;
        //dest->trueCompleteClass= source->trueCompleteClass;
        dest->trueCompleteTick= source->trueCompleteTick;
        dest->pc= source->pc;
        dest->isAddr= source->isAddr;
        dest->addr= source->addr;
        dest->addrEnd= source->addrEnd;
        for(int i=0;i<3; i++){
                dest->iwalkAddr[i]= source->iwalkAddr[i];
                dest->dwalkAddr[i]= source->dwalkAddr[i];
        }
        for(int i=0;i<TD_SIZE;i++){
                dest->train_data[i]= source->train_data[i];
        }
}

struct SQ {
  Inst insts[SQSIZE];
  int size= SQSIZE;
  int head = 0;
  int len = 0;
  int tail = 0;
  __device__ int inc(int input) {
    if (input == size - 1)
      return 0;
    else
      return input + 1;
  }
  __device__ int dec(int input) {
    if (input == 0)
      return size - 1;
    else
      return input - 1;
  }
  __device__ bool is_empty() { return head == tail; }
  __device__ bool is_full() { return head == inc(tail); }
  __device__ Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    return &insts[old_tail];
  }
  __device__ Inst *getHead() {
    return &insts[head];
  }
  __device__ void retire() {
    assert(!is_empty());
    head = inc(head);
  }

  __device__ int retire_until(Tick tick, SQ *sq = nullptr) {
	   int retired=0;
         while (!is_empty() && insts[head].completeTick <= tick &&
             retired < RETIRE_BANDWIDTH) {
        if (insts[head].isStore()) {
          if (sq->is_full())
            break;
          Inst *newInst = sq->add();
          newInst->init(insts[head]);
        }
        retire();
        retired++;
      }
      return retired;
   }


__device__ Inst &tail_inst() { return insts[dec(tail)]; }

__device__ int make_input_data(float *input, Tick tick, Inst &new_inst) {
     int warpTID = threadIdx.x % WARPSIZE;
    Addr pc = new_inst.pc;
    int isAddr = new_inst.isAddr;
    Addr addr = new_inst.addr;
    Addr addrEnd = new_inst.addrEnd;
    Addr iwalkAddr[3], dwalkAddr[3];
    int __shared__ num[4];
    for (int i = 0; i < 3; i++) {
      num[i]=0;
      iwalkAddr[i] = new_inst.iwalkAddr[i];
      dwalkAddr[i] = new_inst.dwalkAddr[i];
    }
     int W= threadIdx.x/WARPSIZE;
    int length= len - 1;   
    int i;
      assert(!is_empty());
      assert(&new_inst == &insts[dec(tail)]);
      memcpy(input, new_inst.train_data, sizeof(float)*TD_SIZE);
      i = warpTID;
     int start_context = dec(dec(tail));
    int end_context = dec(head);
//for (; i != dec(head); i = dec(i)) {
      // Update context instruction bits.
    while(i < length) {
      int context = start_context - i;
      context = (context >= 0) ? context : context + SQSIZE;
      insts[context].train_data[ILINEC_BIT] = insts[context].pc == pc ? 1.0 : 0.0;
      int conflict = 0;
      for (int j = 0; j < 3; j++) {
        if (insts[context].iwalkAddr[j] != 0 && insts[context].iwalkAddr[j] == iwalkAddr[j])
          conflict++;
      }
      insts[context].train_data[IPAGEC_BIT] = (float)conflict;
      insts[context].train_data[DADDRC_BIT] = (isAddr && insts[context].isAddr && addrEnd >= insts[context].addr && addr <= insts[context].addrEnd) ? 1.0 : 0.0;
      insts[context].train_data[DLINEC_BIT] = (isAddr && insts[context].isAddr && getLine(addr) == getLine(insts[context].addr)) ? 1.0 : 0.0;
      conflict = 0;
      if (isAddr && insts[context].isAddr)
        for (int j = 0; j < 3; j++) {
          if (insts[context].dwalkAddr[j] != 0 && insts[context].dwalkAddr[j] == dwalkAddr[j])
            conflict++;
        }
      insts[context].train_data[DPAGEC_BIT] = (float)conflict;
      int poss= atomicAdd(&num[W], 1);
      memcpy(input + TD_SIZE * poss, insts[context].train_data, sizeof(float) * TD_SIZE);
      i+= WARPSIZE;
    }
    return num[W];
  }


  __device__ void update_fetch_cycle(Tick tick) {
    int warpTID = threadIdx.x % WARPSIZE;
    int i;
    int length= len - 1;
    int context;
    int start_context = (dec(tail));
    int end_context = dec(head);

    //for (; i != dec(head); i = dec(i)) {
    while(i < length){
      context = start_context - i;
      context = (context >= 0) ? context : context + SQSIZE;
      insts[context].train_data[0] += tick;
      assert(insts[context].train_data[0] >= 0.0);
    }
  }
};


__device__ Tick max_(Tick a, Tick b)
{
   if (a>b) return a;
   else return b;
}


__device__ Tick min_(Tick a, Tick b)
{
   if (a<b) return a;
   else return b;
}



__global__ void
update(ROB *rob_d, SQ *sq_d, float *output, int *status, int Total_Trace)
{
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int index = TID;
  ROB *rob;
  while (index < Total_Trace)
  {
#if defined(COMBINED)
    int offset= index * 22;
#else
    int offset = index * LAT_NUM;
#endif   
    Tick nextFetchTick = 0;
    rob= &rob_d[index];
        int tail= rob->dec(rob->tail);
        int context_offset = rob->dec(rob->tail) * TD_SIZE;
    #if defined(COMBINED)
    int classes[LAT_NUM];
    for(int i=0; i<LAT_NUM;i++)
    {
	     float max = output[offset + CLASS_NUM*i+LAT_NUM];
	     int idx=0;
	     //printf("i: %d, max: %d\n", i ,max);
	     for (int j = 1; j < 10; j++) {
		     //printf("i: %d, max: %f, value: %f\n", i ,max,output[offset + 10*i+2+j]);
		     if (max < output[offset + CLASS_NUM*i+LAT_NUM+j]) {
            max = output[offset + CLASS_NUM*i+LAT_NUM+j];
            idx = j;
          }
	   classes[i]= idx;   
	     }
    }
    //printf("fclass: %d, cclass: %d\n", f_class, c_class);
#endif
    float fetch_lat = output[offset + 0];
    float complete_lat = output[offset + 1];
    float store_lat= output[offset + 2];
    int int_fetch_lat = round(fetch_lat);
    int int_complete_lat = round(complete_lat);
    int int_store_lat = round(store_lat);
    
    #ifdef DEBUG
    printf("%ld, %.3f, %d, %.3f, %d\n",rob->curTick, output[offset+0], int_fetch_lat, output[offset+1], int_finish_lat);
    #endif 
#if defined(COMBINED)
      if (classes[0] <= 8)
        int_fetch_lat = classes[0];
      if (classes[1] <= 8)
        int_complete_lat = classes[1] + MIN_COMP_LAT;
      if (classes[2] == 0)
        int_store_lat = 0;
      else if (classes[2] <= 8 )
        int_store_lat = classes[2] + MIN_ST_LAT - 1;
#endif

      // Calibrate latency.
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;
      if (int_complete_lat < MIN_COMP_LAT)
        int_complete_lat = MIN_COMP_LAT;
      if (!rob->insts[tail].isStore()) {
        //assert(newInst->trueStoreTick == 0);
        int_store_lat = 0;
      } else if (int_store_lat < MIN_ST_LAT)
        int_store_lat = MIN_ST_LAT;

    rob->insts[tail].train_data[0] = -int_fetch_lat;
    rob->insts[tail].train_data[1] = int_complete_lat;
    rob->insts[tail].train_data[2] = int_store_lat;
    #ifdef DEBUG
    printf("Index: %d, offset: %d, Fetch: %.4f, Finish: %.4f, Rob0: %.2f, Rob1: %.2f, Rob2: %.2f, Rob3: %.2f\n", index, rob->tail, output[offset + 0], output[offset + 1], rob_pointer[0], rob_pointer[1], rob_pointer[2], rob_pointer[3]);
#endif
    rob->insts[tail].storeTick = rob->curTick +  int_fetch_lat + int_store_lat;
    rob->insts[tail].completeTick = rob->curTick + int_fetch_lat + int_complete_lat + 1;
    rob->lastFetchTick = rob->curTick;
    if (int_fetch_lat)
    {
      status[index]=1;
      nextFetchTick = rob->curTick + int_fetch_lat;
        //printf("Break with int fetch\n");
    }
    else{status[index]=0; index += (gridDim.x * blockDim.x);continue;}
    if ( int_fetch_lat)
    {
      Tick nextCommitTick= max_(rob->getHeadTick(), rob->curTick + 1);
      rob->curTick= min_(nextCommitTick, nextFetchTick);
      //printf("getting max, cur = %lu\n",rob->curTick);
    }
    else if (rob->curTick < nextFetchTick)
    {
      Tick nextCommitTick= max_(rob->getHeadTick(), rob->curTick + 1);
      rob->curTick= min_(nextCommitTick, nextFetchTick);
      //printf("fastforward to fetch\n");
    }
    else if (rob->is_full())
    {
      rob->curTick =  max_(rob->getHeadTick(), rob->curTick + 1);
      //printf("fastforward to retire\n");
    }
    else{
	rob->curTick =  max_(rob->getHeadTick(), rob->curTick + 1);
    }
    int_fetch_lat= 0;
    index += (gridDim.x * blockDim.x);
  }
}



int read_trace_mem(char trace_file[], char aux_trace_file[], float *trace, Tick *aux_trace, unsigned long int instructions)
{
  FILE *trace_f = fopen(trace_file, "rb");
  if (!trace_f)
  {
    printf("Unable to read trace binary.");
    return 1;
  }

  unsigned long int tot= TRACE_DIM * instructions;
  Tick r = fread(trace, sizeof(float), TRACE_DIM * instructions, trace_f);
  printf("tot: %lu, Toread: %lu, read :%lu values for trace.\n",tot,TRACE_DIM * instructions, r);
  //display(trace,TRACE_DIM,2);

  FILE *aux_trace_f = fopen(aux_trace_file, "rb");
  if (!aux_trace_f)
  {
    printf("Unable to aux_trace binary.");
    return 1;
  }
  int k = fread(aux_trace, sizeof(Tick), AUX_TRACE_DIM * instructions, aux_trace_f);
  printf("read :%d values for aux_trace.\n", k);
  //display(aux_trace,AUX_TRACE_DIM,2);
  return true;
}

#endif
