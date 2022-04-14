#ifndef SIM_H
#define SIM_H
#include <stdio.h>
#include <iostream>
#define TD_SIZE 50
#define INST_SIZE 51
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
#define WIDTH 4

#define WARPSIZE 32
#define TRACE_DIM 50
#define AUX_TRACE_DIM 10
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
//#define COMBINED
//#define DEBUG
#define WARP
//#define TRACK
#define Stream_width 3 

typedef long unsigned Tick;
typedef long unsigned Addr;
float default_val[ML_SIZE];
float zeros[TD_SIZE];

__device__ __host__ Addr getLine(Addr in) { return in & ~0x3f; }



  __device__ void copier(float *destination, float *source, int size)
  {
    int warpTID = threadIdx.x % WARPSIZE;
    int i=warpTID;
    while (i<size)
    {      
      destination[i]= source[i];
      i+=WARPSIZE;
      //printf("T: %d, source: %.2f, destination: %.2f\n",warpTID,source[i],destination[i]);
    }
  }


void display(float *data, int size, int rows)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%.0f  ", data[i * size + j]);
      /*
      if (data[i * size + j]!=0){
          printf("%.2f  ", data[i * size + j]);}
      else{printf("   ");}*/
    }
    printf("\n");
  }
}


struct Inst
{
  float train_data[(TD_SIZE)];
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
  int ml_pos;
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
    ml_pos= copy.ml_pos;
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
    //std::cout<< "storeTick: "<< trueStoreTick << std::endl;
    //cout<< "Before: "<< pc << ", After: "<< getLine(pc) << endl;
    pc= getLine(pc); 
    assert(trueCompleteTick >= MIN_COMP_LAT || trueCompleteTick == 0);
    assert(trueStoreTick == 0 || trueStoreTick >= MIN_ST_LAT);
    //printf("loop start:train %.2f\n",trace[offset]);
    //int offset= index * TD_SIZE;
    int offset=0;
    //printf("Index: %d,",index);
    for (int i = 3; i < TD_SIZE; i++)
    {
      train_data[i] = trace[i+offset];
      printf("%.1f,",train_data[i]);
    }
    std::cout << std::endl;
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = 0; 
    isAddr = aux_trace[1];
    addr = aux_trace[2];
    addrEnd = aux_trace[3];
    for (int i = 0; i < 3; i++)
      iwalkAddr[i] = aux_trace[4 + i];
    for (int i = 0; i < 3; i++)
      dwalkAddr[i] = aux_trace[7 + i];
    //printf("%d,",index);
    //{printf("%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",index,train_data[3],train_data[4],train_data[5],train_data[6],train_data[18],train_data[19]);}
    for (int i=3;i<19;i++){
	  //printf("%.1f,",train_data[i]);
    }
    //pr`intf("\n");
    return true;
  }


bool batched_copy(float *trace, uint64_t *aux_trace, float *stream, int index){
  memcpy(stream, trace, sizeof(float)*TD_SIZE*Stream_width);
  //printf("Trace: \n");
  //display(trace, TD_SIZE, Stream_width);
  pc = aux_trace[0];
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

struct SQ {
  Inst insts[SQSIZE+1];
  int size= SQSIZE;
  int head = 0;
  int len = 0;
  int tail = 0;
  int sq_num = 0;
  __device__ int inc(int input) {
    if (input == SQSIZE)
      return 0;
    else
      return input + 1;
  }

  __device__ int dec(int input) {
    if (input == 0)
      return SQSIZE;
    else
      return input - 1;
  }

  __device__ bool is_empty() { return head == tail; }
  __device__ bool is_full() { return head == inc(tail); }
  __device__ Inst *add() {
    assert(!is_full());
    len+=1;
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
    len-=1;
  }

  __device__ int retire_until(Tick tick)
  {
    int retired = 0;
#ifdef DEBUG
    printf(" , %.3f, %.3f, %.3f\n", fetch_lat, complete_lat, store_lat);
	    printf("Head: %d, Head Tick: %lu, Tick: %lu\n",head,insts[head].completeTick,tick);
#endif
    //printf("tick: %lu, sotretick: %lu\n",tick,insts[head].storeTick);
    while (!is_empty() && insts[head].storeTick <= tick)
    {
           retire();
      retired++;
    }
    //printf("SQ size:%d, head: %d, tail:%d\n", SQSIZE,head,tail);
    assert(head <= SQSIZE);
    assert(tail <= SQSIZE);
    //printf("after: %d, retired: %d\n", head, retired);
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
    int __shared__ num[2];
    for (int i = 0; i < 3; i++) {
      //num[i]=0;
      iwalkAddr[i] = new_inst.iwalkAddr[i];
      dwalkAddr[i] = new_inst.dwalkAddr[i];
    }
    int W= threadIdx.x/WARPSIZE;
    int length= len;   
    int i= warpTID;
    int start_context = (dec(tail));
    int end_context = dec(head);
    num[W]=0;
    while(i < length) {
      int context = start_context - i;
      context = (context >= 0) ? context : context + SQSIZE+1;
      //printf("SQ: input context: %d\n",context);
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
      //memcpy(&input[poss*TD_SIZE],&insts[context].train_data,  sizeof(float)*TD_SIZE);
      //float *d_p= input + poss*TD_SIZE;
      //float *s_p= insts[context].train_data;
      //copier(d_p,s_p, TD_SIZE);
      i+= WARPSIZE;
    }
    sq_num= num[W];
    return num[W];
  }

  __device__ void update_fetch_cycle(Tick tick) {
    int warpTID = threadIdx.x % WARPSIZE;
    int length= len ;
    int context;
    int start_context = (dec(tail));
    int end_context = dec(head);
    int i=warpTID;
    //if(warpTID==0){printf("len: %d\n",length);}
    //for (; i != dec(head); i = dec(i)) {
    while(i < length){
      context = start_context - i;
      context = (context >= 0) ? context : context + SQSIZE+1;
      //printf("SQ: Context: %d, tick: %lu, %.2f\n", context, tick, insts[context].train_data[0]);
      insts[context].train_data[0] += tick;
       //printf("SQ: Context: %d, tick: %lu, %.2f\n", context, tick, insts[context].train_data[0]);
      assert(insts[context].train_data[0] >= 0.0);
      i+=WARPSIZE;
    }
  }
};


struct ROB
{
  Inst insts[ROBSIZE+1];
  int size= ROBSIZE;
  int head = 0;
  int tail = 0;
  int ml_head = Stream_width;   // row ID
  int ml_tail = Stream_width;   // row ID
  int len = 0;
  int rob_num=0;
  float *input_Ptr;
  Tick curTick=0;
  Tick curTick_d=0;
  Tick lastFetchTick=0;
  __host__ __device__ void init(int h, int t, int width)
  {
    head=h; tail=t;
  }


//__host__ __device__ int update_ml_tail(int index){ml_head=ml_head-1;}
  __host__ __device__ int inc(int input)
  {
	  //ml_head= ml_head - 1;
    if (input == (ROBSIZE)){
      return 0;}
    else{ 
      return input + 1;}
  }
 

  __host__ __device__ int dec(int input)
  {
    if (input == 0){
     	    return (ROBSIZE);}
    else{
            return input - 1;}
  }
  __host__ __device__ bool is_empty() { return head == tail; }
  __host__ __device__ bool is_full() { return head == inc(tail); }

  __host__ __device__ int get_index(int context){
  printf("context: %d, ml_context: %d\n", context, insts[context].ml_pos);
  //return 0; 
  return tail;
  }

  __host__ __device__ Inst *add()
  {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    insts[old_tail].ml_pos=ml_tail;
    printf("tail: %d, ml_tail: %d, len: %d\n", tail, ml_tail, len);
    //printf("%p\n",insts[old_tail]);
    //printf("old_tail: %d, tail: %d, ml_tail: %d, written: %d\n",old_tail,tail,ml_tail, insts[old_tail].ml_pos);
    ml_tail-=1;
    //assert(head <=size);
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
    //printf("%d or %d retired.\n", head,ml_head);
    head = inc(head);
    ml_head-=1;
    //assert(head <=size);
    len -= 1;
    //printf("%d or %d retired.\n", head);
#ifdef DEBUG
    printf("len decreased to retire: %d\n",len);
#endif
  }


__device__ int retire_until(Tick tick, SQ *sq = nullptr) {
	   int retired=0;
	            while (!is_empty() && insts[head].completeTick <= tick &&
             retired < RETIRE_BANDWIDTH) {
		         if (insts[head].isStore()) {
		          if (sq->is_full()){
            		  break;}
          Inst *newInst = sq->add();
	            newInst->init(insts[head]);
        }
	        retire();
        retired++;
      }
	       return retired;
   }


  __device__ Inst &tail_inst() { return insts[dec(tail)]; }

    
  __device__ void update_fetch_cycle(Tick tick, float *inputs)
  {
    int warpTID = threadIdx.x % WARPSIZE;
    assert(!is_empty());
    int context;
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    int length = len-1;
    int i = warpTID;
    while (i < length) {
      context= start_context - i;
      context= (context >= 0) ? context : context + ROBSIZE + 1;
      int ml_context= insts[context].ml_pos;
      float *ml_context_address= inputs + ml_context * TD_SIZE;
      //printf("I:%d,  Context: %d, previous: %.2f\n",i, context, insts[context].train_data[0]);
      //printf("Context: %d, Before, %.3f, %.3f, Next: %d\n", context, inst[0], inst[1], dec(i - 32));
      //insts[context].train_data[0] += tick;
      //ml_context_address[0] += tick; 
      //printf("ROB: Context: %d, tick: %lu, %.2f\n", context, tick, insts[context].train_data[0]);
      //assert(ml_context_address[0] >= 0.0);
      i += WARPSIZE;
    }
    __syncwarp();
    //if(warpTID==0){printf("ROB: head: %d, tail: %d \n", head, tail);}
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
    if(warpTID==0){printf("Here. Head: %d, Tail: %d, dec(tail): %d, len: %d\n",head,tail,dec(tail),len-1);}
#endif
    __syncwarp();
    assert(!is_empty());
    //assert(&new_inst == &insts[dec(tail)]);
    __shared__ int num[4];
    int length= len - 1; 
    //printf("make %p \n",&new_inst);   
    Addr pc = new_inst.pc;
    int isAddr = new_inst.isAddr;
    Addr addr = new_inst.addr;
    Addr addrEnd = new_inst.addrEnd;	
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
	          //num[i]=0;
	          iwalkAddr[i] = new_inst.iwalkAddr[i];
		        dwalkAddr[i] = new_inst.dwalkAddr[i];
			    }
          //printf("Starting\n");
    __syncwarp();
    int i = warpTID;
   num[W]=1;
   while (i < length)
    {
      int context = start_context - i;
      context = (context >= 0) ? context : context + ROBSIZE + 1;
      int ml_context= insts[context].ml_pos;
      //printf("context: %d,  ml_context: %d\n", context,ml_context);
      float *ml_context_address= inputs + ml_context * TD_SIZE; 
      //insts[context].train_data[ILINEC_BIT] = insts[context].pc == pc ? 1.0 : 0.0;
      ml_context_address[ILINEC_BIT] = insts[context].pc == pc ? 1.0 : 0.0; 
      int conflict = 0;
       for (int j = 0; j < 3; j++){
         if (insts[context].iwalkAddr[j] != 0 && insts[context].iwalkAddr[j] == iwalkAddr[j])
         conflict++;
       }
      //printf("context: %d, conflict: %d\n", context, conflict);
      //insts[context].train_data[IPAGEC_BIT] = (float)conflict;
      ml_context_address[IPAGEC_BIT] = (float)conflict;
      ml_context_address[DADDRC_BIT] = (isAddr && insts[context].isAddr && addrEnd >= insts[context].addr && addr <= insts[context].addrEnd) ? 1.0 : 0.0;
      ml_context_address[DLINEC_BIT] = (isAddr && insts[context].isAddr && (addr & ~0x3f) == (insts[context].addr & ~0x3f)) ? 1.0 : 0.0; 
      conflict = 0;
      if (isAddr && insts[context].isAddr)
        for (int j = 0; j < 3; j++)
        {
        if (insts[context].dwalkAddr[j] != 0 && insts[context].dwalkAddr[j] == dwalkAddr[j])  
            conflict++;
        }
      ml_context_address[DPAGEC_BIT] = (float)conflict;     
      #ifdef DEBUG
      printf("context: %d,ilinec: %.2f,ipagec: %.2f,daddr: %.2f,dlinec: %.2f,dpagec: %.2f\n",context,insts[context].train_data[ILINEC_BIT],insts[context].train_data[IPAGEC_BIT],insts[context].train_data[DADDRC_BIT],insts[context].train_data[DLINEC_BIT],insts[context].train_data[DPAGEC_BIT]);
	#endif
      //int poss= atomicAdd(&num[W], 1); 
     
      //printf("Context: %d, poss: %d\n",context,poss);
      //memcpy(&inputs[poss*TD_SIZE],&insts[context].train_data,  sizeof(float)*TD_SIZE);
      int poss= atomicAdd(&num[W], 1); 
      //memcpy(&inputs[poss*TD_SIZE],&insts[context].train_data,  sizeof(float)*TD_SIZE);
      i += WARPSIZE;
      rob_num= num[W];
    }
       //printf("Here. 3\n");
    //__syncwarp();
    return rob_num;
  }
};


  __device__ void
  dis(float *data, int size, int rows)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < size; j++)
      {
          //if (data[i * size + j]!=0){
	  printf("%.0f  ", data[i * size + j]);
	  //else{printf("   ");}
      }
      printf("\n");
    }
  }
  

__global__ void result(ROB *rob_d, int Total_Trace, int instructions, Tick *sum)
{
  sum[0]=0;
  //printf("\n");
  for (int i = 0; i < Total_Trace; i++)
  {
    //printf("I: %d\n",i);
    ROB *rob= &rob_d[i];
#ifdef WARMUP
    //sum[0] += (rob->curTick - rob->curTick_d);
    if(i==0){
	    sum[0] += (rob->curTick_d); 
	    //printf("T: %d, Tick: %lu,Reduced: %lu final: %lu\n", i, rob->curTick, (rob->curTick - rob->curTick_d),rob->curTick_d);
    }
    else {
	    sum[0] += (rob->curTick - rob->curTick_d);
    	    //printf("T: %d, Tick: %lu,Reduced: %lu final: %lu\n", i, rob->curTick, rob->curTick_d, (rob->curTick - rob->curTick_d));
    }
#else
    sum[0] += rob->curTick;
    //if(rob->curTick>100000)printf("T: %d, Tick: %lu\n", i, rob->curTick);
#endif
  }
  printf("%llu,",sum[0]);
  //printf("~~~~~~~~~Instructions: %d, Batch: %d, Prediction: %lu ~~~~~~~~~\n", instructions,Total_Trace, sum[0]);
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
initialization(ROB *rob_d, int Total_Trace)
{
  ROB *rob;
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  int index = TID;
  while (index<Total_Trace){
    rob=&rob_d[index];
    rob->ml_head= WIDTH;
    rob->ml_tail=WIDTH;
    index += (gridDim.x * blockDim.x);
  }
}

__global__ void
update(ROB *rob_d, SQ *sq_d, float *input, float *output, int *status, int Total_Trace, int shape, int iteration, int W, int Batch_size, int *index_all)
//update(ROB *rob_d, SQ *sq_d, float *output, float *inputs, int *status, int Total_Trace, int shape)
{
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
  //if (TID==0){printf("Update started\n");}
  __syncthreads();
  int index = TID;
  ROB *rob; SQ *sq;
  while (index < Total_Trace)
  {
	
#if defined(COMBINED)
    int offset= index * shape;
#else
    int offset = index * shape;
#endif  
#ifdef DEBUG 
    if (threadIdx.x == 0)
	        {
	printf("Input_Ptr\n");
	dis(output, 33, 1);
			    }
    __syncwarp();
#endif
    Tick nextFetchTick = 0;
    rob= &rob_d[index];
    sq= &sq_d[index];    
    int tail= rob->dec(rob->tail);
    int tail_sq= sq->dec(sq->tail);
    int context_offset = rob->dec(rob->tail) * TD_SIZE;
    //float *inp=  inputs + (ML_SIZE+ Stream_width*TD_SIZE) * index;
    float *ml_context_address= input +  (rob->ml_tail+1) * TD_SIZE;
    #if defined(COMBINED)
    int classes[3];
    for(int i=0; i<3;i++)
    {
	     float max = output[offset + CLASS_NUM*i+3];
	   
	     int idx=0;
	     //printf("i: %d, max: %.2f\n", i ,max);
	     for (int j = 1; j < CLASS_NUM; j++) {
		     //printf("i: %d, max: %.2f, value: %.2f\n", i ,max,output[offset + 10*i+3+j]);
	       if (max < output[offset + CLASS_NUM*i+3+j]) {
                 max = output[offset + CLASS_NUM*i+3+j];
                 idx = j;
          }
	   classes[i]= idx;   
	     }
	 //printf("combined: class0: %d, class1: %d, class2: %d\n", classes[0],classes[1],classes[2]);
    }
    //printf("fclass: %d, cclass: %d\n", f_class, c_class);
#endif
    float fetch_lat = output[offset + 0];
    float complete_lat = output[offset + 1];
    float store_lat= output[offset + 2];
    int int_fetch_lat = round(fetch_lat);
    int int_complete_lat = round(complete_lat);
    int int_store_lat = round(store_lat);
     //printf("rob tail: %d, sq tail: %d,\n %.3f, %.3f, %.3f\n", rob->tail, sq->tail, fetch_lat, complete_lat, store_lat);    
    #ifdef DEBUG
    printf("%.3f, %.3f, %.3f\n",fetch_lat, complete_lat, store_lat);
    #endif 
#if defined(COMBINED)
      if (classes[0] <= 8){
        int_fetch_lat = classes[0];}
      if (classes[1] <= 8){
	//printf("complete\n");
        int_complete_lat = classes[1] + MIN_COMP_LAT;}
      if (classes[2] == 0){
        int_store_lat = 0;}
      else if (classes[2] <= 8 ){
        int_store_lat = classes[2] + MIN_ST_LAT - 1;}
      //printf("Combined: fetch: %d, complete: %d, store: %d\n",int_fetch_lat, int_complete_lat,int_store_lat);
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
    //float *inp=  inputs + (ML_SIZE+ WIDTH*TD_SIZE) * index; 
    //float *ml_context_address= inp +  (rob->ml_tail+1) * TD_SIZE;
    printf("ML tail: %d\n", rob->ml_tail);
    ml_context_address[0]=-int_fetch_lat;
    ml_context_address[1]=int_complete_lat;
    ml_context_address[2]=int_store_lat; 
    rob->insts[tail].train_data[0] = -int_fetch_lat;
    rob->insts[tail].train_data[1] = int_complete_lat;
    rob->insts[tail].train_data[2] = int_store_lat;
      

#ifdef WARMUP
    if ((iteration<W) && (index!=0)){
	    //{printf("%d,%d,%d,%d,%d,%d,%d,%lu [Warmup]\n",index ,index_all[index],-int_fetch_lat, int_complete_lat, int_store_lat,rob->rob_num,sq->sq_num, rob->curTick);}
    }
    
    else if((iteration>=Batch_size) && (index==0)){
	    //printf("%d,%d,%d,%d,%d,%d,%d,%lu [Warmup]\n",index ,index_all[index],-int_fetch_lat, int_complete_lat, int_store_lat,rob->rob_num,sq->sq_num, rob->curTick);
    }
   
else {
      //printf("%d,%d,%d,%d,%lu\n",index,index_all[index],-int_fetch_lat,rob->rob_num,rob->curTick);
    }

#else
{
//printf("%d,%d,%d,%d,%d,%d,%d,%lu\n",index ,index_all[index],-int_fetch_lat, int_complete_lat, int_store_lat,rob->rob_num,sq->sq_num,rob->curTick);
//printf("%d,%d,%d,%d,%lu\n",index,index_all[index],-int_fetch_lat,rob->rob_num,rob->curTick);
printf("%d,%d,%d,%d,%lu\n",-int_fetch_lat,int_complete_lat, int_store_lat,rob->rob_num,rob->curTick);
printf("%.0f,%.0f,%.0f\n", ml_context_address[0], ml_context_address[1], ml_context_address[2]);
}
    #endif


    //{printf("%d,%d,%d,%d,%d,%d,%d,%lu\n",index ,index_all[index],-int_fetch_lat, int_complete_lat, int_store_lat,rob->rob_num,sq->sq_num,rob->curTick);}
    //printf(",%d,%d,%d\n", -int_fetch_lat, int_complete_lat, int_store_lat);
    //if((index_all[index]> 40271870) && (index_all[index]<40271880))
    //{printf(",%d,%d,%d,%d,%d,%d,%d,%lu\n",index ,index_all[index],-int_fetch_lat, int_complete_lat, int_store_lat,rob->rob_num,sq->sq_num, rob->curTick);}
    rob->insts[tail].storeTick = rob->curTick +  int_fetch_lat + int_store_lat;
    rob->insts[tail].completeTick = rob->curTick + int_fetch_lat + int_complete_lat + 1;
    rob->lastFetchTick = rob->curTick;
    //printf("%lu, %lu, %lu\n", rob->insts[tail].completeTick, rob->insts[tail].storeTick, rob->lastFetchTick);
    if (int_fetch_lat)
    {
      status[index]=1;
      nextFetchTick = rob->curTick + int_fetch_lat;
        //printf("Break with int fetch, set status : %d\n",status[index]);
    }
    else if(rob->is_full()){
	    //printf("RoB full\n");
	    status[index]=1;
    }
    else{
	    //printf("continue loop without update\n");
	    status[index]=0; index += (gridDim.x * blockDim.x);continue;}
    int count=0;
    do{
            if(count>0){
	      //assert((sq->head) 
              //printf("retiring..\n SQ %p: head: %d, tail: %d\n",sq,sq->head,sq->tail);
	      //int sq_retired = sq->retire_until(temp);
	      //int rob_retired = rob->retire_until(rob->curTick,sq);
	      //int_fetch_lat=0;
      	      //printf("SQ: %p, retired: %d, rob retired: %d\n",sq,sq_retired, rob_retired);
      }
    //printf("head: %d, curTick: %lu \n", rob->head, rob->curTick);
    //printf("Rob full: %d\n", rob->is_full());
	    if ( int_fetch_lat)
    {
      Tick nextCommitTick= max_(rob->getHead()->completeTick, rob->curTick + 1);
      rob->curTick= min_(nextCommitTick, nextFetchTick);
      //printf("%d,1\n",index_all[index]);
      //printf("case 1 cur = %lu\n",rob->curTick);
    }
    else if (rob->curTick < nextFetchTick)
    {
      Tick nextCommitTick= max_(rob->getHead()->completeTick, rob->curTick + 1);
      rob->curTick= min_(nextCommitTick, nextFetchTick);
      //printf("%d,2\n",index_all[index]);
      //printf("case 2 cur = %lu, nextcommit= %lu\n",rob->curTick, nextCommitTick);
    }
    else if (rob->is_full())
    {
      rob->curTick =  max_(rob->getHeadTick(), rob->curTick + 1);
      //printf("%d,3\n",index_all[index]);
	//printf("case 3 cur = %lu\n",rob->curTick);
    }
    else{
	rob->curTick =  max_(rob->getHeadTick(), rob->curTick + 1);
	//printf("case 4 cur = %lu\n",rob->curTick);
	//printf("%d,4\n",index_all[index]);
    }
    int_fetch_lat= 0;
    //printf("nextFetch: %lu\n", nextFetchTick);
    //int temp= rob->curTick;
    count++;
    if(status[index]==1){
    	int sq_retired = sq->retire_until(rob->curTick);
    	int rob_retired = rob->retire_until(rob->curTick,sq);
	//printf("SQ retired: %d, ROB Retired: %d\n", sq_retired, rob_retired);
        //printf("Retire until: %ld, Retired: %d\n",curTick, retired);
    	int_fetch_lat=0;}
              //printf("SQ: %p, retired: %d, rob retired: %d\n",sq,sq_retired, rob_retired);
    //printf("Rob full?: %d, curtick and next: %d\n",rob->is_full(),rob->curTick>=nextFetchTick);
    
    } while (!(rob->curTick >=nextFetchTick) || rob->is_full());
    index += (gridDim.x * blockDim.x);
  }
   //if (TID==0){printf("Update completed\n");}
     __syncthreads();
}



  __global__ void shift(float *source, float *destination, ROB *rob_d, int Total_Trace){
  //if(threadIdx.x==0){printf("shifted, head:%d, tail:%d\n", ml_head, ml_tail);}
    int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (TID==0){printf("sift started: source: %p, destination: %p, offset: %d\n",source,destination,TD_SIZE* (Stream_width));}
   //  __syncthreads();
  int warpID = TID / WARPSIZE;
  int warpTID = TID % WARPSIZE;
  int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
   int index= warpID;
    while(index < Total_Trace){
    	ROB *rob= &rob_d[index];
	int total= TD_SIZE*(rob->ml_head+1);
	int offset= TD_SIZE* (Stream_width);
    	int i=warpTID;	
	while (i<total){       
	  destination[i+offset]= source[i];
	  printf("source: %f, destination: %f\n",destination[i+offset],source[index] );
	  i+=WARPSIZE;	
	}
  	// Update the ml index for each instruction in ROB
  	int start= threadIdx.x + rob->ml_tail+1;
  	while(start<=rob->ml_head){
    		//printf("Previous: %d, new: %d\n",insts[start].ml_pos, insts[start].ml_pos+WIDTH);
    		rob->insts[start].ml_pos+=Stream_width;
    		start+=WARPSIZE;
  	}
  	rob->ml_tail=rob->ml_tail+Stream_width;
  	rob->ml_head=rob->ml_head+Stream_width;
    index+=TotalWarp;
    }
  }


__global__ void
preprocess(ROB *rob_d,SQ *sq_d, Inst *insts, float *default_val, float *inputPtr, int *status, int Total_Trace, int *index_all, int iteration, int W, int Batch_size, int window_index)
//preprocess(ROB *rob_d,SQ *sq_d, Inst *insts, float *default_val, float *inputPtr,  int *status, int Total_Trace)
{ 
  int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
   //if (TID==0){printf("Update started\n");}
   //  __syncthreads();
  int warpID = TID / WARPSIZE;
  int warpTID = TID % WARPSIZE;
  int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
  int index, Total;
  //int retired=0;
  ROB *rob;
  SQ *sq;
  float *input_device, *input_window;
#ifdef WARP
  index = warpID;
  Total = TotalWarp;
#else
  index = blockIdx.x;
  Total = gridDim.x;
#endif

 while (index < Total_Trace)
  {
    rob= &rob_d[index];
    sq= &sq_d[index];
#ifdef WARMUP
      if ((iteration==W) && (index!=0)){ // Change W to batch_size??
	 rob->curTick_d= rob->curTick;}

      if ((iteration==Batch_size) && (index==0))
	  {rob->curTick_d= rob->curTick;}
#endif    
    Tick curTick = rob->curTick;
    Tick lastFetchTick = rob->lastFetchTick;
    //printf("Index: %d\n",index);
    input_device= inputPtr + (ML_SIZE+ WIDTH*TD_SIZE) * index;   
    //int old_head= rob->head;
    //printf("InputPtr: %p\n", input_Ptr);
    
        
    if(warpTID==0){
            Inst *newInst = rob->add();
	    //printf("head: %d, tail: %d, ml_head:%d, ml_tail:%d \n", rob->head, rob->tail, rob->ml_head, rob->ml_tail);
            memcpy(newInst, &insts[index], sizeof(Inst));
            rob->insts[rob->tail-1].ml_pos=rob->ml_tail;
	    //printf("mem copied.. \n");
            //inst_copy(&rob->insts[rob->tail],&insts[index]);  
    }
    __syncwarp();
    //printf("Curtick: %ld, lastFetchTick: %ld\n", curTick, lastFetchTick);
    if (curTick != lastFetchTick)
    {
      rob->update_fetch_cycle(curTick - lastFetchTick, input_device);
      __syncwarp();
      sq->update_fetch_cycle(curTick - lastFetchTick);
    }
   //if(warpTID==0){ printf("both retired.. \n");} 
    __syncwarp();
    int rob_num= rob->make_input_data(input_device, curTick, insts[index]);
        __syncwarp();
    //int sq_num= sq->make_input_data(input_Ptr + rob_num * TD_SIZE, curTick, insts[index]);
    
	__syncwarp();
    //int num= rob_num + sq_num;
    // copy default values
/*    
if(num < CONTEXTSIZE && warpTID==0)
    {
     printf("%d, %d,",rob_num,sq_num); 
     memcpy(input_Ptr+num*TD_SIZE, default_val +num*TD_SIZE, sizeof(float)*(CONTEXTSIZE-num)*TD_SIZE);
        }
*/

	/*
    if (warpTID==0){printf("%d, %d\n",rob_num,sq_num);}
if(num < CONTEXTSIZE)
    {
     float *d_p= input_Ptr+(num+rob->ml_head)*TD_SIZE;
     float *s_p= default_val+num*TD_SIZE;
     copier(d_p,s_p, (CONTEXTSIZE-num)*TD_SIZE);
     int i= warpTID;
     while(i<(CONTEXTSIZE-num)*TD_SIZE)
     {
	     d_p[i]=0;
     i+=WARPSIZE;
     }
     __syncwarp();
     //printf("default value copied.. \n");
    }
     */
    if (warpTID == 0){  
      printf("stream width:%d, window_index: %d, offset: %d\n", Stream_width, window_index, window_index*TD_SIZE);
      printf("Input device:%p, input window: %p\n", input_device, input_window);
      dis(input_device, TD_SIZE, Stream_width+3);
    }
   __syncwarp();
   index += Total;
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
  //int instr= instructions/10;
  //Tick r=0;
  //for (int i=0; i<10;i++){
  //unsigned long int tot= TRACE_DIM * instr;
  fread(trace, sizeof(float), TRACE_DIM * instructions, trace_f);
  //trace+=r;
  //assert(r=tot);
  //}
  //printf("tot: %lu, Toread: %lu, read :%lu values for trace.\n",tot,TRACE_DIM * instructions, r);
  //display(trace,TRACE_DIM,2);

  FILE *aux_trace_f = fopen(aux_trace_file, "rb");
  if (!aux_trace_f)
  {
    printf("Unable to aux_trace binary.");
    return 1;
  }
  int k = fread(aux_trace, sizeof(Tick), AUX_TRACE_DIM * instructions, aux_trace_f);
  //printf("read :%d values for aux_trace.\n", k);
  //display(aux_trace,AUX_TRACE_DIM,2);
  return true;
}

#endif
