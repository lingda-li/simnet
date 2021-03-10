#ifndef SIM_H
#define SIM_H

#define TD_SIZE 51
#define INST_SIZE 51
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
#define WARPS 1
#define TRACE_DIM 51
#define AUX_TRACE_DIM 10
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6
typedef long unsigned Tick;
typedef long unsigned Addr;
float factor[TD_SIZE];
float mean[TD_SIZE];
float default_val[TD_SIZE];
Addr getLine(Addr in) { return in & ~0x3f; }

struct Inst
{
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
  int offset;
  //H_ERR(cudaMalloc((void **)&train_data_d, sizeof(int)*TD_SIZE));
  // Read simulation data.

  void
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

  Addr getLine(Addr in) { return in & ~0x3f; }

  bool read_sim_mem(float *trace, uint64_t *aux_trace, int index)
  {
    //train_data = train_d;
    trueFetchClass = trace[0];
    trueFetchTick = trace[1];
    trueCompleteClass = trace[2];
    trueCompleteTick = trace[3];
    pc = aux_trace[0];
    //cout<< "Before: "<< pc << ", After: "<< getLine(pc) << endl;
    pc= getLine(pc);
    offset = INST_SIZE * index;
    //cout<< "Offset: "<< offset <<"   Memory: "<<train_data;
    //assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++)
    {
      train_data[i] = trace[i] / factor[i];
      //cout<< trace[i]<<"\t" << train_data[i+offset]<<"\n";
    }
    train_data[0] = train_data[1] = 0.0;
    train_data[2] = train_data[3] = 0.0; 
    isAddr = aux_trace[1];
    addr = aux_trace[2];
    addrEnd = aux_trace[3];
    for (int i = 0; i < 3; i++)
      iwalkAddr[i] = aux_trace[4 + i];
    for (int i = 0; i < 3; i++)
      dwalkAddr[i] = aux_trace[7 + i]; 
    for (int i = 0; i < TD_SIZE; i++)
    {
    } //cout << train_data[i] << " ";
      //cout << "\n";
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
  __host__ __device__ int inc(int input)
  {
    if (input == (ROBSIZE-1)){
      //printf("Input: 400, Inc from %d to %d\n", input, 0);
      return 0;}
    else{
      //printf("Inc from %d to %d\n", input, input+1);
      return input + 1;}
  }
 

  __host__ __device__ int dec(int input)
  {
    if (input == 0){
      //printf("Input: 0, Dec from %d to %d\n", input, ROBSIZE);
	    return (ROBSIZE-1);}
    else{
      //printf("Dec from %d to %d\n", input, input-1);
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
    printf("len decreased to retire: %d\n",len);
  }

  __device__ int retire_until(Tick tick)
  {
    int retired = 0;
    printf("Head: %d, Head Tick: %lu, Tick: %lu\n",head,insts[head].completeTick,tick);
    while (!is_empty() && insts[head].completeTick <= tick)
    {
      printf("Retire\n");
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
        printf("%.3f  ", data[i * size + j]);
      }
      printf("\n");
    }
  }

  
  __device__ void update_fetch_cycle(Tick tick, Tick curTick, float *factor)
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
      //printf("ROB:, head: %d, tail: %d \n", head, tail);
      //dis(insts, INST_SIZE, 4);
    }
    __syncwarp();
    while (i < length)
    {
      //printf("I: %d\n",i);
      context = start_context - i;
      context = (context >= 0) ? context : context + ROBSIZE;
      //printf("warpTID:%d, Context: %d, curTick: %ld, %.2f\n", warpTID, context, curTick, inst[COMPLETETICK]);
      if (insts[context].completeTick <= curTick)
      {
        //printf("COntext: %d, warpTID: %d, Curtick: %ld, Inst: %.2f,continue\n", context, warpTID, curTick, inst[COMPLETETICK]);
        i += WARPSIZE;
        continue;
      }
      //printf("Context: %d, Before, %.3f, %.3f, Next: %d\n", context, inst[0], inst[1], dec(i - 32));
      insts[context].train_data[0] += tick / factor[0];
      if (insts[context].train_data[0] >= 9 / factor[0])
        insts[context].train_data[0] = 9 / factor[0];
      insts[context].train_data[1] += tick / factor[1];
      assert(insts[context].train_data[0] >= 0.0);
      assert(insts[context].train_data[1] >= 0.0);
      i += WARPSIZE;
    }
    __syncwarp();
  }

  __device__ int make_input_data(float *inputs, Tick tick, float *factor, float *default_val)
  { 
    int TID = (blockIdx.x * blockDim.x) + threadIdx.x;
    int warpID = TID / WARPSIZE;
    int warpTID = TID % WARPSIZE;
    int curr = dec(tail);
    int start_context = dec(dec(tail));
    int end_context = dec(head);
    int W= threadIdx.x/WARPSIZE;
    if(warpTID==0){printf("Here. Head: %d, Tail: %d, dec(tail): %d, len: %d\n",head,tail,dec(tail),len);}
    __syncwarp();
    assert(!is_empty());
    saturated = false;
    __shared__ int num[4];
    int length= len - 1;
       
    Addr pc = insts[dec(tail)].pc;
    int isAddr = insts[dec(tail)].isAddr;
    Addr addr = insts[dec(tail)].addr;
    Addr addrEnd = insts[dec(tail)].addrEnd;
    //if(warpTID==0){printf(" ROB:%p, PC: %ld, isAddr: %ld, addr: %ld\n",(void *)&insts[dec(tail)], pc, isAddr, addr);}
    Addr iwalkAddr[3], dwalkAddr[3];
    for (int i = 0; i < 3; i++) {
      num[i]=1;
      iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
      dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
    }
   if(warpTID==0){memcpy(inputs, insts[dec(tail)].train_data, sizeof(float)*TD_SIZE);}
   //printf("%.2f, %.2f\n",inputs[warpTID],insts[dec(tail)].train_data[warpTID]); 
   __syncwarp();
    int i = warpTID;
    //if(warpTID==0){printf("Inst\n");dis(insts[dec(tail)].train_data,TD_SIZE,1);}
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
      insts[context].train_data[ILINEC_BIT] = insts[context].pc == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
       int conflict = 0;
       for (int j = 0; j < 3; j++){
         if (insts[context].iwalkAddr[j] != 0 && insts[context].iwalkAddr[j] == iwalkAddr[j])
         conflict++;
#ifdef DEBUG
	 printf("context: %d, j: %d, %lu, %lu, %lu,conflict: %d\n", context,j,insts[context].iwalkAddr[j],insts[context].iwalkAddr[j],iwalkAddr[j] ,conflict);
#endif
       }
      //printf("context: %d, conflict: %d\n", context, conflict);
      insts[context].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
      insts[context].train_data[DADDRC_BIT] = (isAddr && insts[context].isAddr && addrEnd >= insts[context].addr && addr <= insts[context].addrEnd) ? 1.0 / factor[DADDRC_BIT] : 0.0;
      insts[context].train_data[DLINEC_BIT] = (isAddr && insts[context].isAddr && (addr & ~0x3f) == (insts[context].addr & ~0x3f)) ? 1.0 / factor[DLINEC_BIT] : 0.0; 
      conflict = 0;
      if (isAddr && insts[context].isAddr)
        for (int j = 0; j < 3; j++)
        {
        if (insts[context].dwalkAddr[j] != 0 && insts[context].dwalkAddr[j] == dwalkAddr[j])  
            conflict++;
        }
      insts[context].train_data[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];     
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
    //printf("Here. 3\n");
    __syncwarp();
    return 0;
  }
};

#endif
