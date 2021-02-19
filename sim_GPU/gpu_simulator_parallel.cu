#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
//#include "init.cuh"
//#include <torch/script.h> // One-stop header.
#include "wtime.h"
using namespace std;
#define NO_MEAN
#define GPU
typedef long unsigned Tick;
typedef long unsigned Addr;
#define TD_SIZE 51
#define ROBSIZE 400
#include "herror.h"
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define TD_SIZE 51
#define INST_SIZE 62
#define CONTEXTSIZE 111
#define ROBSIZE 400
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define ML_SIZE (TD_SIZE * CONTEXTSIZE)
#define MIN_COMP_LAT 6
#define WARP
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
#define TRACE_DIM 39
#define AUX_TRACE_DIM 10
//#define Total_Trace 1024

Tick Num = 0;

float factor[TD_SIZE];
float mean[TD_SIZE];
float default_val[TD_SIZE];

struct params{
   bool is_empty;
   bool is_full;
   int saturated;
   Tick fetched;
   Tick completeTick;
   bool eof;
   int ROB_flag;
};

class Inst {
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
  Inst(){} 
  Inst(float *pointer){
	train_data= pointer;
   }
  bool read_sim_data(ifstream &trace, ifstream &aux_trace, float *train_d, int index) { 
    train_data= train_d;
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    offset= INST_SIZE * index;
    //cout<< "Offset: "<< offset <<"   Memory: "<<train_data;
    assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++) {
      trace >> train_data[i+offset];
      train_data[i+offset] /= factor[i];
    }
    train_data[0 + offset] = train_data[1 + offset] = 0.0;
    train_data[2 + offset] = train_data[3 + offset] = 0.0;
    aux_trace >> isAddr >> addr >> addrEnd;
    for (int i = 0; i < 3; i++)
      aux_trace >> iwalkAddr[i];
    for (int i = 0; i < 3; i++)
      aux_trace >> dwalkAddr[i];
    train_data[PC]=pc;
    train_data[ISADDR]=isAddr;
    train_data[ADDR]=addr;
    train_data[ADDREND]=addrEnd;
    train_data[IWALK0]=iwalkAddr[0];
    train_data[IWALK1]=iwalkAddr[1];
    train_data[IWALK2]=iwalkAddr[2];
    train_data[DWALK0]=dwalkAddr[0];
    train_data[DWALK1]=dwalkAddr[1];
    train_data[DWALK2]=dwalkAddr[2];
    assert(!trace.eof() && !aux_trace.eof());
    //cout << "in: ";
    //for (int i = 0; i < TD_SIZE; i++)
    //  cout << train_data[i] << " ";
    //cout << "Read complete\n";
    //H_ERR(cudaMemcpy(train_data_d, train_data, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
    return true;
  }

 bool read_sim_mem(float *trace, uint64_t *aux_trace, float *train_d, int index) {
    train_data= train_d;
    //trace >> trueFetchClass >> trueFetchTick;
    //trace >> trueCompleteClass >> trueCompleteTick;
    trueFetchClass= trace[0];
    trueFetchTick= trace[1];
    trueCompleteClass= trace[2];
    trueCompleteTick= trace[3];
    pc= aux_trace[0];
    
    
    offset= INST_SIZE * index;
    //cout<< "Offset: "<< offset <<"   Memory: "<<train_data;
    //assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++) {
      train_data[i+offset]= trace[i]/factor[i];
      //train_data[i+offset] /= factor[i];
    }
    train_data[0 + offset] = train_data[1 + offset] = 0.0;
    train_data[2 + offset] = train_data[3 + offset] = 0.0;
    //aux_trace >> isAddr >> addr >> addrEnd;
    isAddr= aux_trace[1];
    addr= aux_trace[2];
    addrEnd= aux_trace[3];
    for (int i = 0; i < 3; i++)
      iwalkAddr[i]=aux_trace[3+i];
    for (int i = 0; i < 3; i++)
      dwalkAddr[i]=aux_trace[6+i];
    train_data[PC]=pc;
    train_data[ISADDR]=isAddr;
    train_data[ADDR]=addr;
    train_data[ADDREND]=addrEnd;
    train_data[IWALK0]=iwalkAddr[0];
    train_data[IWALK1]=iwalkAddr[1];
    train_data[IWALK2]=iwalkAddr[2];
    train_data[DWALK0]=dwalkAddr[0];
    train_data[DWALK1]=dwalkAddr[1];
    train_data[DWALK2]=dwalkAddr[2];
    //assert(!trace.eof() && !aux_trace.eof());
    return true;
 }	
};

class ROB{
public:
    //Inst insts[ROBSIZE+1];
    float *insts;
    int head= 0;
    int tail= 0;
    bool saturated= false; 
    void init(){
        //insts.init();
        //H_ERR(cudaMalloc((void **)&train_data, sizeof(int)*(ROBSIZE +1)));
    }
    ~ROB(){};
    ROB(){
        H_ERR(cudaMalloc((void **)&insts, sizeof(float)*(ROBSIZE*INST_SIZE)));
    };
    __host__ __device__ int inc(int input) {
        if (input == ROBSIZE)
          return 0;
        else
          return input + 1;
    }

    __host__ __device__ int dec(int input) {
        if (input == 0)
          return ROBSIZE;
        else
          return input - 1;
    }
    __host__ __device__ bool is_empty() { return head == tail; }
    __host__ __device__ bool is_full() { return head == inc(tail); }

__host__ __device__
     void add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    //printf("index updated.\n");
    //return &insts[old_tail];
  }
    /*
    __device__
    Inst *getHead() {
        return &insts[head];
      }
      */

__device__ void
	retire(){
		assert(!is_empty());
		head= inc(head);
	}

 __device__
 int retire_until(Tick tick, float *insts) {
	//printf("Head: %d\n",head);
	 int completeTick;
	int retired = 0;
	while (!is_empty() && insts[COMPLETETICK] <= tick) {
		retire();
		retired++;
	}
	return retired;
 }

/*
	  __device__
    void update_fetch_cycle(Tick tick, Tick curTick, float *factor) {
        int TID= (blockIdx.x * blockDim.x) + threadIdx.x;
	//int warpID= TID / WARPSIZE;
	int  warpTID= TID/ WARPSIZE;

    	assert(!is_empty());
        int start = dec(dec(tail));
        int end= dec(head);
        //for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
        //printf("start: %d, end: %d\n",start,end);
        int i= (start - warpTID); 
	
	//for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
          while(i>end){
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

*/      
__device__ 
	  int make_input_data(float *context, float *insts, Tick tick, float *factor, float *default_val) {
 	//if(){printf("Here. Head: %d, Tail: %d\n",head,tail);}

 	int TID= (blockIdx.x * blockDim.x) + threadIdx.x;
	int warpID= TID / WARPSIZE;
	int  warpTID= TID % WARPSIZE;
	int offset;
#ifdef WARP
	offset= warpID;
#else
	offset= blockIdx.x;
#endif
 	int curr= dec(tail);
	int start_context= dec(dec(tail));
	int end_context= dec(head);
	//insts= insts + offset + INST_SIZE;
	//if(warpTID==0){printf("Here. Head: %d, Tail: %d\n",head,tail);}
	assert(!is_empty());
        saturated = false;
	__shared__ int num[4];
        Addr pc = insts[curr * INST_SIZE + PC];
        int isAddr= insts[curr * INST_SIZE + ISADDR];
        Addr addr = insts[curr * INST_SIZE + ADDR];
        Addr addrEnd = insts[curr * INST_SIZE + ADDREND];
        Addr iwalkAddr[3], dwalkAddr[3];
        int i= warpTID;
	//if (warpTID==0){
	while(i<3){
	//for (int i = 0; i < 3; i++) {
          iwalkAddr[i] = insts[curr*INST_SIZE + IWALK0 + i];
          dwalkAddr[i] = insts[curr*INST_SIZE + DWALK0 + i];
        i++;
	}
	__syncwarp();
	
	//int start = dec(dec(tail)); int end= dec(head);
        //int num= end-start;
        if(warpTID==0){printf("Here. Head: %d, Tail: %d,current:%d, start: %d, end: %d, curr: %d \n",head,tail,curr,start_context,end_context,dec(tail));}	
	i= start_context - warpTID;
	while(i > end_context){  
	  printf("ThreadID: %d, inst id: %d\n",warpTID, i);
	  if (insts[i*INST_SIZE+COMPLETETICK] <= tick)
            continue;
          if (num[warpID] >= CONTEXTSIZE) {
            saturated = true;
            return 0;
          }
          // Update context instruction bits.
          insts[i*INST_SIZE+ ILINEC_BIT] = insts[i*INST_SIZE+PC] == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
          int conflict = 0;
          for (int j = 0; j < 3; j++) {
            if (insts[i*INST_SIZE+j] != 0 && insts[i*INST_SIZE+j] == iwalkAddr[j])
              conflict++;}
          //insts[i].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
          //insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].train_data[ISADDR] && addrEnd >= insts[i].train_data[ADDR] && addr <= insts[i].train_data[ADDREND]) ? 1.0 / factor[DADDRC_BIT] : 0.0;
          //insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].train_data[ISADDR] && (addr & ~0x3f) == (insts[i].train_data[ADDR] & ~0x3f)) ? 1.0 / factor[DLINEC_BIT] : 0.0;
          conflict = 0;
          if (isAddr && insts[i*INST_SIZE+ISADDR])
            for (int j = 0; j < 3; j++) {
              if (insts[i*INST_SIZE+j] != 0 && insts[i*INST_SIZE+j] == dwalkAddr[j])
                conflict++;}
          insts[i*INST_SIZE+DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
          //std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
          //num++;
	  atomicAdd(&num[warpID],1);
        i-=WARPSIZE;
	}
	__syncwarp();
       //if(warpTID==0){printf("Here. Head: %d, Tail: %d, start: %d, end: %d, curr: %d \n",head,tail,start,end,dec(tail));}	
	/* Data copy: current instruction and ROB instructions*/
	/*
	int j= warpTID;
	while(j>=end_context)
	{
		if(warpTID==0){ printf("Working on context: %d\n",j);}
		i= warpTID;	 
		while(i<TD_SIZE){
			context[i]= insts[j*INST_SIZE+i];
			i+=WARPSIZE;
		}
		j-=1;
	}
	*/
        i= warpTID;
        while (i<TD_SIZE){
                //for (int i = num; i < CONTEXTSIZE; i++) { //printf("thread: %d, i: %d\n",warpTID,i);
                //if(warpTID==0){printf("");}
                int j= curr;
                while(j!= end_context){
                        context[i+j*TD_SIZE]= insts[j*INST_SIZE+i];
                        //printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", j,i,i+j*TD_SIZE,warpTID, default_val[i]);
                        j=dec(j);}
        i+=WARPSIZE;}
	__syncwarp();

	//printf("Adding default values.\n");
	i= warpTID;
	while (i<TD_SIZE){
        	//for (int i = num; i < CONTEXTSIZE; i++) { //printf("thread: %d, i: %d\n",warpTID,i);
		//if(warpTID==0){printf("");}
		int j= 1;
		while(j< CONTEXTSIZE){
			context[i+j*TD_SIZE]= default_val[i];
			//printf("Context: %d, index: %d,pos: %d, thread: %d, write: %.2f\n", j,i,i+j*TD_SIZE,warpTID, default_val[i]);
			j++;}
	i+=WARPSIZE;}
	__syncwarp();
	return 0;
      }
};


class ROB_d {
   public:
	ROB *rob;
       ROB_d(int Total_Trace){
       		//ROB rob[Total_Trace]; 		
		H_ERR(cudaMalloc((void **)&rob, sizeof(ROB)*(Total_Trace)));

       }	
};

__device__ void
dis(float *data, int size, int rows)
{
	for(int i=0;i<rows;i++)
	{
		for(int j=0; j<size;j++){
		printf("%.1f  ",data[i*size+j]);
		}
		printf("\n");
	}
}

__global__ void
preprocess(ROB_d *rob_d, float *insts,  float *factor, float *mean, float *default_val, float *inputPtr, float *train_data, params *param, int Total_Trace )
{
    
    int fetched=0, curTick=0, lastFetchTick=0;
    int TID=(blockIdx.x * blockDim.x) + threadIdx.x ;
    int warpID= TID/WARPSIZE;
    int warpTID = TID%WARPSIZE;
    int TotalWarp = (gridDim.x * blockDim.x) / WARPSIZE;
    int index,Total;
    ROB *rob;
    float *rob_pointer;
    float *input_ptr; 
#ifdef WARP	
    index= warpID;
    Total= TotalWarp;
#else
    index= blockIdx.x;
    Total= gridDim.x;
#endif
     while(index<Total_Trace){
     	rob = &rob_d->rob[index];
    //if(warpTID==0) { printf("Read: Warp: %d, assigned: %d, next: %d\n",warpID, index, index + Total);}
    //push new instruction to respective ROB but not latency
    int tail= rob->dec(tail);
    //if(warpTID==0) { printf("Read: Warp: %d, assigned: %d, next: %d\n",warpID, index, index + Total);}
    //int tail= rob->dec(tail);
    rob_pointer= insts + ROBSIZE * INST_SIZE * index;	
     float *input_Ptr = inputPtr + ML_SIZE * index;
    int i= warpTID+4; 
    while(i<INST_SIZE)
    {
	    rob_pointer[i]= train_data[i + warpID * INST_SIZE];
	    //printf("t: %d, i: %d, offset: %d\n",TID,i,train_offset);	
	    i+=WARPSIZE;		
    }
    __syncwarp();
    //rob = &rob_d->rob[TID];
    //if(warpTID==0) { printf("Inpt: %d\n",warpID);} 
    if(warpTID==0){
       	int retired = rob->retire_until(curTick, insts); 
    	//printf("Retired. \n");
	fetched++;	
    	rob->add();
	//printf("Update: ROB: %d, thread: %d, head:%d, tail: %d, newIndex: %d\n", index, threadIdx.x, rob->head, rob->tail, (index + gridDim.x * blockDim.x));
    	if (curTick != lastFetchTick) {
        	//rob->update_fetch_cycle(curTick - lastFetchTick, curTick, factor);
   	}
    }
    __syncwarp();
    //if(TID==0){printf("update completed\n"); }
    //rob = &rob_d->rob[index]; 
    //while(index<Total_Trace){
	//if(warpTID==0) { printf("Make input: Warp: %d, assigned: %d,offset: %d, next: %d\n",warpID, index,ML_SIZE*index, index + Total);}
    	rob->make_input_data(input_Ptr, rob_pointer, curTick, factor, default_val);        
    	index+= Total;  
	if(warpTID==0){
		printf("Input_Ptr\n");
		dis(input_Ptr,TD_SIZE,3 );
	}	
    }
}

__global__ void
update( ROB_d *rob_d, float* output, float* inputPtr, float* factor, float* mean ){

	      int TID=(blockIdx.x * blockDim.x) + threadIdx.x ;
      	      int offset = TID *2;
	      float fetch_lat = output[offset+0] * factor[1] + mean[1];
	      float finish_lat = output[offset+1] * factor[3] + mean[3];
	      int int_fetch_lat = round(fetch_lat);
	      int int_finish_lat = round(finish_lat);
	      if (int_fetch_lat < 0)
		int_fetch_lat = 0;
	      if (int_finish_lat < MIN_COMP_LAT)
		int_finish_lat = MIN_COMP_LAT;

            inputPtr = inputPtr + ML_SIZE * TID; 
	   inputPtr[0] = (-int_fetch_lat - mean[0]) / factor[0];
 	   inputPtr[1] = (-int_fetch_lat - mean[1]) / factor[1];
	   inputPtr[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
	   if (inputPtr[2] >= 9 / factor[2])
	   	inputPtr[2] = 9 / factor[2];
	   inputPtr[3] = (int_finish_lat - mean[3]) / factor[3];
	//newInst->tickNum = int_finish_lat;
	//newInst->completeTick = curTick[0] + int_finish_lat + int_fetch_lat;
	   //int lastFetchTick = curTick;
	   /*
	   if (total_num && fetched_inst_num[0] == total_num) {
		eof[0] = true;
		break;
		}
	if (int_fetch_lat) {
		nextFetchTick = curTick + int_fetch_lat;
		break;
		}


	ROB *rob = &rob_d->rob[TID];
	if (rob->is_empty())
	{

	
	}
	*/
}



void display(float *data, int size, int rows)
{
	for(int i=0;i<rows;i++){
		for(int j=0;j<size;j++){
			printf("%.2f\t",data[i*size+j]);
		}
		printf("\n");
	}
}

void display(unsigned long *data, int size, int rows)
{
	        for(int i=0;i<rows;i++){
			                for(int j=0;j<size;j++){
						                        printf("%.ld\t",data[i*size+j]);
									                }
					                printf("\n");
							        }
}



float *read_numbers(char *fname, int sz) {
  float *ret = new float[sz];
  ifstream in(fname);
  //printf("Trying to read from %s\n", fname);
  for(int i=0;i<sz;i++)
    in >> ret[i];
  return ret;
}

int read_trace_mem(char trace_file[], char aux_trace_file[], float *trace, unsigned long *aux_trace, int instructions)
{
  FILE *trace_f=fopen(trace_file,"rb");
  if(!trace_f){
	printf("Unable to read trace binary.");
	return 1;
	}
    int r=fread(trace,sizeof(float),TRACE_DIM*instructions,trace_f);
    printf("read :%d values for trace.\n",r);
    //display(trace,TRACE_DIM,2);

  FILE *aux_trace_f=fopen(aux_trace_file,"rb");
  if(!aux_trace_f){
        printf("Unable to aux_trace binary.");
        return 1;
        }
    int k=fread(aux_trace,sizeof(unsigned long),AUX_TRACE_DIM*instructions,aux_trace_f);  
    printf("read :%d values for aux_trace.\n",k);
    display(aux_trace,AUX_TRACE_DIM,2);
}

int main(int argc, char *argv[]) {
printf("args count: %d\n",argc);
#ifdef CLASSIFY
  if (argc != 8) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst> <Total trace>" << endl;
    return 0;
  }
#else
  if (argc != 7) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <variances> <Total trace> <#Insts>" << endl;
#endif
    return 0;
  } 
  int arg_idx=4;
  float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);
  for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i]; 
    }
  int Total_Trace= atoi(argv[arg_idx++]);
  int Instructions= atoi(argv[arg_idx++]);  
  
  float *trace;
  unsigned long *aux_trace;
  trace=(float*) malloc(TRACE_DIM*Instructions*sizeof(float));
  aux_trace=(unsigned long*) malloc(AUX_TRACE_DIM*Instructions*sizeof(unsigned long));
  read_trace_mem(argv[1],argv[2],trace,aux_trace,Instructions); 
  int Batch_size = Instructions / Total_Trace;
  int stop_flag, inst_num;
  //cout << "Batch size:  "<<Batch_size<<endl;
  //cout<<"Parameters read..\n";
 
   omp_set_num_threads(96);
   double measured_time = 0.0;
 
  ROB_d  *rob_d;
    Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  Tick Case4 = 0;
  Tick Case5 = 0;
  float *inputPtr;
  //ifstream *trace = new ifstream[Total_Trace];
  //ifstream *aux_trace = new ifstream[Total_Trace];
  Tick *curTick = new Tick[Total_Trace];
  Tick *nextFetchTick = new Tick[Total_Trace];
  Tick *lastFetchTick = new Tick[Total_Trace];
  int *index = new int[Total_Trace];
  int *inst_num_all = new int[Total_Trace];
  int *fetched_inst_num = new int[Total_Trace];
  int *fetched = new int[Total_Trace];
  int *ROB_flag = new int[Total_Trace];
  int *int_fetch_latency = new int[Total_Trace];
  int *int_finish_latency = new int[Total_Trace];
  bool *eof = new bool[Total_Trace];
  int total_num= 10000;
  float *trace_all[Total_Trace];
  unsigned long *aux_trace_all[Total_Trace];
  //printf("variable init\n");
  
#pragma omp parallel for
for(int i = 0; i < Total_Trace; i++) {
    curTick[i] = 0;
    nextFetchTick[i] = 0;
    lastFetchTick[i] = 0;
    inst_num_all[i] = 0;
    fetched_inst_num[i] = 0;
    fetched[i] = 0;
    eof[i] = false;
    int offset = i * Batch_size;
    trace_all[i]= trace + offset * TRACE_DIM;
    aux_trace_all[i]= aux_trace + offset * AUX_TRACE_DIM;
    //std::string line, line1;
    int number_of_lines = 0;
     }
 // printf("Allocated. \n");
  //return 0;
  float *factor_d, *default_val_d, *mean_d;
  float* train_data;
  //train_data= (float*) malloc(Total_Trace*TD_SIZE*sizeof(float));
  cudaHostAlloc((void**)&train_data, Total_Trace*INST_SIZE*sizeof(float),
		          cudaHostAllocDefault);
  //printf("before rob\n");
  //return 0;
  ROB_d rob=ROB_d(Total_Trace);
  //printf("rob allocated\n");
  //return 0;
  //cout<<"Rob tail: "<<rob.tail<<"\n"; 
  H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float)*ML_SIZE*Total_Trace));
  //printf("Total mem: %d\n",ML_SIZE*Total_Trace);
  H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB_d)));
  //H_ERR(cudaMalloc((void **)&insts, sizeof(float)*Total_Trace*ROB_SIZE*INST_SIZE));
  H_ERR(cudaMalloc((void **)&factor_d, sizeof(float)*(TD_SIZE)));
  H_ERR(cudaMalloc((void **)&mean_d, sizeof(float)*(TD_SIZE)));
  H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float)*(TD_SIZE)));
  H_ERR(cudaMemcpy(rob_d, &rob, sizeof(ROB_d), cudaMemcpyHostToDevice));
  H_ERR(cudaMemcpy(factor_d, &factor, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
  H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
  H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
  struct timeval start, end, total_start, total_end;
  //printf("Cuda allocated\n");
  //return 0;
  gettimeofday(&total_start, NULL);
  bool is_empty=true;
  bool is_full=false;
  bool saturated=false;
  Tick retired=0;
  Tick completeTick=0;
  //cout<<"Loop starting....\n";
  Inst Inst_;
  struct params Host, *Device;
  struct Inst *newInst;
  float* train_data_d, *insts;
  H_ERR(cudaMalloc((void **)&train_data_d, sizeof(float)*Total_Trace*INST_SIZE));
  H_ERR(cudaMalloc((void **)&insts, sizeof(float)*Total_Trace*ROBSIZE*INST_SIZE));
  H_ERR(cudaMalloc((void **)&Device, sizeof(params)));
  //printf("InputPtr: %d, Train:%d\n ",Total_Trace*ML_SIZE,Total_Trace*TD_SIZE); 
  //H_ERR(cudaMalloc((void **)&Inst_.train_data_d, sizeof(float)*TD_SIZE));
  while(stop_flag!=1){
   double st= wtime(); 
    #pragma omp parallel for
   for(int i=0; i< Total_Trace; i++){
    	// Retire instructions.
    	inst_num += retired;
    	int fetched = 0;
    	int int_fetch_lat;
    	//int i=0;
    	//cout<<"First loop.i:" <<i<<endl; 
    	Inst newInst(train_data);    
    	//double st=wtime();
    	//trace+=((i%512)*39);
	//if (!newInst.read_sim_data(trace[i], aux_trace[i], train_data, i)) {
          if(!newInst.read_sim_mem(trace_all[i],aux_trace_all[i],train_data,i)){
		eof[i] = true;
		cout<<"Inside 1st\n";
        	//rob->tail = rob->dec(rob->tail);
      	}
	  trace_all[i]+=TRACE_DIM;
	  aux_trace_all[i]+=AUX_TRACE_DIM;

      }	 
      //display(train_data,INST_SIZE,2);
      double check1= wtime();
      H_ERR(cudaMemcpy(train_data_d, train_data, sizeof(float)*Total_Trace*INST_SIZE, cudaMemcpyHostToDevice));
      double check2= wtime();
      /*
      for(int i=0; i<TD_SIZE;i++)
      {
	      printf("%.3f\t",Inst_.train_data[i]);
	      if(i%10==0)
		      printf("\n");
      }
      printf("calling gpu function\n");
	*/
        int block= Total_Trace/2;

        preprocess<<<1,32>>>(rob_d, insts,factor_d, mean_d, default_val_d,inputPtr,train_data_d, Device, Total_Trace);
	H_ERR(cudaDeviceSynchronize());		
      	double en= wtime(); 
	printf("%d, %.6f, %.6f, %.6f, %.6f\n", Total_Trace,(check1-st),(check2-check1),(en-check2),(en-st));
        return 0;	
	H_ERR(cudaMemcpy(&Host, Device, sizeof(params), cudaMemcpyDeviceToHost));
	//cout<<"Here\n";
	is_empty= Host.is_empty;
	is_full= Host.is_full;
	saturated= Host.saturated;
	//cout<<"Done\n";
	//return 0;
	 float output[]={1.5,3.20,0,0,0,0};
      	measured_time += (end.tv_sec - start.tv_sec) * 1000000.0 + end.tv_usec - start.tv_usec;
      //cout << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << "\n";
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  //trace[0].close();
  //aux_trace[0].close();
#ifdef RUN_TRUTH
  cout << "Truth" << "\n";
#endif
  cout << inst_num << " instructions finish by " << (curTick - 1) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << " " << Case4 << " " << Case5 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  cout << "Lat Model: " << argv[3] << "\n";
#endif
  return 0;
}

