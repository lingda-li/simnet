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
#define WARPSIZE 32

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

struct Inst {
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
   
  bool read_sim_data(ifstream &trace, ifstream &aux_trace, float *train_d, int index) {
    cout<<train_d[0]<<endl;
	  cout<<"read started...\n";
    train_data= train_d;
    cout<<train_data[0]<<endl;
    cout<<"train_data_read\n";
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
      assert(aux_trace.eof());
      return false;
    }
    offset= TD_SIZE * index;
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
    assert(!trace.eof() && !aux_trace.eof());
    //cout << "in: ";
    //for (int i = 0; i < TD_SIZE; i++)
    //  cout << train_data[i] << " ";
    cout << "Read complete\n";
    //H_ERR(cudaMemcpy(train_data_d, train_data, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
    return true;
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
    	H_ERR(cudaMalloc((void **)&factor, sizeof(float)*(TD_SIZE)));
	H_ERR(cudaMalloc((void **)&mean, sizeof(float)*(TD_SIZE)));
	H_ERR(cudaMalloc((void **)&default_val, sizeof(float)*(TD_SIZE)));
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
     Inst *add() {
    assert(!is_full());
    int old_tail = tail;
    tail = inc(tail);
    //printf("index updated.\n");
    return &insts[old_tail];
  }

    __device__
    Inst *getHead() {
        return &insts[head];
      }

__device__ void
	retire(){
		assert(!is_empty());
		head= inc(head);
	}

 __device__
 int retire_until(Tick tick) {
	//printf("Retire...\n");
	int retired = 0;
	while (!is_empty() && insts[head].completeTick <= tick) {
		retire();
		retired++;
	}
	return retired;
 }


	  __device__
    void update_fetch_cycle(Tick tick, Tick curTick) {
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

__device__ 
	  int make_input_data(float *context, Tick tick) {
 	//printf("Here. Head: %d, Tail: %d\n",head,tail);

 	int TID= (blockIdx.x * blockDim.x) + threadIdx.x;
	//int warpID= TID / WARPSIZE;
	int  warpTID= TID % WARPSIZE;
	assert(!is_empty());
        saturated = false;
        Addr pc = insts[dec(tail)].pc;
        int isAddr = insts[dec(tail)].isAddr;
        Addr addr = insts[dec(tail)].addr;
        Addr addrEnd = insts[dec(tail)].addrEnd;
        __shared__ Addr iwalkAddr[3], dwalkAddr[3];
        if (warpTID==0){
	for (int i = 0; i < 3; i++) {
          iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
          dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
        }}
	//__syncwarps();
	int i=warpTID;
	while(i<TD_SIZE)
	{
		//context[i]=insts[dec(tail)].train_data[i];
		//i+=WARPSIZE;
		//printf("THread: %d, Copy: Source: train_data %d, Dest:context %d Value: %.3f, %.3f\n ",warpTID,i,dec(tail),context[i],insts[dec(tail)].train_data[i] );
		i+=WARPSIZE;
	}
	
	int start = dec(dec(tail));
	int end= dec(head);
        int num= end-start; 
	i= start - warpTID;
	//printf("WarpID: %d, i: %d\n", warpTID,i);
	while(i > end){  
	  //printf("ThreadID: %d, inst id: %d\n",warpTID, i);
	  if (insts[i].completeTick <= tick)
            continue;
          if (num >= CONTEXTSIZE) {
            saturated = true;
            return 0;
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
          //num++;
        i-=WARPSIZE;
	}
	__syncwarp();
	//printf("Adding default values.\n");
	
	i= warpTID;
	while (i<TD_SIZE){
        //for (int i = num; i < CONTEXTSIZE; i++) {
	//printf("thread: %d, i: %d\n",warpTID,i);
	int j= num;
	
	while(j< CONTEXTSIZE){
		context[i+j*TD_SIZE]= default_val[i];
		//printf("Context: %d, index: %d\n", j,i+j*TD_SIZE);
		j++;
	}

		i+=WARPSIZE;
	//std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
        }
	return 0;
      }
};

__device__ void
dis(float *data, int len)
{
	for(int i=0;i<len;i++)
	{
		printf("%.3f\t",data[i]);
		if(i%6==0){printf("\n"); }
	}
}

__global__ void
preprocess(ROB *rob, int fetched, int curTick, int lastFetchTick, float *inputPtr, params *param )
{
    int TID=(blockIdx.x * blockDim.x) + threadIdx.x ;
    int warpID= TID/WARPSIZE;
    int warpTID = TID%WARPSIZE;
    //printf("GPU func called\n");
    //rob->add();
    //printf("new inst added");
    if(warpTID==0){
    	int retired = rob->retire_until(curTick); 
    //printf("Retired:%d \n ", retired);
    	rob->tail = rob->dec(rob->tail);
	fetched++;
    //newInst->inTick = curTick;
    	rob->add();
    
    	//printf("Head: %d,Tail: %d\n",rob->head,rob->tail);
    	if (curTick != lastFetchTick) {
        	rob->update_fetch_cycle(curTick - lastFetchTick, curTick);
   	 }
    }
    __syncwarp();
    rob->make_input_data(inputPtr, curTick);
    param->fetched = fetched;
    param->is_full= rob->is_full();
    param->saturated= rob->saturated;
    param->is_empty= rob->is_empty();
    param->completeTick= rob->getHead()->completeTick;
    //if(threadIdx.x==0)	{dis(inputPtr,ML_SIZE);}    
}


float *read_numbers(char *fname, int sz) {
  float *ret = new float[sz];
  ifstream in(fname);
  printf("Trying to read from %s\n", fname);
  for(int i=0;i<sz;i++)
    in >> ret[i];
  return ret;
}

int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc != 8) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <class module> <variances> <# inst> <Total trace>" << endl;
#else
  if (argc != 7) {
    cerr << "Usage: ./simulator_q <trace> <aux trace> <lat module> <variances> <# inst>" << endl;
#endif
    return 0;
  }
  ifstream trace_test(argv[1]);
  if (!trace_test.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream aux_trace_test(argv[2]);
  if (!aux_trace_test.is_open()) {
    cerr << "Cannot open auxiliary trace file.\n";
    return 0;
  }
  int arg_idx=4;
  float *varPtr = read_numbers(argv[arg_idx++], TD_SIZE);
  unsigned long long total_num = atol(argv[arg_idx++]);
  cout<<"Total: "<<total_num<<endl;
  for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i];
    cout << default_val[i] << " ";
  }

  int Total_Trace = atoi(argv[arg_idx++]);

  std::string line;
  int lines = 0;
  while (std::getline(trace_test, line))
    ++lines;
  int Total_instr = lines;
  int Batch_size = Total_instr / Total_Trace;
  int stop_flag, inst_num;
  cout << "\n";
  cout<<"Parameters read..\n";
 
  double measured_time = 0.0;
 
  ROB rob, *rob_d;
    Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  Tick Case4 = 0;
  Tick Case5 = 0;
  float *inputPtr;
  ifstream *trace = new ifstream[Total_Trace];
  ifstream *aux_trace = new ifstream[Total_Trace];
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
    std::string line, line1;
    int number_of_lines = 0;
    trace[i].open(argv[1]);
    aux_trace[i].open(argv[2]);
    while (std::getline(trace[i], line) && std::getline(aux_trace[i], line1) &&
           (number_of_lines < offset))
      ++number_of_lines; 
    if (i == 0) {
      trace[0].seekg(0, trace[0].beg);
      aux_trace[0].seekg(0, aux_trace[0].beg);
    }
  }


  float *factor_d, *default_val_d, *mean_d;
  float *train_data= (float*) malloc(Total_Trace*TD_SIZE*sizeof(float));
  rob= ROB();
  cout<<"Rob tail: "<<rob.tail<<"\n"; 
  H_ERR(cudaMalloc((void **)&inputPtr, sizeof(float)*ML_SIZE));
  H_ERR(cudaMalloc((void **)&rob_d, sizeof(ROB)));

   H_ERR(cudaMalloc((void **)&factor_d, sizeof(float)*(TD_SIZE)));
   H_ERR(cudaMalloc((void **)&mean_d, sizeof(float)*(TD_SIZE)));
   H_ERR(cudaMalloc((void **)&default_val_d, sizeof(float)*(TD_SIZE)));

  H_ERR(cudaMemcpy(rob_d, &rob, sizeof(ROB), cudaMemcpyHostToDevice));
  H_ERR(cudaMemcpy(factor_d, &factor, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
   H_ERR(cudaMemcpy(default_val_d, &default_val, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mean_d, &mean, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  bool is_empty=true;
  bool is_full=false;
  bool saturated=false;
  Tick retired=0;
  Tick completeTick=0;
  cout<<"Loop starting....\n";
  Inst Inst_;
  struct params Host, *Device;
  struct Inst *newInst;
  H_ERR(cudaMalloc((void **)&newInst, sizeof(float)*Total_Trace*TD_SIZE));
  H_ERR(cudaMalloc((void **)&Device, sizeof(params)));
   //H_ERR(cudaMalloc((void **)&Inst_.train_data_d, sizeof(float)*TD_SIZE));
  while(stop_flag!=1){
   double st= wtime(); 
    
   for(int i=0; i< Total_Trace; i++){
    	// Retire instructions.
    	inst_num += retired;
    	int fetched = 0;
    	int int_fetch_lat;
    	//int i=0;
    	cout<<"First loop\n"; 
    	Inst *newInst;    
    	//double st=wtime();
    	if (!newInst->read_sim_data(trace[i], aux_trace[i], train_data, i)) {
        	eof[i] = true;
		cout<<"Inside 1st\n";
        	//rob->tail = rob->dec(rob->tail);
        	break;
      	}
    }
      H_ERR(cudaMemcpy(inputPtr, newInst, sizeof(float)*TD_SIZE, cudaMemcpyHostToDevice));
      /*
      for(int i=0; i<TD_SIZE;i++)
      {
	      printf("%.3f\t",Inst_.train_data[i]);
	      if(i%10==0)
		      printf("\n");
      }
      printf("calling gpu function\n");
	*/
        preprocess<<<1,32>>>(rob_d, fetched[0],curTick[0],lastFetchTick[0],inputPtr, Device);
	H_ERR(cudaDeviceSynchronize());		
      	double en= wtime(); 
	printf("Time: %.6f\n", en-st);
     	
	H_ERR(cudaMemcpy(&Host, Device, sizeof(params), cudaMemcpyDeviceToHost));
	cout<<"Here\n";
	is_empty= Host.is_empty;
	is_full= Host.is_full;
	saturated= Host.saturated;
	cout<<"Done\n";
	//return 0;
	 float output[]={1.5,3.20,0,0,0,0};
      	measured_time += (end.tv_sec - start.tv_sec) * 1000000.0 + end.tv_usec - start.tv_usec;
      //cout << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << "\n";
#ifdef CLASSIFY
      int f_class, c_class;
      for (int i = 0; i < 2; i++) {
        float max = cla_output[0][10*i].item<float>();
        int idx = 0;
        for (int j = 1; j < 10; j++) {
          if (max < cla_output[0][10*i+j].item<float>()) {
            max = cla_output[0][10*i+j].item<float>();
            idx = j;
          }
        }
        if (i == 0)
          f_class = idx;
        else
          c_class = idx;
     }
#endif
      float fetch_lat = output[0] * factor[1] + mean[1];
      float finish_lat = output[0] * factor[3] + mean[3];
      int int_fetch_lat = round(fetch_lat);
      int int_finish_lat = round(finish_lat);
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;
      if (int_finish_lat < MIN_COMP_LAT)
        int_finish_lat = MIN_COMP_LAT;
#ifdef CLASSIFY
      if (f_class <= 8)
        int_fetch_lat = f_class;
      if (c_class <= 8)
        int_finish_lat = c_class + MIN_COMP_LAT;
      //std::cout << curTick << ": ";
      //std::cout << " " << f_class << " " << fetch_lat << " " << int_fetch_lat << " " << newInst->trueFetchTick << " :";
      //std::cout << " " << c_class << " " << finish_lat << " " << int_finish_lat << " " << newInst->trueCompleteTick << '\n';
#endif
#ifdef DUMP_ML_INPUT
      int_finish_lat = newInst->trueCompleteTick;
      int_fetch_lat = newInst->trueFetchTick;
#endif
      newInst->train_data[0] = (-int_fetch_lat - mean[0]) / factor[0];
      newInst->train_data[1] = (-int_fetch_lat - mean[1]) / factor[1];
      newInst->train_data[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
      if (newInst->train_data[2] >= 9 / factor[2])
        newInst->train_data[2] = 9 / factor[2];
      newInst->train_data[3] = (int_finish_lat - mean[3]) / factor[3];
      newInst->tickNum = int_finish_lat;
      newInst->completeTick = curTick[0] + int_finish_lat + int_fetch_lat;
      lastFetchTick = curTick;
      if (total_num && fetched_inst_num[0] == total_num) {
        eof[0] = true;
        break;
      }
      if (int_fetch_lat) {
        nextFetchTick = curTick + int_fetch_lat;
        break;
      }
    
#ifdef DEBUG
    if (fetched)
      cout << curTick << " f " << fetched << "\n";
#endif
    if ((Host.is_full || Host.saturated) && int_fetch_lat) {
      // Fast forward curTick to the next cycle when it is able to fetch and retire instructions.
      curTick[0] = max(Host.completeTick, nextFetchTick[0]);
      if (Host.is_full)
        Case1++;
      else
        Case2++;
    } else if (int_fetch_lat) {
      // Fast forward curTick to fetch instructions.
      curTick[0] = nextFetchTick[0];
      Case0++;
    } else if (Host.is_full || Host.saturated) {
      // Fast forward curTick to retire instructions.
      //curTick = rob->getHead()->completeTick;
      if (Host.is_full)
        Case3++;
      else
        Case4++;
    } else {
      curTick++;
      Case5++;
    }
    int_fetch_lat = 0;
  
for (int i = 0; i < Total_Trace; i++) {
	      // cout<< "eof[ " << i << "]= " << eof[i]<<endl;
	if (!eof[i] || !Host.is_empty) {
	stop_flag = false;
	break;
	}
	   }

  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace[0].close();
  aux_trace[0].close();
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

