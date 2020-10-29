#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <torch/script.h> // One-stop header.

using namespace std;
//#define CLASSIFY
// #define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU
#define PREFETCH
#define ROB_ANALYSIS
// #define HALF
#define MAXSRCREGNUM 8
#define MAXDSTREGNUM 6
#define ROBSIZE 94
#define TICK_STEP 500.0
#define FETCH_BANDWIDTH 3
#define RETIRE_BANDWIDTH 4
#define TD_SIZE 39
#define ML_SIZE (TD_SIZE * ROBSIZE)
#define MIN_COMP_LAT 6
#define ILINEC_BIT 8
#define IPAGEC_BIT 13
#define DADDRC_BIT 17
#define DLINEC_BIT 18
#define DPAGEC_BIT 22
typedef long unsigned Tick;
typedef long unsigned Addr;

Tick Num = 0;

float factor[TD_SIZE] = {
    4.312227940421713,
    23.77553044744847,
    4.555194111127984,
    35.2434314352038,
    15.265764787317297,
    0.29948663397402053,
    0.03179227967763891,
    1.0,
    0.3734605619940018,
    20.161017665013475,
    0.01824348838297176,
    0.0182174151577969,
    0.018094302417033036,
    0.0034398387559557687,
    1.0,
    1.0,
    0.38473787518201347,
    0.04555131694662134,
    0.09744266378500231,
    0.12392501424650113,
    0.12391492952272068,
    0.1238875588320556,
    0.06099769717651739,
    1.0,
    1.0,
    24.855520819765943,
    26.09130907621545,
    24.95569646402784,
    9.621855610687513,
    5.900770232974813,
    4.979413184960278,
    4.993246078979618,
    0.5821067191490825,
    14.657204204150103,
    22.100692970367547,
    22.241366163707752,
    20.590947782553783,
    2.58817823205487,
    0.13659586939272889};

float mean[TD_SIZE];

float default_val[TD_SIZE];

struct Inst
{
  int op;
  int isMicroOp;
  int isCondCtrl;
  int isUncondCtrl;
  int isMemBar;
  int srcNum;
  int destNum;
  int srcClass[MAXSRCREGNUM];
  int srcIndex[MAXSRCREGNUM];
  int destClass[MAXDSTREGNUM];
  int destIndex[MAXDSTREGNUM];
  Tick inTick;
  Tick completeTick;
  Tick tickNum;
  float train_data[TD_SIZE];
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

  // Read one instruction.
  bool read(ifstream &trace)
  {
    //Num++;
    //if (Num > 1000000)
    //  return false;
    Tick tmp;
    trace >> op >> tmp >> tmp >> tmp;
    if (trace.eof())
      return false;
    trace >> srcNum;
    for (int i = 0; i < srcNum; i++)
      trace >> srcClass[i] >> srcIndex[i];
    trace >> destNum;
    for (int i = 0; i < destNum; i++)
      trace >> destClass[i] >> destIndex[i];
    assert(!trace.eof());
    return true;
  }

  bool read_train_data(ifstream &trace, ifstream &aux_trace)
  { 
    // cout <<"Current pointer: "<< trace.tellg()<<endl;
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof())
    {
      assert(aux_trace.eof());
      return false;
    }
    assert(trueCompleteTick >= MIN_COMP_LAT);
    for (int i = 4; i < TD_SIZE; i++)
      trace >> train_data[i];
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
    //cout << "\n";
    return true;
  }

  void dump(Tick tick)
  {
    cout << op << " " << tick - inTick << " " << tickNum << " ";
    cout << srcNum << " ";
    for (int i = 0; i < srcNum; i++)
      cout << srcClass[i] << " " << srcIndex[i] << " ";
    cout << destNum << " ";
    for (int i = 0; i < destNum; i++)
      cout << destClass[i] << " " << destIndex[i] << " ";
  }
};

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

  
  Inst *getHead() {
    return &insts[head];
  }
  void retire() {
    assert(!is_empty());
    head = inc(head);
  }
  int retire_until(Tick tick) {
    int retired = 0;
    while (!is_empty() && insts[head].completeTick <= tick &&
           retired < RETIRE_BANDWIDTH) {
      retire();
      retired++;
    }
    return retired;
  }
  void dump(Tick tick) {
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      insts[i].dump(tick);
    }
    cout << "\n";
  }
  void make_train_data(float *context) {
    int num = 0;
    Addr pc = insts[dec(tail)].pc;
    int isAddr = insts[dec(tail)].isAddr;
    Addr addr = insts[dec(tail)].addr;
    Addr addrEnd = insts[dec(tail)].addrEnd;
    Addr iwalkAddr[3], dwalkAddr[3];
    // for (int i = num; i < ROBSIZE; i++) {
    //   cout<<default_val[i]<<"\t";
    // }
    // cout<<endl;
    for (int i = 0; i < 3; i++) {
      iwalkAddr[i] = insts[dec(tail)].iwalkAddr[i];
      dwalkAddr[i] = insts[dec(tail)].dwalkAddr[i];
    }
    for (int i = dec(tail); i != dec(head); i = dec(i)) {
      if (i != dec(tail)) {
        // Update context instruction bits.
        insts[i].train_data[ILINEC_BIT] = insts[i].pc == pc ? 1.0 / factor[ILINEC_BIT] : 0.0;
        int conflict = 0;
        for (int j = 0; j < 3; j++) {
          if (insts[i].iwalkAddr[j] != 0 && insts[i].iwalkAddr[j] == iwalkAddr[j])
            conflict++;
        }

        // cout<<"Ilinec"<< insts[i].train_data[ILINEC_BIT] << "conflict: "<<conflict << endl;
        insts[i].train_data[IPAGEC_BIT] = (float)conflict / factor[IPAGEC_BIT];
        insts[i].train_data[DADDRC_BIT] = (isAddr && insts[i].isAddr && addrEnd >= insts[i].addr && addr <= insts[i].addrEnd) ? 1.0 / factor[DADDRC_BIT] : 0.0;
        insts[i].train_data[DLINEC_BIT] = (isAddr && insts[i].isAddr && (addr & ~0x3f) == (insts[i].addr & ~0x3f)) ? 1.0 / factor[DLINEC_BIT] : 0.0;
        conflict = 0;
        // cout<<"Train Data: "<<insts[i].train_data[IPAGEC_BIT]<<" "<<insts[i].train_data[DADDRC_BIT]<<" "<<insts[i].train_data[DLINEC_BIT]<<endl;
        if (isAddr && insts[i].isAddr)
          for (int j = 0; j < 3; j++) {
            if (insts[i].dwalkAddr[j] != 0 && insts[i].dwalkAddr[j] == dwalkAddr[j])
              conflict++;
          }
        insts[i].train_data[DPAGEC_BIT] = (float)conflict / factor[DPAGEC_BIT];
      }
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      num++;
    }
    for (int i = num; i < ROBSIZE; i++) {
      std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
    }
    // for (int i = num; i < ROBSIZE; i++) {
    //   cout<<default_val[i]<<"\t";
    // }
    // cout<<endl;
  }
  void update_fetch_cycle(Tick tick) {
    for (int i = dec(dec(tail)); i != dec(head); i = dec(i)) {
      insts[i].train_data[0] += tick / factor[0];
      if (insts[i].train_data[0] >= 9 / factor[0])
        insts[i].train_data[0] = 9 / factor[0];
      insts[i].train_data[1] += tick / factor[1];
      assert(insts[i].train_data[0] >= 0.0);
      assert(insts[i].train_data[1] >= 0.0);
    }
  }
};

float *read_numbers(char *fname, int sz)
{
  float *ret = new float[sz];
  ifstream in(fname);
  printf("Trying to read from %s\n", fname);
  for (int i = 0; i < sz; i++)
    in >> ret[i];
  return ret;
}

int main(int argc, char *argv[])
{
  // cout << "main function called. Argc:  " <<argc<< endl;
  if (argc < 4)
  {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <# parallel traces>" << endl;
  }
  ifstream trace_test(argv[1]);
  if (!trace_test.is_open())
  {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream aux_trace_test(argv[2]);
  if (!aux_trace_test.is_open())
  {
    cerr << "Cannot open auxiliary trace file.\n";
    return 0;
  }

  torch::jit::script::Module lat_module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    lat_module = torch::jit::load(argv[3]);
#ifdef GPU
    lat_module.to(torch::kCUDA);
#endif
  }

  catch (const c10::Error &e)
  {
    cerr << "error loading the model\n";
    return 0;
  }

  float *varPtr = NULL;
#ifdef CLASSIFY
  if (argc > 6)
    varPtr = read_numbers(argv[5], TD_SIZE);
#else
  if (argc > 5)
    varPtr = read_numbers(argv[4], TD_SIZE);
#endif
  if (varPtr)
    cout << "Use input factors.\n";

      for (int i = 0; i < TD_SIZE; i++) {
#ifdef NO_MEAN
    mean[i] = -0.0;
#endif
    if (varPtr)
      factor[i] = sqrtf(varPtr[i]);
    default_val[i] = -mean[i] / factor[i];
    // cout << default_val[i] << " ";
  }

  int Total_Trace = atoi(argv[4]);
  std::string line;
  int lines=0;
   while (std::getline(trace_test, line))
        ++lines;
    // std::cout << "Number of lines in text file: " << lines;
  int Total_instr = lines;
  int nGPU = 1;
  int Batch_size = Total_instr / Total_Trace;
  ifstream trace[Total_Trace];
  ifstream aux_trace[Total_Trace];
  Tick curTick[Total_Trace];
  Tick nextFetchTick[Total_Trace];
  Tick lastFetchTick[Total_Trace];
  int index[Total_Trace];
  unsigned long long inst_num_all[Total_Trace];
  int fetched[Total_Trace];
  int ROB_flag[Total_Trace];
  int int_fetch_latency[Total_Trace];
  int int_finish_latency[Total_Trace];
  bool eof[Total_Trace];
#ifdef PREFETCH
  char Trace_Buffer[Total_Trace][20000];
  char AuxTrace_Buffer[Total_Trace][20000];
#endif
#pragma omp parallel for
  for (int i = 0; i < Total_Trace; i++)
  {
    curTick[i] = 0;
    nextFetchTick[i] = 0;
    lastFetchTick[i] = 0;
    inst_num_all[i] = 0;
    fetched[i] = 0;
    eof[i] = false;
    int offset = i * Batch_size;
    std::string line;
    int number_of_lines = 0;
    #ifdef PREFETCH
    	trace[i].rdbuf()->pubsetbuf(Trace_Buffer[i], 20000);    
    	aux_trace[i].rdbuf()->pubsetbuf(AuxTrace_Buffer[i], 20000);
    #endif
    trace[i].open(argv[1]);
    while (std::getline(trace[i], line) && (number_of_lines < offset))
      ++number_of_lines;
    aux_trace[i].open(argv[2]);
    number_of_lines = 0;
    while (std::getline(aux_trace[i], line) && (number_of_lines < offset))
      ++number_of_lines;
    // cout <<endl<< "I:" << i << "\tSkipped: " << number_of_lines << endl;
    if (i == 0)
    {
      trace[0].seekg(0, trace[0].beg);
      aux_trace[0].seekg(0, aux_trace[0].beg);
    }
  }

  // float *inputPtr = input.data_ptr<float>();
  int i, count = 0, stop_flag=0;
  // cout<<"Batch size: "<< Batch_size <<endl;
  struct ROB *rob = new ROB[Total_Trace];
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;
  double measured_time = 0.0;
  struct timeval start, end, total_start, total_end, end_first, start_first;
  gettimeofday(&total_start, NULL);
  while (stop_flag != 1)
  {
    at::Tensor input = torch::ones({Total_Trace, ML_SIZE});
    int inference_count[nGPU];
    Inst **newInst;
    newInst = new Inst *[Total_Trace];    
    std::vector<at::Tensor> inputs_vec[nGPU];
    at::Tensor output[nGPU];
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++)
    {
      
      //at::Tensor input = torch::ones({1, ML_SIZE});
      //float *inputPtr = input.data_ptr<float>();
      // cout<<"I: "<<i<<" Pointer: "<<inputPtr<<endl;
      if (!eof[i] || !rob[i].is_empty())
      {
        // Retire instructions.
        if (ROB_flag[i])
        {
          int retired = rob[i].retire_until(curTick[i]);
          #ifdef DEBUG
          cout<<"Count: "<<count<<"Retired: "<<retired<<"Retired until: "<<inst_num_all[i]<<endl;
          #endif
          inst_num_all[i] += retired;
          fetched[i] = 0;
          #ifdef DEBUG
          cout << "************" << "Curtick: " <<curTick[i] << "Retired: " <<retired << "Is_full: " << rob->is_full() << "************"
               << "\n";
          #endif
        }
        if (fetched[i] < FETCH_BANDWIDTH && !rob[i].is_full() && !eof[i])
        {
          ROB_flag[i] = false;
          if (inst_num_all[i] >( Batch_size-1))
          {
            eof[i]= true;
            rob[i].head = (rob[i].tail);
            #ifdef DEBUG
            cout <<"Trace: "<<i<< " ,end of batch size inst_num_all. Is empty: "<< rob[i].is_empty() << endl;
            #endif
          }
          newInst[i] = rob[i].add();
          if (!newInst[i]->read_train_data(trace[i], aux_trace[i]))
          {
            eof[i]= true;
            #ifdef DEBUG
            cout <<"Trace: "<<i<<"Instr: "<<inst_num_all[i]<<"eof true from read_train_data"<<endl;
            #endif
            rob[i].tail = rob[i].dec(rob[i].tail);
          }
          
          fetched[i]++;
          if(fetched[i]>Batch_size){eof[i]= true;
          #ifdef DEBUG
          cout <<"Trace: "<<i<< " ,end of batch size from fetch." << endl;
          #endif
          }
          #pragma omp atomic
            count+=1;
          #ifdef DEBUG
          cout <<"T: "<<i<< "Fetched: " << fetched[i] << endl;
          #endif
          newInst[i]->inTick = curTick[i];
          #ifdef DEBUG
          cout<<"T: "<<i<<"newInst->inTick: "<< newInst[i]->inTick<< endl;
          #endif
          if (curTick[i] != lastFetchTick[i])
          {
            #ifdef DEBUG 
            cout <<"T: "<<i<< "Update fetch cycle: "<<(curTick[i] - lastFetchTick[i])<<endl;
            #endif
            rob[i].update_fetch_cycle(curTick[i] - lastFetchTick[i]);
          }
          // cout<<input<<endl;
          float *inputPtr = input.data_ptr<float>();
          inputPtr= inputPtr + ML_SIZE * i;
	  rob[i].make_train_data(inputPtr);
          // cout<<input<<endl;
          // Determine the GPU to push the result.
          // int GPU_ID = 0;
          int GPU_ID = (i+1)%nGPU;
          if ((fetched[i] == FETCH_BANDWIDTH) )
          {
            ROB_flag[i] = true; 
            #ifdef DEBUG
            cout<<"Rob flagged" <<endl;
            #endif
          }
        }
        else
        {
          ROB_flag[i] = true;
          #ifdef DEBUG
          cout << "Else condition" << endl;
          #endif
        }
#ifdef DEBUG
        cout << "Count:" << count << endl;
#endif
      }
    }
    gettimeofday(&start_first, NULL);
    i=0;
    // Parallel inference
    /************************************************************************************************/
    #pragma omp parallel for
    for(i=0;i<nGPU; i++){    
      if(inference_count[i]){
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.cuda());
        #ifdef ROB_ANALYSIS
	#else
	output[i] = lat_module.forward(inputs).toTensor();
        #endif
     	inference_count[i]=0;
      }
    }
    gettimeofday(&end_first, NULL);
    double total_time = end_first.tv_sec - start_first.tv_sec + (end_first.tv_usec - start_first.tv_usec) / 1000000.0;
    // cout<<"Rob time: "<<total_time<< endl;
    // break;
    measured_time += (end.tv_sec - start.tv_sec) * 1000000.0 + end.tv_usec - start.tv_usec;
    // cout<<output<<endl;
    // break;
    // Aggregate results
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++)
    { 
      if(!eof[i]){
      int GPU_ID = (i+1)%nGPU;
      //float fetch_lat = output[0][i][0].item<float>() * factor[1] + mean[1];
      //float finish_lat = output[0][i][1].item<float>() * factor[3] + mean[3];
      float fetch_lat =  2.5665* factor[1] + mean[1];
      float finish_lat =  3.3545* factor[3] + mean[3];
      // cout<<"fetch: "<<fetch_lat<<"finish: "<<finish_lat<<endl;
      int int_fetch_lat = round(fetch_lat);
      int int_finish_lat = round(finish_lat);
      if (int_fetch_lat < 0)
        int_fetch_lat = 0;            
      if (int_finish_lat < MIN_COMP_LAT)
        int_finish_lat = MIN_COMP_LAT;
#ifdef ROB_ANALYSIS
      int_finish_lat = newInst[i]->trueCompleteTick;
            int_fetch_lat = newInst[i]->trueFetchTick;
#endif
      // cout <<"Trace: "<<i<< "curtick: " <<curTick[i] << ", fetch latency: " << int_fetch_lat << ", finish latency: " << int_finish_lat << endl;
      #ifdef DEBUG
      #endif
      newInst[i]->train_data[0] = (-int_fetch_lat - mean[0]) / factor[0];
      newInst[i]->train_data[1] = (-int_fetch_lat - mean[1]) / factor[1];
      newInst[i]->train_data[2] = (int_finish_lat - MIN_COMP_LAT - mean[2]) / factor[2];
      if (newInst[i]->train_data[2] >= 9 / factor[2])
        newInst[i]->train_data[2] = 9 / factor[2];
      newInst[i]->train_data[3] = (int_finish_lat - mean[3]) / factor[3];
      newInst[i]->tickNum = int_finish_lat;
      newInst[i]->completeTick = curTick[i] + int_finish_lat + int_fetch_lat;
      lastFetchTick[i] = curTick[i];
      #ifdef DEBUG 
        cout<<"newInst update"<<endl;
        cout << newInst[i]->train_data[0] << " " << newInst[i]->train_data[1] << " " << newInst[i]->train_data[2] << " " << newInst[i]->train_data[3]<<" "<<newInst[i]->tickNum<<" "<<newInst[i]->completeTick<<" "<<nextFetchTick[i]<< endl;
      #endif
      int_fetch_latency[i] = int_fetch_lat;
      int_finish_latency[i] = int_finish_lat;
      if (int_fetch_lat)
      {
        nextFetchTick[i] = curTick[i] + int_fetch_lat;
        ROB_flag[i] = true;
        #ifdef DEBUG
        cout<<"continue"<<endl;
        #endif
      }
      }
    }
    
      /************************************************************************************************/
#pragma omp parallel for
    for (i = 0; i < Total_Trace; i++){
      if (!rob[i].is_empty()) // this results in 2075
      {

      if (ROB_flag[i])
      {
        if (rob[i].is_full() && int_fetch_latency[i])
        {
          // Fast forward curTick to the next cycle when it is able to fetch and retire instructions.
          curTick[i] = max(rob[i].getHead()->completeTick, nextFetchTick[i]);
        #pragma omp atomic 
          Case0+=1;
        }
        else if (rob[i].is_full())
        {
          // Fast forward curTick to retire instructions.
          curTick[i] = rob[i].getHead()->completeTick;
        #pragma omp atomic
          Case1++;
        }
        else if (int_fetch_latency[i])
        {
          // Fast forward curTick to fetch instructions.
          curTick[i] = nextFetchTick[i];
        #pragma omp atomic
          Case2++;
        }
        else
        {
          curTick[i]++;
        #pragma omp atomic
          Case3++;
        }
        #ifdef DEBUG
        cout<< "curTick: " << curTick[i]<<endl;
        #endif
        int_fetch_latency[i] = 0;
      }
    }
  }
    // stop_flag -= 1;
    stop_flag = true;
    for(int i=0 ; i< Total_Trace;i++)
    {
      // cout<< "eof[ " << i << "]= " << eof[i]<<endl;
      if(!eof[i] || !rob[i].is_empty())
      
      {
        stop_flag=false;
      }
    }
    #ifdef DEBUG
    for(int i=0 ; i< Total_Trace;i++)
    {
      cout<<"Trace:"<<i<<", Inst: "<< inst_num_all[i]<<"curTick: " << curTick[i] << "Rob status: "<< rob[i].is_empty()<< endl;
    }
    #endif
    // cout<<"Stop flag: "<<stop_flag<<endl;
    // #ifdef DEBUG
    // cout<<"Stop flag:"<<stop_flag<<endl<<endl;
    // #endif
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;
  
  int inst_num = 0;
  Tick curTick_final = 0;

  for (i = 0; i < Total_Trace; i++) 
  {
    // cout<<"Inst: "<<inst_num_all[i] <<". Tick: "<<curTick[i]<<endl;
    inst_num += inst_num_all[i];
    curTick_final += curTick[i];
    trace[i].close();
    aux_trace[i].close();
  }

  cout<<"Count: "<<count<<endl;
  cout << inst_num << " instructions finish by " << (curTick_final ) << "\n";
  cout << "Time: " << total_time << "\n";
  cout << "MIPS: " << inst_num / total_time / 1000000.0 << "\n";
  cout << "USPI: " << total_time * 1000000.0 / inst_num << "\n";
  cout << "Measured Time: " << measured_time / inst_num << "\n";
  cout << "Cases: " << Case0 << " " << Case1 << " " << Case2 << " " << Case3 << "\n";
  cout << "Trace: " << argv[1] << "\n";
#ifdef CLASSIFY
  cout << "Model: " << argv[3] << " " << argv[4] << "\n";
#else
  cout << "Lat Model: " << argv[3] << "\n";
#endif
#ifdef RUN_TRUTH
  cout << "Truth"
       << "\n";
#endif
  return 0;
}
