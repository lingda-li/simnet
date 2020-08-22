#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Inference/header.h"
#include "Inference/herror.h"
#include "Inference/wtime.h"
#include "Inference/models.cuh"
#include "Inference/functions.cuh"
using namespace std;

//#define CLASSIFY
//#define DEBUG
//#define VERBOSE
//#define RUN_TRUTH
//#define DUMP_ML_INPUT
#define NO_MEAN
#define GPU

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

#define inst_length 39
#define context_length 94
#define CTA
#define THREAD_REGISTER
using namespace std;

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
    0.13659586939272889
};

float mean[TD_SIZE];

float default_val[TD_SIZE];

struct Inst {
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
  bool read(ifstream &trace) {
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
  bool read_train_data(ifstream &trace, ifstream &aux_trace) {
    trace >> trueFetchClass >> trueFetchTick;
    trace >> trueCompleteClass >> trueCompleteTick;
    aux_trace >> pc;
    if (trace.eof()) {
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
  void dump(Tick tick) {
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
      }
      std::copy(insts[i].train_data, insts[i].train_data + TD_SIZE, context + num * TD_SIZE);
      num++;
    }
    for (int i = num; i < ROBSIZE; i++) {
      std::copy(default_val, default_val + TD_SIZE, context + i * TD_SIZE);
    }
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

float *read_numbers(char *fname, int sz) {
  float *ret = new float[sz];
  ifstream in(fname);
  printf("Trying to read from %s\n", fname);
  for(int i=0;i<sz;i++)
    in >> ret[i];
  return ret;
}

void CNN3_P_inference(cublasHandle_t &handle, CNN3_P *model_device,CNN3_P *model_host, float *output, float *X_device)
{
  int N_blocks = 128; int N_threads = 128; int status=1;
  Convp<<<N_blocks,196>>>(&model_device->conv_p, X_device);
  H_ERR(cudaDeviceSynchronize());
  Conv_thread_2<<<N_blocks,320>>>(&model_device->conv1, &model_device->conv_p);
  H_ERR(cudaDeviceSynchronize());     
  Conv_thread_2<<<N_blocks,320>>>(&model_device->conv2, &model_device->conv1);
  H_ERR(cudaDeviceSynchronize());
  Conv_thread_2<<<N_blocks,320>>>(&model_device->conv3, &model_device->conv2);
  H_ERR(cudaDeviceSynchronize());
  status = gpu_blas_mmul(handle, model_host->conv3.output, model_host->fc1.W, model_host->fc1.output,  1, model_host->f1_input, model_host->f1);
  H_ERR(cudaDeviceSynchronize());
  if(status!=0){printf("Error in FC1 Layer. Status: %d\n",status);}
  matrix_sum_G<<<N_blocks,128>>>(model_host->fc1.output, model_host->fc1.output, model_host->fc1.b, 1, model_host->f1_input,1);
  H_ERR(cudaDeviceSynchronize());
  status = gpu_blas_mmul(handle, model_host->fc1.output, model_host->fc2.W,  model_host->fc2.output, 1, model_host->f1, model_host->out);
  H_ERR(cudaDeviceSynchronize());
  if(status!=0){printf("Error in FC2 Layer. Status: %d\n",status);}
  matrix_sum_G<<<N_threads,128>>>(model_host->fc2.output, model_host->fc2.output, model_host->fc2.b, 1, model_host->out,0);
  H_ERR(cudaDeviceSynchronize());
}
 
void CNN3_inference(cublasHandle_t &handle, CNN3 *model_device, CNN3 *model_host, float *output, float *X_device)
{
  int N_blocks = 128; int N_threads = 128; int status = 1;
  // cout<<"CNN3 model\n";
  cout<<"Inference called"<<endl;
  Conv_thread<<<N_blocks,196>>>(&model_device->conv1, X_device);
  H_ERR(cudaDeviceSynchronize());
  Conv_thread_2<<<N_blocks,320>>>(&model_device->conv2,&model_device->conv1);
  H_ERR(cudaDeviceSynchronize());
  Conv_thread_2<<<N_blocks,320>>>(&model_device->conv3,&model_device->conv2);
  H_ERR(cudaDeviceSynchronize());
  G_display<<<1,1>>>(model_host->conv1.output,64,86);
  H_ERR(cudaDeviceSynchronize()); 
  status = gpu_blas_mmul(handle, model_host->conv3.output, model_host->fc1.W, model_host->fc1.output,  1, model_host->f1_input, model_host->f1);
  H_ERR(cudaDeviceSynchronize());
  if(status!=0){printf("Error in FC1 Layer. Status: %d\n",status);}
  matrix_sum_G<<<N_blocks,128>>>(model_host->fc1.output, model_host->fc1.output, model_host->fc1.b, 1, model_host->f1_input,1);
  H_ERR(cudaDeviceSynchronize());
  status = gpu_blas_mmul(handle, model_host->fc1.output, model_host->fc2.W,  model_host->fc2.output, 1, model_host->f1, model_host->out);
  H_ERR(cudaDeviceSynchronize());
  if(status!=0){printf("Error in FC2 Layer. Status: %d\n",status);}
  matrix_sum_G<<<N_threads,128>>>(model_host->fc2.output, model_host->fc2.output, model_host->fc2.b, 1, model_host->out,0);
  H_ERR(cudaDeviceSynchronize());
  
}



int main(int argc, char *argv[]) {
#ifdef CLASSIFY
  if (argc < 5) {
    cerr << "Usage: ./simulator <trace> <aux trace> <lat module> <class module> <variances (optional)>" << endl;
#else
  if (argc < 3) {
    cerr << "Usage: ./simulator <trace> <aux trace>  <variances (optional)> " << endl;
#endif
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  ifstream aux_trace(argv[2]);
  if (!aux_trace.is_open()) {
    cerr << "Cannot open auxiliary trace file.\n";
    return 0;
  }

  // H_ERR(cudaSetDevice(3));
  #ifdef CNN3_MODEL
        CNN3 *model_device;
        H_ERR(cudaMalloc((void **)&model_device, sizeof(CNN3)));
        printf("Size of CNN3: %d\n",sizeof(CNN3));
        CNN3 model_host( 2, 5, 64, 5, 64, 5, 256, 400);
        H_ERR(cudaMemcpy(model_device, &model_host, sizeof(CNN3), cudaMemcpyHostToDevice));
    #else
        CNN3_P *model_device;
        H_ERR(cudaMalloc((void **)&model_device, sizeof(CNN3_P)));
        printf("Size of CNN3_P: %d\n",sizeof(CNN3_P));
        CNN3_P model_host(2, 64, 5, 64, 5, 64, 5, 256, 400);
        H_ERR(cudaMemcpy(model_device, &model_host, sizeof(CNN3_P), cudaMemcpyHostToDevice));
    #endif
  
  
  int batch_size=1;
  custom_t *X_host, *X_device;
  X_host = (custom_t *) malloc(inst_length*batch_size*context_length*sizeof(custom_t));
  H_ERR(cudaMalloc((void **)&X_device, sizeof(custom_t)*inst_length*batch_size*context_length));
  cublasHandle_t handle;
  cublasCreate(&handle);
  float *varPtr = NULL;
#ifdef CLASSIFY
  if (argc > 5)
    varPtr = read_numbers(argv[5], TD_SIZE);
#else
  if (argc > 4)
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
    cout << default_val[i] << " ";
  }
  cout << "\n";
  // at::Tensor input = torch::ones({1, ML_SIZE});
  // float *inputPtr = input.data_ptr<float>();
  float *inputPtr = (float*) malloc(ML_SIZE*sizeof(float));

  unsigned long long inst_num = 0;
  double measured_time = 0.0;
  Tick curTick = 0;
  Tick lastFetchTick = 0;
  bool eof = false;
  struct ROB *rob = new ROB;
  Tick nextFetchTick = 0;
  Tick Case0 = 0;
  Tick Case1 = 0;
  Tick Case2 = 0;
  Tick Case3 = 0;

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);
  while(!eof || !rob->is_empty()) {
    // Retire instructions.
    int retired = rob->retire_until(curTick);
    inst_num += retired;
    //if (inst_num >= 10)
    //  break;
    //if (inst_num)
    //  cout << ".";
#ifdef DEBUG
    if (retired)
      cout << curTick << " " << retired << "\n";
#endif
    // Fetch instructions.
    int fetched = 0;
    int int_fetch_lat;
    while (fetched < FETCH_BANDWIDTH && !rob->is_full() && !eof) {
      Inst *newInst = rob->add();
      if (!newInst->read_train_data(trace, aux_trace)) {
        eof = true;
        rob->tail = rob->dec(rob->tail);
        break;
      }
      fetched++;
      newInst->inTick = curTick;
#ifdef RUN_TRUTH
      int int_finish_lat = newInst->trueCompleteTick;
      int_fetch_lat = newInst->trueFetchTick;
#else
      // Predict fetch and completion time.
      gettimeofday(&start, NULL);
      //input.data_ptr<c10::ScalarType::Float>();
      if (curTick != lastFetchTick) {
        rob->update_fetch_cycle(curTick - lastFetchTick);
      }
      rob->make_train_data(inputPtr);
#ifdef DUMP_ML_INPUT
      cout << input << "\n";
#endif

      gettimeofday(&end, NULL);
      float *X_host;
      X_host = (custom_t *) malloc(inst_length*batch_size*context_length*sizeof(custom_t));
    //   FILE *dims;
    //   dims = fopen("input.bin","rb");
    //   if (!dims)
    // {
    //   printf("Unable to open input file!");
    //   exit(0);
    //   }
    //   int reads= fread(X_host,sizeof(float),(inst_length*context_length),dims);
    //   printf("%d items read.\n",reads); 
      // for(int i=0;i<(inst_length*context_length);i++)
      // {
          // printf("%.5f, \t",X_host[i]);
      // }
      H_ERR(cudaMemcpy(X_device,inputPtr,sizeof(float)*inst_length*batch_size*context_length, cudaMemcpyHostToDevice));
      float *output= (float*) malloc(2*sizeof(float)); output[0] = 0; output[1] = 0; 
      #ifdef CNN3_MODEL
        CNN3_inference(handle, model_device, &model_host, output, X_device);  
      #else 
        CNN3_P_inference(handle, model_device, &model_host, output, X_device);
      #endif
      // at::Tensor output = lat_module.forward(inputs).toTensor();
      H_ERR(cudaMemcpy(output,model_host.fc2.output,sizeof(custom_t )*2, cudaMemcpyDeviceToHost));
      printf("Output: %.4f, %.4f\n",output[0],output[1]);
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
      float finish_lat = output[1] * factor[3] + mean[3];
      int_fetch_lat = round(fetch_lat);
      int int_finish_lat = round(finish_lat);
      // cout<<"Output: "<<int_fetch_lat<<", "<<int_finish_lat<<endl;
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
#endif
      newInst->tickNum = int_finish_lat;
      newInst->completeTick = curTick + int_finish_lat + int_fetch_lat;
      lastFetchTick = curTick;
      if (int_fetch_lat) {
        nextFetchTick = curTick + int_fetch_lat;
        break;
      }
      
    }
    
#ifdef DEBUG
    if (fetched)
      // cout << curTick << " f " << fetched << "\n";
#endif
    if (rob->is_full() && int_fetch_lat) {
      // Fast forward curTick to the next cycle when it is able to fetch and retire instructions.
      curTick = max(rob->getHead()->completeTick, nextFetchTick);
      Case0++;
    } else if (rob->is_full()) {
      // Fast forward curTick to retire instructions.
      curTick = rob->getHead()->completeTick;
      Case1++;
    } else if (int_fetch_lat) {
      // Fast forward curTick to fetch instructions.
      curTick = nextFetchTick;
      Case2++;
    } else {
      curTick++;
      Case3++;
    }
    int_fetch_lat = 0;
  }
  gettimeofday(&total_end, NULL);
  double total_time = total_end.tv_sec - total_start.tv_sec + (total_end.tv_usec - total_start.tv_usec) / 1000000.0;

  trace.close();
  aux_trace.close();
  cout << inst_num << " instructions finish by " << (curTick - 1) << "\n";
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
  cout << "Truth" << "\n";
#endif
  return 0;
}

