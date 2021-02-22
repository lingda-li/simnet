
#include <NvInfer.h>
#include <memory>

#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>
#include <sys/time.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "wtime.h"
#include <iterator>
//#define  batch_size 1
using namespace std;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
	if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
	}
    }    
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context, int batch_size)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
     //TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetwork()};
    printf("Builder created\n");
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    printf("Network built\n");
    //auto profile = builder->createOptimizationProfile();
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 28);
    // use FP16 mode if possible
    
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // we have only one image in batch
    builder->setMaxBatchSize(batch_size);
    // generate TensorRT engine optimized for the target platform
    printf("Before engine and context\n");
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
    printf("Done parsing.\n");
    //context->getBindingDimensions(0);
    //printf("Context size: %d \n", context->getBindingDimensions(0)); 
    //std::cout<<context->getBindingDimensions(0)<<"\n";
}

std::string readBuffer(std::string const& path)
{
	string buffer;
	ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}



void serializer(TRTUniquePtr<nvinfer1::ICudaEngine>& engine)
{
    TRTUniquePtr<nvinfer1::IHostMemory> serializedModel{engine->serialize()};
    std::ofstream p("tensorrt_models/simnet1.engine",std::ios::binary);
    p.write((const char*)serializedModel->data(),serializedModel->size());
    p.close();
    cout<<"Serialized\n";
    //serializedModel->destroy();
}

void deseralizer(TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context, string model_path)
{
  std::string buffer= readBuffer(model_path);
    if(buffer.size()){
            TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
            engine.reset(runtime->deserializeCudaEngine(buffer.data(),buffer.size(),nullptr));
    }
    else{cout<<"couldn't read model.\n";}
    //engine.reset(engine);
    context.reset(engine->createExecutionContext());
    printf("Model loaded\n");
}

void load(){

}

void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}



int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx \n";
        return -1;
    }
      double inf = 0.0, inf_only= 0.0;
      //struct timeval start, start1,check1,end; 
    int batch_size= atoi(argv[2]);
    std::string model_path(argv[1]);
    TRTUniquePtr< nvinfer1::ICudaEngine > engine{nullptr};
    TRTUniquePtr< nvinfer1::IExecutionContext > context{nullptr};
    deseralizer(engine,context,model_path);
    //serializer(engine);
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    //parseOnnxModel(model_path, engine, context, batch_size); 
    printf("Max batch size for model: %d\n",engine->getMaxBatchSize());
    std::vector<void*> buffers(engine->getNbBindings());
    //printf("Context size: %d \t (Memory for model operations) \n", context->getBindingDimensions(0));
    //printf("NB bindings: %d \n",engine->getNbBindings());
    printf("Optimization profiles: %d\n",engine->getNbOptimizationProfiles());
    for (size_t i = 0; i < engine->getNbBindings(); ++i){
    	auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
	std::cout<<"Index: "<<i<<" Binding_size: "<<binding_size<< " Engine binding Dim 0: "<<engine->getBindingDimensions(i).d[0]<<" Dim 1: "<<engine->getBindingDimensions(i).d[1]<< "\n";
	cudaMalloc(&buffers[i], binding_size);
	
	if (engine->bindingIsInput(i)){
            input_dims.emplace_back(engine->getBindingDimensions(i));
            
	}
        else{
            output_dims.emplace_back(engine->getBindingDimensions(i));
	            }
    
	}
    	if (input_dims.empty() || output_dims.empty()){
	    std::cerr << "Expect at least one input and one output for network\n";
	    return -1;
   	 }
    float *inputBuffer, *outputBuffer;
    std::cout<<"Index 0: "<<engine->getBindingName(0)<<" Index 1: "<<engine->getBindingName(1)<<"\n";
    int total= batch_size * 5661;
    float *ptr = (float*) malloc(total * sizeof(float));
    float *result=  (float*) malloc(2 *batch_size* sizeof(float));
    for (int j=0;j<total;j++)
    {
	    ptr[j]=1; 
    }
    double st=wtime();
    cudaMemcpy(buffers[0],ptr,total*sizeof(float),cudaMemcpyHostToDevice);
    // Copy data to GPU
    cudaStreamSynchronize(0);
    double check1= wtime();
    context->enqueue(batch_size, buffers.data(), 0, nullptr); 
    cudaStreamSynchronize(0);
    double en=wtime();
    std::cout<< "Data: " << check1-st << " Inferece: " << en-check1 << endl;
    cudaMemcpy(result,buffers[1], 2 * batch_size, cudaMemcpyDeviceToHost);    
    printf("\n Result: \n");
    for(int j=0; j<(4*2);j++)
    {
	if((j>0) && (j%2==0)){ printf("\n");}
	printf("%.3f\t",result[j]);
	//if((j>0) && (j%2==0)){ printf("\n"); }
    }
    std::cout<<"\n";
    //printf("Time: %f, %f\n",inf_only, inf); 
    for (void* buf : buffers){
	cudaFree(buf);
    }
    return 0;
}
