/*
//#include "argsParser.h"
#include "buffers.h"

//#include "common.h"

#include "logger.h"

#include "parserOnnxConfig.h"

#include "NvInfer.h"
*/
#include <NvInfer.h>
#include <memory>
//#include <buffers>
//#include <logger.h>
//#include <common.h>
//#include <parserOnnxConfig.h>
//#include <argsParser.h>
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

//using namespace std;

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
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //const auto explicitBatch= 16384 << (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    printf("explictBatch: %d\n",explicitBatch);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    
    if (builder->platformHasFastFp16())
    {
        //config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // we have only one image in batch
    builder->setMaxBatchSize(1);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
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
      struct timeval start, start1,  end; 
    std::string model_path(argv[1]);
    int batch_size = 1;
    TRTUniquePtr< nvinfer1::ICudaEngine > engine{nullptr};
    TRTUniquePtr< nvinfer1::IExecutionContext > context{nullptr};
    parseOnnxModel(model_path, engine, context);
    
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void*> buffers(engine->getNbBindings());
   
   printf("%d\n",buffers.end() - buffers.begin()); 

    printf("%d \n",engine->getNbBindings());
    for (size_t i = 0; i < engine->getNbBindings(); ++i){
    	auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
	printf("%d, %d\n",i,binding_size);
	cudaMalloc(&buffers[i], binding_size);

	if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
            printf("Is input. \n");
	    printf("%d\n",engine->getBindingDimensions(i));
	}

        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
	    printf("Is output\n");
        }
    
	}
   	if (input_dims.empty() || output_dims.empty()){
	    std::cerr << "Expect at least one input and one output for network\n";
	    return -1;
   	 }
    
    int total= 1024*64 * 5661;
float *ptr = (float*) malloc(total * sizeof(float));

    for (int j=0;j<total;j++)
    {
	    ptr[j]=0.05*(j%7); 
    }

    gettimeofday(&start1, NULL);
    cudaMemcpy(ptr,buffers[0],total,cudaMemcpyDeviceToHost);
    // Copy data to GPU
    //memcpy(&buffers[0], ptr, total*sizeof(float));
    // Memcpy from host input buffers to device input buffers
    // inference
    gettimeofday(&start, NULL);
    context->executeV2(buffers.data());
    
    gettimeofday(&end, NULL);    
    inf_only = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    inf= end.tv_sec - start1.tv_sec + (end.tv_usec - start1.tv_usec) / 1000000.0;
    //context->enqueueV2(buffers.data(), 0, nullptr); 
     //gettimeofday(&end, NULL);
    std::cout<<buffers[1]<<"\n";
    printf("%f, %f\n",inf_only, inf);
     
    for (void* buf : buffers){
	cudaFree(buf);
    }
    return 0;
}
