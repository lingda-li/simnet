#ifndef TRT_H
#define TRT_H
#include <fstream> 
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <NvInfer.h>
#include <NvOnnxParser.h>
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
    //printf("Model loaded\n");
}
#endif
