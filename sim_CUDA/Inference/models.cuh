
#ifndef MODEL_H
#define MODEL_H
#include <random>
#include<iostream>
#include<fstream>
#include <cassert>
#include "header.h"
#include "herror.h"
#define inst_length 39
#define context_length 94
// #define inst_length 4
// #define context_length 4
// #define MODEL_DEBUG
 #define CNN3_MODEL

#include <string.h>


// void param_loader(CNN3 *model)
// {
//     FILE *conv1_w;
//     conv1_w = fopen("params/conv1_w.bin","rb");
//     if (!src)
// 	{
// 		printf("Unable to open file!");
// 		return 1;
// 	}

// }

void read_dimension(int64_t *dimensions, int shape)
{
    FILE *dims;
    #ifdef CNN3_MODEL
        dims = fopen("params/CNN3/dims.bin","rb");
    #else
        dims = fopen("params/CNN3_P/dims.bin","rb");
    #endif
    if (!dims)
	{
		printf("Unable to open dimension file!");
		exit(0);
    }
    int reads= fread(dimensions,sizeof(int64_t),shape,dims);
    // printf("%d items read.\n",reads); 
    // for(int i=0;i<shape;i++)
    // {
    //     printf("%d \n",dimensions[i]);
    // }
}

void read_parameters(custom_t *variable, char *filename, int len)
{
    FILE *param;
    std::string file = {filename};
    #ifdef CNN3_MODEL
        std::string path = std::string("params/CNN3/") + filename + std::string(".bin");
    #else
        std::string path = std::string("params/CNN3_P/") + filename + std::string(".bin");
    #endif    
        #ifdef MODEL_DEBUG
    printf("FIle path: %s\n",path.c_str());
    #endif
    param = fopen(path.c_str(),"rb");
    if (!param)
	{
		printf("Unable to open file!");
		exit(0);
    }
        int reads= fread(variable,sizeof(custom_t),len,param);
    #ifdef MODEL_DEBUG
    // printf("%d items read.\n",reads);
    #endif
    // for(int i=0;i<len;i++)
    // {
    //     printf("%.2f \t",variable[i]);
    // }
    // printf("\n");
}


class One_DConv
{
    public:
        int in_channel, out_channel, kernel_size, out_column; 
        custom_t *W, *output,*b;
        One_DConv(){};
        ~One_DConv(){};

        void init(int in_channel_, int out_channel_, int kernel_size_, int out_column_, int weight_dim, int bias_dim, int batch_size, char *W_name, char *B_name){
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<> dist(-1,1);
            in_channel = in_channel_;out_channel = out_channel_;kernel_size = kernel_size_;out_column = out_column_;
            #ifdef MODEL_DEBUG
                printf("\nIn_ch: %d, out_ch:%d, kl: %d, out_col: %d\n",in_channel,out_channel,kernel_size, out_column);
            #endif
            H_ERR(cudaMalloc((void **)&W, sizeof(custom_t)* kernel_size * in_channel*out_channel));
            H_ERR(cudaMalloc((void **)&output, sizeof(custom_t)* batch_size * out_column * out_channel));
            H_ERR(cudaMalloc((void **)&b, sizeof(custom_t)* out_channel));
            #ifdef MODEL_DEBUG
                printf("Name: %s, in: %d, out: %d, kl: %d, out_column: %d\n",W_name,in_channel,out_channel, kernel_size,out_column);
                printf("Conv. W_G: %d, W_C: %d, B_g: %d, B_c: %d \n",weight_dim, kernel_size * in_channel*out_channel,bias_dim,out_channel);
            #endif
            custom_t *H_w, *H_b;
            H_w= (custom_t *) malloc(kernel_size*in_channel*out_channel*sizeof(custom_t));
            H_b = (custom_t *) malloc(out_channel*sizeof(custom_t));
            read_parameters(H_w, W_name,weight_dim);
            read_parameters(H_b, B_name,bias_dim);
            // for(int i=0;i<(in_channel*(kernel_size)*out_channel);i+=1) 
            // {
            //     // H_W[i]= filter[i];
            //     H_w[i]= dist(mt); 
            //     // printf("H_W: %f, i: %d\n",H_W[i],i);
            // }
            H_ERR(cudaMemcpy(W,H_w,sizeof(custom_t )*kernel_size*in_channel*out_channel, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(b,H_b,sizeof(custom_t )*out_channel, cudaMemcpyHostToDevice));
            #ifdef MODEL_DEBUG
            printf("Layer creation successful.\n");
            #endif
        }
};

class FC
{
    public:
        custom_t *output, *W, *b;
        int in, out;
        
        FC(){};
        ~FC(){};
        void init(int in_, int out_, int weight_dim, int bias_dim, int batch_size, char *W_name, char *B_name){
            std::random_device rd;
            std::mt19937 mt(rd()); 
            std::uniform_real_distribution<> dist(-1,1);
            in = in_; out = out_;
            // printf("\nin: %d, out: %d\n",in,out);
            #ifdef MODEL_DEBUG
                printf(" G: %d, C: %d, bias_G: %d, bias_C: %d\n",weight_dim, in*out, bias_dim, out);
            #endif
            H_ERR(cudaMalloc((void **)&output, sizeof(custom_t) * batch_size * out));
            H_ERR(cudaMalloc((void **)&W, sizeof(custom_t) * in * out));
            H_ERR(cudaMalloc((void **)&b, sizeof(custom_t) * out));
            custom_t *H_w, *H_b;
            H_w=(custom_t *) malloc(in* out* sizeof(custom_t));
            H_b=(custom_t *) malloc( out* sizeof(custom_t));
            read_parameters(H_w, W_name,weight_dim);
            read_parameters(H_b, B_name,bias_dim);
            // for(int i=0;i<(in* out);i+=1) 
            // {
            //     // H_w[i]= dist(mt); 
            //     // H_W[i]= i; 
            // }
            H_ERR(cudaMemcpy(W,H_w,sizeof(custom_t )* in * out, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(b,H_b,sizeof(custom_t )* out, cudaMemcpyHostToDevice));
        }
};


class CNN3
{
    public:
    int out, ck1, ch1, ck2, ch2, ck3, ch3, f1, f1_input;
    int conv1_out, conv2_out, conv3_out, batch_size;
    One_DConv conv1; 
    One_DConv conv2;
    One_DConv conv3;
    FC fc1;
    FC fc2;
    ~CNN3(){};
    CNN3(int out_,int ck1_, int ch1_, int ck2_, int ch2_, int ck3_, int ch3_, int f1_,int batch_size_){
        // CNN3(int out_){
        out = out_;
        ck1 = ck1_;
        ch1 = ch1_; 
        ck2 = ck2_; ch2 = ch2_; ck3 = ck3_; ch3 = ch3_; f1 = f1_;
        int var_count= 10;
        conv1_out = context_length - ck1 +1; 
        conv2_out = conv1_out -ck2 +1;
        conv3_out = conv2_out - ck3 +1;
        f1_input = ch3 * (context_length - ck1 - ck2 - ck3 + 3);
        batch_size = batch_size_;
        // f1_input = 39;
        // printf("conv1: %d, conv2: %d, conv3: %d, f1_input: %d \n",conv1_out,conv2_out,conv3_out,f1_input);
        int64_t *dims = (int64_t*) malloc(var_count*sizeof(int64_t));
        read_dimension(dims,var_count);
        conv1.init(inst_length, ch1, ck1,conv1_out, dims[0], dims[1],batch_size,"conv1_w","conv1_b");
        conv2.init(ch1, ch2, ck2, conv2_out,dims[2], dims[3],batch_size,"conv2_w","conv2_b");
        conv3.init(ch2, ch3, ck3, conv3_out,dims[4],dims[5],batch_size, "conv3_w","conv3_b");
        fc1.init(f1_input,f1,dims[6],dims[7],batch_size, "fc1_w","fc1_b");
        fc2.init(f1, out,dims[8],dims[9],batch_size, "fc2_w","fc2_b");
    }  

};

class CNN3_P
{
    public:
    int out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1, f1_input;
    int conv1_out, conv2_out, conv3_out, convp_out;
    int var_count= 12;
    One_DConv conv_p; 
    One_DConv conv1; 
    One_DConv conv2;
    One_DConv conv3;
    FC fc1;
    FC fc2;
    int batch_size;
    ~CNN3_P(){};
    CNN3_P(int out_, int pc_,int ck1_, int ch1_, int ck2_, int ch2_, int ck3_, int ch3_, int f1_, int batch_size_){
        // CNN3(int out_){
        out = out_;
        ck1 = ck1_;
        ch1 = ch1_; 
        ck2 = ck2_; ch2 = ch2_; ck3 = ck3_; ch3 = ch3_; f1 = f1_;
        pc = pc_;
        batch_size = batch_size_;
        int64_t *dims = (int64_t*) malloc(var_count*sizeof(int64_t));
        read_dimension(dims,var_count);
        convp_out = context_length - 1; 
        conv1_out = convp_out - ck1 +1; 
        conv2_out = conv1_out - ck2 +1;
        conv3_out = conv2_out - ck3 +1;
        f1_input = ch3 * (context_length - ck1 - ck2 - ck3 + 3 -1);
        // f1_input = 39;
        // printf("CNN3_P from model.\n");
        // printf("conv1: %d, conv2: %d, conv3: %d, f1_input: %d \n",conv1_out,conv2_out,conv3_out,f1_input);
        conv_p.init(inst_length, pc, 2, convp_out,dims[0],dims[1],batch_size, "convp_w","convp_b");
        conv1.init(ch1, ch1, ck1,conv1_out,dims[2],dims[3],batch_size, "conv1_w","conv1_b");
        conv2.init(ch1, ch2, ck2, conv2_out,dims[4],dims[5],batch_size, "conv2_w","conv2_b");
        conv3.init(ch2, ch3, ck3, conv3_out,dims[6],dims[7],batch_size, "conv3_w","conv3_b");
        fc1.init(f1_input,f1,dims[8],dims[9],batch_size, "fc1_w","fc1_b");
        fc2.init(f1, out,dims[10],dims[11],batch_size, "fc2_w","fc2_b");
    }  
};
 
#endif 
