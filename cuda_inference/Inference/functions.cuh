#ifndef FUNC_H
#define FUNC_H
#define Threadsize 128
#include "models.cuh" 
#include <cublas_v2.h>
#include <cuda_fp16.h>
// #define DEBUG
#define TILE_DIM 16
// #define HALF

__global__ void 
G_display(custom_t  *a, int row,int column)
{
		printf("\n");		
		for(int i=0; i<row; i++)
		{
            printf("Row Id: %d\t",i);
			for(int j=0;j<column;j++)
			{ 
				printf("%.4f,\t",a[i*column+j ]);
			}
			printf("\n");
		}		
}


__device__ void
d_display(custom_t  *a, int row,int column)
{
        printf("\n");		
        for(int i=0; i<row; i++)
        {
            printf("Row: %d\n",i);
            for(int j=0;j<column;j++)
            {
                printf("%.2f\t",a[i*column+j]);
            }
            printf("\n");
        }		
}


__global__ void
  matrix_sum_G(custom_t* result, custom_t* A, custom_t* B, int rowSize, int columnSize, int Relu, int batch_size)
{
    // int Threadsize= 1;
    int tid =threadIdx.x + blockIdx.x * blockDim.x;
    int offset = rowSize * columnSize;
    // for(int i=tid; i<(batch_size*offset); i+=(blockDim.x*gridDim.x))
    int i = tid;
    while(i<(batch_size*offset))
    {
        result[i]= A[i] + B[i%offset];
        if(Relu){
            #ifdef HALF
                if(__hlt(result[i],0)){result[i]=0;}
            #else
                if(result[i]<0){result[i]=0;}
            #endif
        }
        i+=(blockDim.x*gridDim.x);
    }
}

template<typename T>
__device__  void  dot_product(T* result, int result_start, T* input, int input_start, T* filter, int filter_start, int rowSize, int columnSize)
{

    // rowSize and columnSize is for kernel size
    // int Threadsize= 1;
    int tid =threadIdx.x + blockIdx.x * blockDim.x;
    int warpTid = threadIdx.x%32;
    // result_start = result_start * columnSize;
    int input_start_row_index = filter_start * columnSize;
    // B_start_column_index = B_start * columnSize;
    #ifdef DEBUG1
    if(warpTid==0){ printf("R_start: %d, A_start: %d, B_start: %d, Rsize: %d, Csize: %d\n",result_start,input_start,filter_start,rowSize,columnSize);}
    #endif
    int i=threadIdx.x;
    float temp=0;
    while( i<(rowSize* columnSize) )
    {
        int row = i/columnSize;
        int column = i% columnSize;
        int index_B = row * 2 + input_start + i;
        #ifdef DEBUG1
        if(warpTid==0){
        printf("R: %d, C: %d, index: %d, index_B: %d, ",row,column,i, index_B);
        printf("A: %.3f, B: %.3f\n",(filter[i+input_start_row_index]),input[index_B]);
        }
        #endif
        temp+= (filter[i+input_start_row_index] * input[index_B]);
        i+=blockDim.x;
    }
    
    atomicAdd(&result[result_start],temp);
    __syncthreads();
    if(threadIdx.x==0){if(result[result_start]<0){result[result_start]==0;}}
    
    #ifdef DEBUG1
    if(warpTid==0){printf("Result: %f, index: %d\n",result[result_start], result_start);}
    #endif
}

#ifdef HALF
int gpu_blas_mmul(cublasHandle_t &handle,const half *A, const half *B, half *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const half alf = __float2half(1);
    const half bet = __float2half(0);
    const half *alpha = &alf;
    const half *beta = &bet;  
    // cout<<"M: "<<m<<", K: "<<k<<", N: "<<n<<endl;
    // Do the actual multiplication
    int res = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // cout<<"Res: "<<res<<endl;
    return res;
}

#else
int gpu_blas_mmul(cublasHandle_t &handle,const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;  
    // cout<<"M: "<<m<<", K: "<<k<<", N: "<<n<<endl;
    // Do the actual multiplication
    int res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // cout<<"Res: "<<res<<endl;
    return res;
}
#endif

int gpu_strided_blas_mmul(cublasHandle_t &handle,const float *A, const float *B, float *C, const int m, const int k, const int n, const int batch_size)
{
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    const int strideA = m*k;
    const int strideB = 0;
    const int strideC = m*n;
    int res = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_size);
    return res;
}

__global__ void 
Convp(One_DConv *conv_p, custom_t *X, int batch_size)
{
    float A[94];
    int kl = conv_p -> kernel_size; 
    int in_ch = conv_p -> in_channel;
    int out_ch = conv_p -> out_channel;
    int out = conv_p -> out_column;
    int len = out_ch - in_ch + 1;
    int row = threadIdx.x/kl;
    int column = threadIdx.x%kl;
    int index = (row*context_length+column);
    int blockSize= 128/batch_size;
    if(batch_size>128){blockSize=1;}
    int blockStart=blockIdx.x * blockSize;
    int batch_id=blockIdx.x/blockSize; 
    int batch_block_id= blockIdx.x % blockSize;
    // if(threadIdx.x==0){printf("F_id: %d, kl: %d, in_ch: %d, out_ch: %d \n", blockIdx.x, kl,in_ch,out_ch);}
    // __syncthreads();


    while(batch_id < batch_size)
    {
        int X_batch_offset = batch_id * context_length * inst_length;
        #pragma unroll
        for(int i=0;i<(context_length-1);i++)
        {
            // if(threadIdx.x==1){printf("i: %d, X: %.2f\n",i,X[index+i]);}
            __syncthreads();
            A[i]= X[index +i];
        }
        /* filter implementation  */
        int filterId = batch_block_id;
        // Assign each filter for each block
        while(filterId<out_ch){
            // if(threadIdx.x==0){printf("F_id: %d\n", filterId);}
            // __syncthreads();
            // if(threadIdx.x==0){printf("B: %d, filterId: %d\n", blockIdx.x, filterId);}
            float temp=0;
            int filter_offset = filterId * kl * inst_length;
            // For each slide
            for(int j=0; j<(out);j++)
            {
                // printf("X: %d\n", (index + j));
                
                int result_start = filterId* out + j;
                conv_p->output[result_start] = 0;
                if(threadIdx.x<(in_ch*kl))
                {
                    float z = conv_p->W[threadIdx.x+filter_offset];
                    // If first column  
                    if (column==0){temp=A[0]; 
                        #ifdef DEBUG1
                        printf("Tid: %d, z_index: %d, Z: %.2f, A_index: %d, Reg_index: %d, A: %.2f\n",threadIdx.x, (index+filter_offset),z, index,0,temp );
                        #endif
                    }
                    else{temp = A[j];
                        #ifdef DEBUG1
                        printf("Tid: %d, z_index: %d, Z: %.2f, A_index: %d, Reg_index: %d, A: %.2f\n",threadIdx.x, (index+filter_offset), z, index +j,j,temp ); 
                        #endif
                    }
                    float res = temp * z;
                    atomicAdd(&conv_p->output[result_start], res );
                    // if(threadIdx.x==0){printf("Result_index: %d, Dot product: %.2f\n",result_start,conv_p->output[result_start]);}
                }
                __syncthreads();
                if(threadIdx.x==0){
                        // printf("Res_index: %d, Result: %.2f\n",result_start,conv->output[result_start]);
                        atomicAdd(&conv_p->output[result_start], conv_p->b[filterId]);
                        #ifdef HALF
                            if(__hlt(conv_p->output[result_start],0)){conv_p->output[result_start]=0;}
                        #else 
                            if(conv_p->output[result_start]<0){conv_p->output[result_start]=0;}
                        #endif
                }
                __syncthreads();
            }
            filterId+=gridDim.x;
        }
        batch_id+=gridDim.x;   // May create some problem later
    }
}
__global__ void
Conv_thread(One_DConv *conv, custom_t *X, int batch_size)
{

    float A[94];
    int kl = conv-> kernel_size; 
    int in_ch = conv-> in_channel;
    int out_ch = conv-> out_channel;
    int out = conv-> out_column;
    // int len = out_ch - in_ch + 1;
    int row = threadIdx.x/kl;
    int column = threadIdx.x%kl;
    int index = (row*context_length+column);
    int blockSize= 128/batch_size;
    if(batch_size>128){blockSize=1;}
    int blockStart=blockIdx.x * blockSize;
    // printf("Tid: %d, row: %d, column: %d, element: %d \n", threadIdx.x,row,column, (row*context_length+column));
    // if(threadIdx.x==0){printf("kl: %d, in_ch:%d, out_ch:%d, out:%d\n",kl,in_ch,out_ch,out);}
    // copy the data to thread registers
    int batch_id=blockIdx.x/blockSize; 
    int batch_block_id= blockIdx.x % blockSize;
    // if(threadIdx.x==0) {printf("blockId: %d, batch_id: %d, batchsize: %d, batch_block_id: %d\n",blockIdx.x, batch_id,batch_size,batch_block_id);}
    while(batch_id < batch_size)
    {
        int X_batch_offset = batch_id * context_length * inst_length;
        #pragma unroll
        for(int i=0;i<(context_length-1);i++)
        {
            // if(threadIdx.x==194){printf("i: %d\n",i);}
            // if((blockIdx.x ==0) && (threadIdx.x==195)){printf("Xoffset: %d, index: %d, final_ind:%d\n",X_batch_offset,index,(X_batch_offset + index +i));}
            A[i]= X[X_batch_offset + index +i];
        }   
        __syncthreads();
        /* filter implementation  */
        int filterId = batch_block_id;
        // if(threadIdx.x==0){printf("%d goes over: %d filter.\n",blockIdx.x,(filterId) );}
        // __syncthreads();
        // Assign each filter for each block
        while(filterId<out_ch){
            int filter_offset = filterId * kl * inst_length;
            int res_batch_offset = batch_id * (out * out_ch);
            custom_t temp=0;
            // For each slide
            for(int j=0; j<(out);j++)
            {
                // printf("X: %d\n", (index + j));
                int result_start = filterId* out + j + res_batch_offset;
                conv->output[result_start] = 0;
                int pos= threadIdx.x;
                while(pos<(in_ch*kl))
                {
                    temp = A[j]; 
                    custom_t z = conv->W[threadIdx.x+filter_offset];
                    // #ifdef HALF
                    //     float res;                        
                    //     half t = __float2half(temp);
                    //     half zz = __float2half(z);
                    //     res= __hmul(t,zz);
                    // #else
                        custom_t res = temp * z;
                    // #endif
                    atomicAdd(&conv->output[result_start], res );
                    pos+=blockDim.x;
                }
                __syncthreads();
                if(threadIdx.x==0){
                    atomicAdd(&conv->output[result_start], conv->b[filterId]);
                    #ifdef HALF
                    if(__hlt(conv->output[result_start],0)){conv->output[result_start]=0;}
                    #else
                    if(conv->output[result_start]<0){conv->output[result_start]=0;}
                    #endif
                    // printf("Res_index: %d, Result: %.4f\n",result_start,conv->output[result_start]);
                }
                __syncthreads();
                
            }
            filterId+=blockSize;
        }
        // if(threadIdx.x==0){d_display(conv->output, out_ch, out);}
        batch_id+=gridDim.x; // May create some problem later
    }    
}

__global__ void
Conv_thread_2(One_DConv *conv, One_DConv *conv_previous, int batch_size)
{
    float A[90];
    int tid = threadIdx.x + gridDim.x * blockDim.x;
    int kl = conv-> kernel_size; 
    int in_ch = conv-> in_channel;
    int out_ch = conv-> out_channel;
    int out = conv-> out_column;
    int prev_out = conv_previous-> out_column;
    int len = out_ch - in_ch + 1;
    int row = threadIdx.x/kl;
    int column = threadIdx.x%kl;
    int context_len = conv->out_column;
    int index = (row*prev_out+column);
    // int batch_id=blockIdx.x, blockSize=1, blockStart=blockIdx.x/blockSize;
    custom_t *X = conv_previous->output;
    int blockSize= 128/batch_size;
    if(batch_size>128){blockSize=1;}
    int blockStart=blockIdx.x * blockSize;
    int batch_id=blockIdx.x/blockSize; 
    int batch_block_id= blockIdx.x % blockSize;
        // printf("Tid: %d, row: %d, column: %d, element: %d \n", threadIdx.x,row,column, (row*context_length+column));
    // copy the data to thread registers
    // if(threadIdx.x==0){printf("in_cha: %d, Out_ch: %d,Prev_out: %d, out : %d \n",in_ch,out_ch,prev_out,out);}  
    // __syncthreads();
    // if(threadIdx.x==0){d_display(X,in_ch,prev_out);}
    __syncthreads();
    while(batch_id < batch_size)
    {    
        int X_batch_offset = batch_id * prev_out * in_ch ;
        #pragma unroll 
        for(int i=0;i<(prev_out-1);i++)
        {
            // if(threadIdx.x==6){printf(",T_id: %d, index: %d, i: %d\n",threadIdx.x,index,i);}
            A[i]= X[X_batch_offset + index +i];
        }   
        __syncthreads();
        /* filter implementation  */
        int filterId = batch_block_id;
        // __syncthreads();
        // if(threadIdx.x==0){printf("F_id: %d\n", filterId);}
        // Assign each filter for each block
        while(filterId<out_ch){
            int filter_offset = filterId * kl * in_ch;
            int res_batch_offset = batch_id * (out * out_ch);
            // if(threadIdx.x==0){printf("B: %d, filterId: %d, offset: %d\n", blockIdx.x, filterId,filter_offset);}
            custom_t temp=0;
            // For each slide
            for(int j=0; j<out;j++)
            {
                int result_start = filterId * out + j + res_batch_offset;
                conv->output[result_start] = 0;
                int pos = threadIdx.x;
                while(pos<(in_ch*kl))
                {
                    // printf("X: %d\n", (index + j));
                    temp = A[j]; 
                    custom_t z = conv->W[threadIdx.x+filter_offset];
                    // #ifdef HALF
                    //     float res;
                    //     half t = __float2half(temp);
                    //     half zz = __float2half(z);
                    //     res = __hmul(t,zz); 
                    custom_t res = __hmul(temp,z); 
                    atomicAdd(&conv->output[result_start], res ); // try to avoid atomic operation here
                    pos+=blockDim.x;
                }
                __syncthreads();
                if(threadIdx.x==0){
                    // Add bias
                    atomicAdd(&conv->output[result_start], conv->b[filterId]);
                    // Relu
                    #ifdef HALF
                    if(__hlt(conv->output[result_start],0)){conv->output[result_start]=0;}
                    #else
                    if(conv->output[result_start]<0){conv->output[result_start]=0;}
                    #endif
                    // printf("J: %d, Res_index: %d, Result: %.4f\n",j,result_start,conv->output[result_start]);
                }
                __syncthreads();
            }
            filterId+=blockSize;
        }
        // if(threadIdx.x==0){printf("\n******Output******\n");d_display(conv->output, out_ch, out);}
        batch_id+=gridDim.x;
    }
}





#endif  