#include <stdio.h>  
#include <cuda_runtime.h>  
//#include "helper_cuda.h"  
#include "wtime.h"
/* A very simple kernel function */
 __global__ void kernel(int *d_var) { d_var[threadIdx.x] += 10; } 
 
 int * host_p;  
 int * host_result;  
 int * dev_p;  
 
int main(void) {  
      int batchsize[]= {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072};
      for(int i=0;i<17;i++){
      int ns = batchsize[i]*5661;  
      int data_size = ns * sizeof(int);
      /* Allocate host_p as pinned memory */
        cudaHostAlloc((void**)&host_p, data_size, 
        cudaHostAllocDefault);  
      /* Allocate host_result as pinned memory */
      ( 
        cudaHostAlloc((void**)&host_result, data_size, 
        cudaHostAllocDefault) );  
      /* Allocate dev_p on the device global memory */
      ( 
        cudaMalloc((void**)&dev_p, data_size) );  
      
      /* Initialise host_p*/
      for (int j=0; j<ns; j++){  
           host_p[j] = j + 1;  
      }  
      
      /* Transfer data to the device host_p .. dev_p */
      double st= wtime();
      ( 
        cudaMemcpy(dev_p, host_p, data_size, cudaMemcpyHostToDevice) );
      double ed= wtime();
      //printf("cpy time: %.3f\n",ed-st);
      double measured= ed-st;
      double data_mem= (data_size)/(1024.0*1024.0);
        double throughput= (float)data_mem/measured;
	  printf("%d, %.3f, %.4f,%.3f,",ns/5661,data_mem,measured,throughput);
      /* Now launch the kernel... */
      kernel<<<1,  ns>>>(dev_p);  
      //getLastCudaError("Kernel error");
     
      st= wtime(); 
      /* Copy the result from the device back to the host */
      ( 
        cudaMemcpy(host_result, dev_p, data_size, cudaMemcpyDeviceToHost) );
      ed= wtime();
      measured= ed-st;
      throughput= (float)data_mem/measured;
      printf("%.3f\n",throughput);
      /* and print the result */
      /*
      for (int i=0; i<ns; i++){  
           printf("result[%d] = %d\n", i, host_result[i]);  
      }  
     */ 
      /*
       * Now free the memory!
       */
      ( cudaFree(dev_p) );  
      ( cudaFreeHost(host_p) );  
      ( cudaFreeHost(host_result) );  
      }
      return 0;  
 } 
