#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved Pair approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used.
 */


// implemented q dependend kernel function

__global__ void reduceUnrolling (double *g_idata, double *g_odata, unsigned int n, unsigned int q) //added int q
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * q + threadIdx.x; // q adapted idx

    // unroll analogous q
    if (idx + blockDim.x*(q-1) < n)
    {
      for (int i=1; i<q; i++)
      {
	g_idata[idx] += g_idata[idx + blockDim.x*i];
      }
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            g_idata[idx] += g_idata[idx + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 1024;   // initial block size
    int q = 2048;
    if(argc > 2)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
        q = atoi(argv[2]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(double);
    double *h_idata = (double *) malloc(bytes);
    double *h_odata = (double *) malloc(grid.x * sizeof(double));
    double *tmp     = (double *) malloc(bytes);

    // initialize the array
    int sign=1;
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = sign*((double)( rand() & 0xFF ));
        sign*=-1;
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    double gpu_sum = 0.0;

    // allocate device memory
    double *d_idata = NULL;
    double *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(double)));

      // kernel: reduceUnrolling optimized with q
    if (grid.x>1)
    {  
 
       dim3 gridq ((grid.x + 1)/q,1); // change grid dim due to q
       CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
       CHECK(cudaDeviceSynchronize());
       iStart = seconds();
       reduceUnrolling<<<gridq.x, block>>>(d_idata, d_odata, size,q); // call optimized kernel function w. q
       CHECK(cudaDeviceSynchronize());
       iElaps = seconds() - iStart;
       CHECK(cudaGetLastError());
       CHECK(cudaMemcpy(h_odata, d_odata, gridq.x * sizeof(double),
                        cudaMemcpyDeviceToHost));
       gpu_sum = 0;

       for (int i = 0; i < gridq.x; i++) gpu_sum += h_odata[i];

       printf("gpu Unrolling optimized w. q = %d  elapsed %f sec gpu_sum: %d <<<grid %d block "
              "%d>>>\n", q, iElaps, gpu_sum, gridq.x, block.x);
      }
    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
