
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> 

cudaError_t matrixAdd(int *c, const int *a, const int *b, unsigned int sizeX, unsigned int sizeY);

__global__ void matrixAddKernel(int *g_odata, const int *g_idataA, const int *g_idataB)
{
	__shared__ int sdatai[16];
	__shared__ int sdataj[16];

    unsigned int bid = threadIdx.x;
	unsigned int tid = blockIdx.x;
	unsigned int id = bid*blockDim.x + tid;

	sdatai[id] = g_idataA[id];
	sdataj[id] = g_idataB[id];
	__syncthreads();

	g_odata[id] = sdatai[id] + sdataj[id];
}

int main()
{
    const int matrixSizeX = 4;
	const int matrixSizeY = 4;
	const int a[matrixSizeX*matrixSizeY] = 
		{ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 };
    const int b[matrixSizeX*matrixSizeY] = 
		{ 20, 30, 40, 50, 30, 40, 50, 60, 40, 50, 60, 70, 50, 60, 70, 80 };
    int c[matrixSizeX*matrixSizeY] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = matrixAdd(c, a, b, matrixSizeX, matrixSizeY);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixAdd failed!");
        return 1;
    }

    
	for (int i = 0; i < matrixSizeX; i++)
	{
		for (int j = 0; j < matrixSizeY; j++)
			printf("%d ",c[i*matrixSizeX+j]);
		printf("\n");
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	system("pause");

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t matrixAdd(int *c, const int *a, const int *b, unsigned int sizeX, unsigned int sizeY)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

	unsigned int num_blocks = sizeX;
	unsigned int num_threads = sizeY;
	unsigned int mem_size = sizeof(int)*sizeX*sizeY;
   
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, mem_size);
    cudaStatus = cudaMalloc((void**)&dev_a, mem_size);
    cudaStatus = cudaMalloc((void**)&dev_b, mem_size);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, mem_size, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, mem_size, cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(num_threads, 1, 1);
	matrixAddKernel<<<grid,threads,mem_size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixAddKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, mem_size, cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
