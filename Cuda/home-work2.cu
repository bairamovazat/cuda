//подключение библиотек
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
using namespace std;

#define N 1
#define M 200

__device__ void randomInt(float *i, float *j) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(1234, idx, 0, &state);
	float ranv1 = curand_uniform(&state);
	float ranv2 = curand_uniform(&state);
	*i = ranv1 * N;
	*j = ranv2 * N;
}

__global__ void monteCarloZuf(int* res1, int *res2) {
	float i, j;
	randomInt(&i, &j);
	if (sqrt(float(i*i+j*j))<N)// ïðîâåðêà óñëîâèÿ 
	{
		atomicAdd(res1, 1);
	}
	else {
		atomicAdd(res2, 1);
	}
}

int homeWork2() {
	int host_a;
	int host_b;
	int* dev_a;
	int* dev_b;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)& dev_a, sizeof(int));
	cudaMalloc((void**)& dev_b, sizeof(int));
	cudaMemset(dev_a, 0, sizeof(int));

	cudaEventRecord(start, 0);

	monteCarloZuf << <300,32 >> >(dev_a, dev_b);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTme: %f millseconds\n", KernelTime);

	cudaMemcpy(&host_a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
	float res = 4* host_a / float(host_a + host_b);
	printf("Value PI: %f\n", res);
	float error = abs(4 * atan(1) - res);
	printf("\nError: %f\n", error);

	//for (int i = 0; i < N; i++) { 
	//	if (host_a != N) 
	//		printf("error [%d] -> %d\n", i, host_a[i]); 
	//}

	// Ïðîâåðêà íà îøèáêó âûïîëíåíèÿ ïðîãðàììû íà äåâàéñå
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("%s ", cudaGetErrorString(err));
	cudaFree(dev_a);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
 }